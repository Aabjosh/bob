# seek.py
# Find a target with the camera and drive up to it using only the wheels (head stays centered).
# "Stop-and-look" visual servoing: YOLO takes ~1 s per frame, so move a short pulse, stop, look again.
#
#   SEARCH  no target in view -> spin in short pulses (toward where it was last seen)
#   ALIGN   target off-center  -> turn a pulse proportional to the bearing error
#   APPROACH aligned           -> drive about half the remaining distance, then re-check
#   ARRIVE  within STOP_DIST_M on two frames in a row
#
# Every constant below marked "calibrate" is a guess; wrong values only make it slower or
# wobblier, not broken, because it re-measures after every pulse.

import math
import time
import navigate as NavClass

STOP_DIST_M = 1.0          # stand-off distance from the target
ALIGN_DEG = 8              # bearing error we accept before driving
PATH_DEG = 12              # obstacles within this bearing count as "in the way"
OBSTACLE_STOP_M = 0.5      # don't drive forward with something this close in the path

TURN_DEG_PER_S = 45.0      # calibrate: how fast the body spins while a turn pulse runs
DRIVE_M_PER_S = 0.30       # calibrate: how fast it drives while a forward pulse runs
TURN_GAIN = 0.8            # command this fraction of the measured error (avoids overshoot)
DRIVE_FRACTION = 0.5       # drive this fraction of the remaining distance per pulse
MIN_TURN_MS, MAX_TURN_MS = 150, 1200
MIN_DRIVE_MS, MAX_DRIVE_MS = 250, 1500
SEARCH_TURN_MS = 400
SEARCH_MAX_DEG = 600       # assumed degrees to spin before giving up; well over 360 so a too-optimistic TURN_DEG_PER_S still completes a lap

CONFIRM_FRAMES = 2         # sightings in a row before we trust a detection
LOST_TOLERANCE = 2         # frames a target may flicker out before we start searching again
ARRIVE_FRAMES = 2          # near-enough frames in a row before we call it arrived
BLOCKED_GIVE_UP = 4        # blocked cycles before giving up


def _clamp(value, low, high):
    return max(low, min(high, value))


def _move(intent, ms):
    NavClass.send_motion(intent, ms)


def seek_target(cam, tag_id, deadline):
    """Returns "arrived", "not found", "blocked" or "timeout". Raises NavClass.Cancelled if interrupted."""
    seen = 0            # consecutive frames with the target in view
    lost = 0            # consecutive frames without it (after we had seen it)
    near = 0            # consecutive frames within STOP_DIST_M
    blocked = 0
    swept_deg = 0.0     # how far we've spun searching since we last saw it
    last_bearing = 0.0  # + = target was to the right
    ever_seen = False

    while time.time() < deadline:
        target, obstacles = cam.getImg(tag_id)
        obstacles = obstacles or []

        if target is None:
            seen = near = 0
            lost += 1
            if ever_seen and lost <= LOST_TOLERANCE:
                continue  # detections flicker; look again before spinning
            if swept_deg >= SEARCH_MAX_DEG:
                return "not found"
            _move("TURN_RIGHT" if last_bearing >= 0 else "TURN_LEFT", SEARCH_TURN_MS)
            swept_deg += TURN_DEG_PER_S * SEARCH_TURN_MS / 1000
            continue

        lost = 0
        swept_deg = 0.0
        seen += 1
        _, dist, x_offset = target
        bearing = math.degrees(math.atan2(x_offset, cam.focalLength))
        last_bearing = bearing
        ever_seen = True
        print(f"seek: bearing={bearing:+.1f} deg dist={dist:.2f} m seen={seen}")

        if seen < CONFIRM_FRAMES:
            continue  # one frame could be a false positive; look again without moving

        if dist <= STOP_DIST_M:
            near += 1
            if near >= ARRIVE_FRAMES:
                return "arrived"
            continue
        near = 0

        if abs(bearing) > ALIGN_DEG:
            ms = _clamp(abs(bearing) / TURN_DEG_PER_S * 1000 * TURN_GAIN, MIN_TURN_MS, MAX_TURN_MS)
            _move("TURN_RIGHT" if bearing > 0 else "TURN_LEFT", ms)
            continue

        in_path = [o for o in obstacles
                   if o[1] < min(OBSTACLE_STOP_M, dist)
                   and abs(math.degrees(math.atan2(o[2], cam.focalLength))) < PATH_DEG]
        if in_path:
            blocked += 1
            print(f"seek: path blocked ({blocked}/{BLOCKED_GIVE_UP})")
            if blocked >= BLOCKED_GIVE_UP:
                return "blocked"
            NavClass.sleep(1.0)
            continue
        blocked = 0

        ms = _clamp((dist - STOP_DIST_M) / DRIVE_M_PER_S * 1000 * DRIVE_FRACTION, MIN_DRIVE_MS, MAX_DRIVE_MS)
        _move("MOVE_FORWARD", ms)

    return "timeout"
