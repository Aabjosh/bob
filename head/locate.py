# locate.py
# Find an object by turning the head, get it centered in the frame, and report where it is.
# The robot never drives; only the head servo moves.
#
#   SCAN    step the head across its range, one frame per stop, until the target is seen twice in a row
#   CENTER  turn the head by the bearing error until the target sits in the middle of the frame
#   REPORT  head bearing + residual error = where the target is relative to the body

import math
import navigate as NavClass

SCAN_BEARINGS = (0, 35, 60, -35, -60)  # head bearings to look at, degrees, + = right. The camera sees ~90 deg wide, so stops overlap.
CENTER_DEG = 5                         # bearing error accepted as "centered"
CENTER_GAIN = 0.8                      # correct this fraction of the error per move (avoids overshoot)
CENTERED_FRAMES = 2                    # centered frames in a row before we trust it
MAX_CENTER_STEPS = 12
LOST_TOLERANCE = 2                     # frames the target may flicker out while centering
SETTLE_S = 0.3                         # let the head stop shaking before taking a frame


def _bearing(cam, target):
    return math.degrees(math.atan2(target[2], cam.focalLength))  # + = right of the camera axis


def _look(cam, tag_id):
    NavClass.sleep(SETTLE_S)
    return cam.getImg(tag_id)[0]


def _sighting(cam, tag_id):
    """Target seen in 2 of up to 3 frames (one flickered detection is tolerated, one stray hit is not)."""
    seen = None
    count = 0
    for frame in range(3):
        target = _look(cam, tag_id)
        if target is not None:
            count += 1
            seen = target
        if count >= 2:
            return seen
        if frame == 1 and count == 0:
            return None  # two empty frames: nothing at this stop
    return None


def _scan(cam, tag_id, deadline, now):
    """Return the first confirmed sighting (a target), or None after a full sweep."""
    for bearing in SCAN_BEARINGS:
        if now() > deadline:
            return None
        NavClass.look_at_bearing(bearing)
        target = _sighting(cam, tag_id)
        if target is not None:
            return target
    return None


def locate_target(cam, tag_id, deadline, now):
    """Returns (bearing_deg, dist_m, centered) for the target relative to the robot's body (+ = right),
    with the head left pointing as close to it as it can turn, or None if it was never found.
    centered is False when the target is beyond the head's turning range."""
    target = _scan(cam, tag_id, deadline, now)
    if target is None:
        return None

    centered = 0
    lost = 0
    for _ in range(MAX_CENTER_STEPS):
        if now() > deadline:
            break
        if target is None:
            lost += 1
            if lost > LOST_TOLERANCE:
                return None
            target = _look(cam, tag_id)
            continue
        lost = 0

        error = _bearing(cam, target)
        head = NavClass.head_bearing_deg()
        print(f"locate: head={head:+.0f} deg, target {error:+.1f} deg off-center, dist={target[1]:.2f} m")

        if abs(error) <= CENTER_DEG:
            centered += 1
            if centered >= CENTERED_FRAMES:
                return head + error, target[1], True
        else:
            centered = 0
            NavClass.look_at_bearing(head + CENTER_GAIN * error)
            if abs(NavClass.head_bearing_deg() - head) < 0.5:
                # the head is at its limit and can't turn further; report what we have
                return head + error, target[1], False
        target = _look(cam, tag_id)

    if target is None:
        return None
    error = _bearing(cam, target)
    return NavClass.head_bearing_deg() + error, target[1], abs(error) <= CENTER_DEG


def describe(name, bearing_deg, dist_m=None, centered=True, with_distance=False):
    """One sentence for Bob to say."""
    side = "right" if bearing_deg > 0 else "left"
    degrees = int(round(abs(bearing_deg) / 5.0) * 5)
    if not centered:
        return f"The {name} is about {degrees} degrees to my {side}, past how far I can turn my head."
    where = "straight ahead of me" if abs(bearing_deg) < 10 else f"about {degrees} degrees to my {side}"
    text = f"The {name} is right where I'm looking, {where}"
    if with_distance and dist_m is not None:
        text += f", roughly {dist_m:.1f} meters away"
    return text + "."
