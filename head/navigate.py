# navigate.py
import math
import threading
import time
import serial

STOP_DIST = 0.3            # meters
CENTERED_THRESHOLD = 40    # px
OBSTACLE_DANGER_DIST = 1.0 # meters
IN_PATH_THRESHOLD = 150    # px

MAX_SWEEP_STEPS = 6  # total head steps to try before giving up on relocating


class Navigator:
    def __init__(self):
        self.avoid_state = None       # None, "avoiding"
        self.avoid_direction = None
        self.avoid_steps_left = 0
        self.lastKnownTarget = None   # updated whenever target is actually seen

        self.TURN_STEPS = 2
        self.ADVANCE_STEPS = 2

    def decide(self, target, obstacles):
        obstacles = obstacles or []

        if target is not None:
            self.lastKnownTarget = target

        # --- mid-avoidance maneuver: commit regardless of target ---
        if self.avoid_state == "avoiding":
            self.avoid_steps_left -= 1
            if self.avoid_steps_left <= 0:
                self.avoid_state = None
                return "relocate target"
            return self.avoid_direction if self.avoid_steps_left > self.ADVANCE_STEPS else "drive forward"

        # --- check if a new avoidance maneuver needs to start ---
        blocking = [o for o in obstacles
                    if o[1] < OBSTACLE_DANGER_DIST and abs(o[2]) < IN_PATH_THRESHOLD]

        if blocking:
            closest = min(blocking, key=lambda o: o[1])
            _, dist, x_offset = closest
            self.avoid_direction = "turn right" if x_offset < 0 else "turn left"
            self.avoid_state = "avoiding"
            self.avoid_steps_left = self.TURN_STEPS + self.ADVANCE_STEPS
            return self.avoid_direction

        # --- no target visible ---
        if target is None:
            if self.lastKnownTarget is not None:
                return "relocate target"
            return ""

        # --- normal pursuit ---
        _, dist, x_offset = target

        if dist <= STOP_DIST:
            return "stop"

        # target not centered in camera -> turn head to find center first
        if abs(x_offset) > CENTERED_THRESHOLD:
            return "turn head right" if x_offset > 0 else "turn head left"

        # camera has target centered -> rotate BODY 90° clockwise so casters face it, then drive
        return "rotate body 90 clockwise"


def relocate_target(cam, tagID, last_known_offset):
    first_direction = "turn head right" if (last_known_offset or 0) > 0 else "turn head left"
    opposite_direction = "turn head left" if first_direction == "turn head right" else "turn head right"

    steps_each_way = MAX_SWEEP_STEPS // 2

    for _ in range(steps_each_way):
        send_head_command(first_direction)
        time.sleep(0.3)
        target, obstacles = cam.getImg(tagID)
        if target is not None:
            return target, obstacles

    for _ in range(steps_each_way):
        send_head_command(opposite_direction)
        time.sleep(0.3)

    for _ in range(steps_each_way):
        send_head_command(opposite_direction)
        time.sleep(0.3)
        target, obstacles = cam.getImg(tagID)
        if target is not None:
            return target, obstacles

    for _ in range(steps_each_way):
        send_head_command(first_direction)
        time.sleep(0.3)

    return None, obstacles


ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # adjust port + baud rate to match your hardware

class Cancelled(Exception):
    pass

# voice.py sets this (via instruct.cancel_navigation) when a new command arrives mid-task,
# so the robot stops what it's doing instead of finishing a long search first
cancel = threading.Event()

def _check_cancel():
    if cancel.is_set():
        raise Cancelled()

def sleep(seconds):
    """time.sleep that notices a cancel within 50 ms."""
    end = time.time() + seconds
    while True:
        _check_cancel()
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))

def send_body_command(cmd):
    _check_cancel()
    ser.write((cmd + "\n").encode())

MOTION_SETTLE_S = 0.5  # after a pulse ends: ESP32 latency + let the camera image settle

def send_motion(intent, duration_ms):
    """Timed wheel pulse via the firmware's PLAN format (intent: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD).
    Blocks until the pulse has finished, so the next camera frame is taken standing still."""
    _check_cancel()
    ser.write(f"PLAN {intent},90,{int(duration_ms)}\n".encode())
    sleep(duration_ms / 1000 + MOTION_SETTLE_S)

# The ESP32's head commands are absolute: "head left N degrees" puts the servo at 90 + N/2 and
# "head right N degrees" at 90 - N/2 (N is clamped to 0-180), and the servo jumps there at full
# speed. To move slower, send a run of small absolute steps from here, one every HEAD_STEP_INTERVAL_S.
HEAD_STEP_DEG = 12           # default sweep step, in servo degrees
HEAD_LIMIT_DEG = 60          # max servo travel each side of center
HEAD_SIGN = 1                # set to -1 if "turn head left" makes the camera look right
HEAD_SPEED_DEG_PER_S = 40    # head speed; lower = gentler (an SG90 free-runs at ~600)
HEAD_STEP_INTERVAL_S = 0.15  # gap between serial steps; every line costs the ESP32 a classify + a quip, so keep it sparse
head_angle = 0.0             # where the head is headed, servo degrees from center, positive = firmware "left"
_sent_angle = 0.0            # last angle actually sent to the servo

def _write_servo_angle(angle):
    value = int(round(abs(angle) * 2))
    side = "left" if angle >= 0 else "right"
    ser.write(f"head {side} {value} degrees\n".encode())

def _move_head_to(target):
    """Ramp the servo from where it is to `target`, blocking until it gets there."""
    global _sent_angle
    max_step = HEAD_SPEED_DEG_PER_S * HEAD_STEP_INTERVAL_S
    steps = math.ceil(abs(target - _sent_angle) / max_step)
    start = _sent_angle
    for i in range(1, steps + 1):
        _check_cancel()
        _sent_angle = start + (target - start) * i / steps
        _write_servo_angle(_sent_angle)
        time.sleep(HEAD_STEP_INTERVAL_S)

def send_head_command(cmd, degrees=None):
    global head_angle
    _check_cancel()
    step = HEAD_STEP_DEG if degrees is None else degrees
    if cmd == "turn head left":
        head_angle += HEAD_SIGN * step
    elif cmd == "turn head right":
        head_angle -= HEAD_SIGN * step
    else:
        ser.write((cmd + "\n").encode())
        return
    head_angle = max(-HEAD_LIMIT_DEG, min(HEAD_LIMIT_DEG, head_angle))
    _move_head_to(head_angle)

def center_head():
    global head_angle
    head_angle = 0.0
    _move_head_to(0.0)

def head_bearing_deg():
    """Where the head points relative to the body, in degrees; positive = right."""
    return -HEAD_SIGN * head_angle

def look_at_bearing(bearing_deg):
    """Turn the head to an absolute bearing (positive = right), limited to +/-HEAD_LIMIT_DEG."""
    global head_angle
    _check_cancel()
    head_angle = max(-HEAD_LIMIT_DEG, min(HEAD_LIMIT_DEG, -HEAD_SIGN * bearing_deg))
    _move_head_to(head_angle)
