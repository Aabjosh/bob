# navigate.py
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

def send_body_command(cmd):
    ser.write((cmd + "\n").encode())

# The ESP32's head commands are absolute: "head left N degrees" puts the servo at 90 + N/2 and
# "head right N degrees" at 90 - N/2 (N is clamped to 0-180). Track the angle here so that
# "turn head left/right" behave as relative steps.
HEAD_STEP_DEG = 12    # default sweep step, in servo degrees
HEAD_LIMIT_DEG = 60   # max servo travel each side of center
HEAD_SIGN = 1         # set to -1 if "turn head left" makes the camera look right
head_angle = 0.0      # servo degrees from center, positive = what the firmware calls "left"

def _write_head_angle():
    value = int(round(abs(head_angle) * 2))
    side = "left" if head_angle >= 0 else "right"
    ser.write(f"head {side} {value} degrees\n".encode())

def send_head_command(cmd, degrees=None):
    global head_angle
    step = HEAD_STEP_DEG if degrees is None else degrees
    if cmd == "turn head left":
        head_angle += HEAD_SIGN * step
    elif cmd == "turn head right":
        head_angle -= HEAD_SIGN * step
    else:
        ser.write((cmd + "\n").encode())
        return
    head_angle = max(-HEAD_LIMIT_DEG, min(HEAD_LIMIT_DEG, head_angle))
    _write_head_angle()

def center_head():
    global head_angle
    head_angle = 0.0
    _write_head_angle()