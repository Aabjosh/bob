# main.py
import math
import ClickClass as CC
import navigate as NavClass
import findObject as FO
import time
import seek

USE_HEAD_TRACKING = False  # True = the older head-centering + rotate-90 navigator below

BODY_90_TURN_STEPS = 4  # tune this: how many "turn right" pulses = 90° body rotation on real hardware

MAX_EMPTY_CYCLES = 4     # give up after this many consecutive cycles with no target in view
MAX_TASK_SECONDS = 180   # hard cap on one navigation task
HEAD_GAIN = 0.7          # fraction of the measured angle error corrected per head move
MUTE_AFTER_SECONDS = 4   # keep muting briefly: the ESP32's last quip arrives after we finish

# share navigate's serial port; opening /dev/ttyUSB0 twice makes two handles fight over reads
ser = NavClass.ser

# persistent objects, created once, reused across every voice command
cam = CC.Click()
nav = NavClass.Navigator()

# voice.py drops the ESP32's spoken quips while this is in the future (every head/wheel command makes Bob quip)
_mute_until = 0.0
_navigating = False

def _clean_for_esp32(text):
    """The ESP32 classifier reads "move your ..." as INTRO (Bob's ~40 s monologue) at 0.563 confidence,
    just over its 0.55 cutoff. The words after it ("head to the left") classify fine, so drop "your"."""
    words = [w for w in text.split() if w.lower().strip(",.!?") != "your"]
    return " ".join(words) or text

def is_muted():
    return time.time() < _mute_until

def cancel_navigation():
    """Called by voice.py when a new command arrives: abort a running navigation task."""
    if _navigating:
        NavClass.cancel.set()

def handle_instruction(instruction):
    """
    Called by the voice-command file with the raw instruction string.
    Tries to resolve an object ID; if found, runs the navigation pipeline
    until the target is reached (or we give up). If not found, forwards the
    raw string to serial as a fallback command.
    """
    global _mute_until, _navigating

    tagID = CC.getItemIndex(instruction)

    if tagID is None:
        # couldn't figure out an object from the instruction -> forward raw string
        ser.write((_clean_for_esp32(instruction) + "\n").encode())
        return

    NavClass.cancel.clear()
    _navigating = True
    _mute_until = float("inf")
    try:
        _navigate_to(tagID)
    except NavClass.Cancelled:
        print("navigation cancelled by a new command")
    finally:
        NavClass.cancel.clear()  # so the cleanup below isn't itself cancelled
        try:
            NavClass.center_head()
        finally:
            _navigating = False
            _mute_until = time.time() + MUTE_AFTER_SECONDS

def _navigate_to(tagID):
    if not USE_HEAD_TRACKING:
        result = seek.seek_target(cam, tagID, time.time() + MAX_TASK_SECONDS)
        print(f"seek finished: {result}")
        return
    _navigate_with_head(tagID)

def _navigate_with_head(tagID):
    # reset navigator state for a fresh task
    nav.avoid_state = None
    nav.avoid_direction = None
    nav.avoid_steps_left = 0
    nav.lastKnownTarget = None

    target, obstacles = FO.find_object(cam, tagID, NavClass.send_head_command)
    if target is None:
        print("giving up: never found the target")
        return

    empty_cycles = 0
    deadline = time.time() + MAX_TASK_SECONDS

    while time.time() < deadline:
        move = nav.decide(target, obstacles)
        print(f"nav: move={move!r} target={target} obstacles={len(obstacles or [])}")

        if move == "relocate target":
            last_offset = nav.lastKnownTarget[2] if nav.lastKnownTarget else None
            target, obstacles = NavClass.relocate_target(cam, tagID, last_offset)
            nav.lastKnownTarget = target
            move = nav.decide(target, obstacles)

        empty_cycles = empty_cycles + 1 if (target is None and move == "") else 0
        if empty_cycles >= MAX_EMPTY_CYCLES:
            print("giving up: lost the target")
            break

        if move == "rotate body 90 clockwise":
            for _ in range(BODY_90_TURN_STEPS):
                NavClass.send_body_command("turn right")
                time.sleep(0.3)
            NavClass.send_body_command("drive forward")

        elif move == "stop":
            NavClass.center_head()  # the firmware's STOP snaps the head to center instantly, so ease it there first
            NavClass.send_body_command("stop")
            break  # target reached, task complete -> return to caller

        elif move == "":
            pass  # nothing to send this cycle

        elif move.startswith("turn head"):
            # head commands are absolute on the ESP32, so step by the angle the target is off-center
            error_deg = math.degrees(math.atan(abs(target[2]) / cam.focalLength))
            NavClass.send_head_command(move, max(3, min(25, error_deg * HEAD_GAIN)))

        else:
            NavClass.send_body_command(move)  # "turn left" / "turn right" / "drive forward"

        time.sleep(0.5)
        target, obstacles = cam.getImg(tagID)
