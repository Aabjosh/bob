# main.py
import head.ClickClass as CC
import navigate as NavClass
import findObject as FO
import time
import serial

BODY_90_TURN_STEPS = 4  # tune this: how many "turn right" pulses = 90° body rotation on real hardware

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# persistent objects, created once, reused across every voice command
cam = CC.Click()
nav = NavClass.Navigator()

def handle_instruction(instruction):
    """
    Called by the voice-command file with the raw instruction string.
    Tries to resolve an object ID; if found, runs the navigation pipeline
    until the target is reached. If not found, forwards the raw string
    to serial as a fallback command.
    """
    tagID = CC.getItemIndex(instruction)

    if tagID is None:
        # couldn't figure out an object from the instruction -> forward raw string
        ser.write((instruction + "\n").encode())
        return

    # reset navigator state for a fresh task
    nav.avoid_state = None
    nav.avoid_direction = None
    nav.avoid_steps_left = 0
    nav.lastKnownTarget = None

    target, obstacles = FO.find_object(cam, tagID)

    while True:
        move = nav.decide(target, obstacles)

        if move == "relocate target":
            last_offset = nav.lastKnownTarget[2] if nav.lastKnownTarget else None
            target, obstacles = NavClass.relocate_target(cam, tagID, last_offset)
            nav.lastKnownTarget = target
            move = nav.decide(target, obstacles)

        if move == "rotate body 90 clockwise":
            for _ in range(BODY_90_TURN_STEPS):
                NavClass.send_body_command("turn right")
                time.sleep(0.3)
            NavClass.send_body_command("drive forward")

        elif move == "stop":
            NavClass.send_body_command("stop")
            break  # target reached, task complete -> return to caller

        elif move == "":
            pass  # nothing to send this cycle

        elif move.startswith("turn head"):
            NavClass.send_head_command(move)

        else:
            NavClass.send_body_command(move)  # "turn left" / "turn right" / "drive forward"

        time.sleep(0.5)
        target, obstacles = cam.getImg(tagID)