# main.py
import head.ClickClass as CC
import navigate as NavClass
import findObject as FO
import time

BODY_90_TURN_STEPS = 4  # tune this: how many "turn right" pulses = 90° body rotation on real hardware

def main():
    cam = CC.Click()
    nav = NavClass.Navigator()
    tagID = CC.getItemIndex("go to the chair")

    # --- initial search: look left, then right, to find the target ---
    target, obstacles = FO.find_object(cam, tagID)

    # --- main loop ---
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

        elif move == "":
            pass  # nothing to send this cycle

        elif move.startswith("turn head"):
            NavClass.send_head_command(move)

        else:
            NavClass.send_body_command(move)  # "turn left" / "turn right" / "drive forward" / "stop"

        time.sleep(0.5)

        # get fresh detections for the next cycle
        target, obstacles = cam.getImg(tagID)

if __name__ == "__main__":
    main()