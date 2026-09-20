# findObject.py
# Initial search routine: look left, then right, to find the target object.
# Locks on (stops searching) as soon as the target is detected.

import time

SEARCH_STEPS = 3  # how many head-turn steps to take in each direction before giving up

def find_object(cam, tagID, send_head_command):
    # look left first
    for _ in range(SEARCH_STEPS):
        send_head_command("turn head left")
        time.sleep(0.3)  # let the head actually move before capturing
        target, obstacles = cam.getImg(tagID)
        if target is not None:
            print("found target while looking left")
            return target, obstacles

    # not found on the left — recenter before checking right
    for _ in range(SEARCH_STEPS):
        send_head_command("turn head right")
        time.sleep(0.3)

    # now sweep right from center
    for _ in range(SEARCH_STEPS):
        send_head_command("turn head right")
        time.sleep(0.3)
        target, obstacles = cam.getImg(tagID)
        if target is not None:
            print("found target while looking right")
            return target, obstacles

    # recenter again since we ended up fully right
    for _ in range(SEARCH_STEPS):
        send_head_command("turn head left")
        time.sleep(0.3)

    print("target not found during initial search")
    return None, obstacles