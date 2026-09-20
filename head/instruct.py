# main.py
import ClickClass as CC
import navigate as NavClass
import locate
import time

MAX_TASK_SECONDS = 60    # hard cap on one search
MUTE_AFTER_SECONDS = 4   # keep muting briefly: the ESP32's last quip arrives after we finish
ANNOUNCE_DISTANCE = False  # also say "roughly N meters away" (the distance estimate is rough)

# share navigate's serial port; opening /dev/ttyUSB0 twice makes two handles fight over reads
ser = NavClass.ser

# persistent object, created once, reused across every voice command
cam = CC.Click()

# voice.py replaces this with the text-to-speech queue
say = print

# voice.py drops the ESP32's spoken quips while this is in the future (every head command makes Bob quip)
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
    """Called by voice.py when a new command arrives: abort a running search."""
    if _navigating:
        NavClass.cancel.set()

def handle_instruction(instruction):
    """
    Called by the voice-command file with the raw instruction string.
    If it names something the camera can detect (a YOLO class such as "person"), turn the head
    until that thing is centered in the frame and say where it is, leaving the head pointing at it.
    Otherwise forward the raw string to the ESP32 over serial as a command.
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
    keep_head_pointed = False
    try:
        name = CC.model.names[tagID]
        found = locate.locate_target(cam, tagID, time.time() + MAX_TASK_SECONDS, time.time)
        if found is None:
            say(f"I can't see a {name}.")
        else:
            bearing, dist, centered = found
            keep_head_pointed = True
            say(locate.describe(name, bearing, dist, centered, ANNOUNCE_DISTANCE))
    except NavClass.Cancelled:
        print("search cancelled by a new command")
    finally:
        NavClass.cancel.clear()  # so the cleanup below isn't itself cancelled
        try:
            if not keep_head_pointed:
                NavClass.center_head()
        finally:
            _navigating = False
            _mute_until = time.time() + MUTE_AFTER_SECONDS
