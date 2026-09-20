# Looks through the camera and says what it sees out loud ("I see a person and a chair.").
# Same Piper -> aplay pipeline as pi_audio.py, no mic or wake word needed.
#   python3 see_and_say.py          # once
#   python3 see_and_say.py --loop   # every 5 seconds until Ctrl+C
import subprocess
import sys
import time

import ClickClass as CC
import vision_router

PIPER_BIN = "/home/bob/piper/piper"
PIPER_MODEL = "/home/bob/piper/en_GB-alan-medium.onnx"

piper_proc = subprocess.Popen(
    [PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE
)
aplay_proc = subprocess.Popen(
    ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
    stdin=piper_proc.stdout
)


def say(line):
    print("SAY:", line)
    piper_proc.stdin.write((line + "\n").encode())
    piper_proc.stdin.flush()


cam = CC.Click()
loop = "--loop" in sys.argv
try:
    while True:
        detections = cam.detectAll(vision_router.MIN_CONF)
        if detections is None:
            say("I couldn't get an image from my camera.")
        else:
            say(vision_router.answer_list(detections))
        if not loop:
            time.sleep(4)  # let the speech finish before the script exits
            break
        time.sleep(5)
except KeyboardInterrupt:
    pass
