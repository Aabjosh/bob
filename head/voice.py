# voice.py
# Runs on the Pi. Listens for a wake word, collects a spoken command, and hands the
# transcript to instruct.handle_instruction(), which decides whether to navigate to a
# detected object or forward the raw text to the ESP32 over serial.
# Also reads the ESP32's serial output and speaks STORY_START..STORY_DONE text through Piper.
#
# Run from this directory (ClickClass loads yolo26n.pt and the json files by relative path):
#   cd head && python voice.py

import queue, json, threading, subprocess, sounddevice as sd, time
from vosk import Model, KaldiRecognizer

import instruct  # loads YOLO + camera and opens the shared serial port via navigate.py

STARTERS = ["bro", "bob", "dude"]
ENDERS = ["now", "thank you"]
PIPER_BIN = "/home/bob/piper/piper"
PIPER_MODEL = "/home/bob/piper/en_GB-alan-medium.onnx"

def find_device():
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and 'lifecam' in d['name'].lower():
            return i
    return None

def process(text, active, buffer):
    low = text.lower()
    pos = 0
    sent_command = None
    while pos < len(text):
        if not active:
            hits = [(low.find(s, pos), s) for s in STARTERS if low.find(s, pos) != -1]
            if not hits:
                break
            idx, starter = min(hits, key=lambda x: x[0])
            active = True
            buffer = ""
            pos = idx + len(starter)
        else:
            hits = [(low.find(e, pos), e) for e in ENDERS if low.find(e, pos) != -1]
            if not hits:
                buffer = (buffer + " " + text[pos:]).strip()
                pos = len(text)
            else:
                idx, ender = min(hits, key=lambda x: x[0])
                buffer = (buffer + " " + text[pos:idx]).strip()
                sent_command = buffer
                buffer = ""
                active = False
                pos = idx + len(ender)
    return active, buffer, sent_command

# --- Persistent Piper -> aplay pipeline ---
piper_proc = subprocess.Popen(
    [PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE
)
aplay_proc = subprocess.Popen(
    ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
    stdin=piper_proc.stdout
)

speech_q = queue.Queue()
command_q = queue.Queue()
instruct.say = speech_q.put  # sentences Bob composes himself, e.g. "the person is right where I'm looking"

def speaker_worker():
    while True:
        line = speech_q.get()
        piper_proc.stdin.write((line + "\n").encode())
        piper_proc.stdin.flush()

def serial_listener(ser):
    in_story = False
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue
        quiet = instruct.is_muted()  # during navigation every head/wheel command makes Bob quip
        if not (quiet and (in_story or line in ("STORY_START", "STORY_DONE"))):
            print("ESP32:", line)
        if line == "STORY_START":
            in_story = True
        elif line == "STORY_DONE":
            in_story = False
        elif in_story and not quiet:
            speech_q.put(line)

def command_worker():
    # handle_instruction blocks until a navigation task finishes, so it runs here
    # instead of in the audio loop (otherwise mic audio backs up while the robot drives)
    while True:
        command = command_q.get()
        try:
            instruct.handle_instruction(command)
        except Exception as e:
            print(f"handle_instruction failed for {command!r}: {e}")

model = Model("model")
q = queue.Queue()
ser = instruct.ser
time.sleep(2)
active = False
buffer = ""

threading.Thread(target=serial_listener, args=(ser,), daemon=True).start()
threading.Thread(target=speaker_worker, daemon=True).start()
threading.Thread(target=command_worker, daemon=True).start()

def callback(indata, frames, time_, status):
    q.put(bytes(indata))

dev = find_device()
print(f"Using mic device index: {dev}")

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                        channels=1, device=dev, callback=callback):
    rec = KaldiRecognizer(model, 16000)
    print("Listening for wake word (bro/bob/dude)...")
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if not text:
                continue
            print("Heard:", text)
            active, buffer, command = process(text, active, buffer)
            if command:
                print(f"Handling instruction: {command}")
                instruct.cancel_navigation()  # a new command supersedes any navigation still running
                command_q.put(command)
            elif active:
                print("(listening for command...)")
