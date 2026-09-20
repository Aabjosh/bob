import queue, json, threading, subprocess, sounddevice as sd, serial, time
from vosk import Model, KaldiRecognizer
import vision_router  # camera questions ("what do you see", ...) are answered here, not by the ESP32

STARTERS = ["bro", "bob", "dude"]
ENDERS = ["now", "thank you"]
SERIAL_PORT = "/dev/ttyUSB0"
BAUD = 115200
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
        print("ESP32:", line)
        if line == "STORY_START":
            in_story = True
        elif line == "STORY_DONE":
            in_story = False
        elif in_story:
            speech_q.put(line)

model = Model("model")
q = queue.Queue()
ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)
active = False
buffer = ""

threading.Thread(target=serial_listener, args=(ser,), daemon=True).start()
threading.Thread(target=speaker_worker, daemon=True).start()
vision_router.preload()  # load YOLO + open camera in the background

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
                if vision_router.handle_vision_command(command, speech_q.put):
                    print(f"Camera question, handled on the Pi: {command}")
                else:
                    print(f"Sending to ESP32: {command}")
                    ser.write((command + "\n").encode())
            elif active:
                print("(listening for command...)")
