# vision_router.py
# Routes spoken camera questions to the camera and speaks the answer.
#   "what do you see"            -> lists everything in view
#   "do you see a chair"         -> yes / no
#   "how far away is the chair"  -> distance to the nearest one
# handle_vision_command() returns True when it consumed the command (so the caller
# must NOT forward it to the ESP32), False when it is a normal robot command.
#
# Deliberately does not import navigate.py / open any serial port, so it can live
# in the same process as the audio script that owns /dev/ttyUSB0.

import json
import os
import re
import threading

HERE = os.path.dirname(os.path.abspath(__file__))

MIN_CONF = 0.5  # detection confidence needed to mention something

# spoken word -> YOLO class name
ALIASES = {
    "phone": "cell phone", "cellphone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone",
    "table": "dining table", "sofa": "couch", "plant": "potted plant", "houseplant": "potted plant",
    "television": "tv", "monitor": "tv", "screen": "tv",
    "bike": "bicycle", "motorbike": "motorcycle", "plane": "airplane", "ball": "sports ball",
    "glass": "wine glass", "fridge": "refrigerator", "teddy": "teddy bear", "hairdryer": "hair drier",
    "hair dryer": "hair drier", "stop sign": "stop sign", "computer": "laptop",
    "me": "person", "someone": "person", "somebody": "person", "anyone": "person",
    "anybody": "person", "people": "person", "human": "person", "man": "person", "woman": "person",
}

LIST_RE = re.compile(r"\b(what|which)\b.*\b(do you see|can you see|are you looking at|is in front of you|is in view)\b"
                     r"|\b(look around|what('?s| is) around you)\b")
DIST_RE = re.compile(r"\bhow (far|close)\b|\bdistance (to|from)\b")
SEE_RE = re.compile(r"\b(do|can|did|could) you see\b|\bis there (a|an|any|some)\b.*\b(in view|in front of you|around|here)\b")

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

_speak = None
_cam = None
_load_error = None
_lock = threading.Lock()   # one camera query at a time, also guards lazy loading


def _vocabulary():
    with open(os.path.join(HERE, "objectWidths.json"), "r") as f:
        names = list(json.load(f).keys())
    return sorted(names, key=len, reverse=True)  # longest first: "cell phone" before "phone"


VOCAB = _vocabulary()


def find_object_name(text):
    """First object mentioned in the spoken text, as a YOLO class name, else None."""
    low = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    candidates = [(alias, name) for alias, name in ALIASES.items()] + [(n, n) for n in VOCAB]
    candidates.sort(key=lambda c: len(c[0]), reverse=True)
    best = None  # (position, name); earliest mention wins, longest breaks ties
    for spoken, name in candidates:
        m = re.search(r" " + re.escape(spoken) + r"(s|es)? ", low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def classify(text):
    """'list' | 'dist' | 'see' | None"""
    low = text.lower().strip()
    if LIST_RE.search(low):
        return "list"
    if DIST_RE.search(low):
        return "dist"
    if SEE_RE.search(low):
        return "see"
    return None


# ---------- answers ----------

def _plural(name, n):
    if n == 1:
        return name
    return "people" if name == "person" else name + "s"


def _count_phrase(name, n):
    if n == 1:
        article = "an" if name[0] in "aeiou" else "a"
        return f"{article} {name}"
    return f"{NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else n} {_plural(name, n)}"


def _meters(d):
    return f"{d:.1f} meters"


def answer_list(detections):
    if not detections:
        return "I don't see anything."
    counts = {}
    for d in detections:
        counts[d["name"]] = counts.get(d["name"], 0) + 1
    parts = [_count_phrase(name, n) for name, n in counts.items()]
    if len(parts) > 1:
        parts[-1] = "and " + parts[-1]
    return "I see " + (", ".join(parts) if len(parts) > 2 else " ".join(parts)) + "."


def answer_see(detections, name):
    matches = [d for d in detections if d["name"] == name]
    if not matches:
        return f"No, I don't see {_count_phrase(name, 1)}."
    nearest = min(matches, key=lambda d: d["dist"] if d["dist"] is not None else 1e9)
    reply = f"Yes, I see {_count_phrase(name, len(matches))}."
    if nearest["dist"] is not None:
        reply += f" The {'closest is' if len(matches) > 1 else 'distance is'} about {_meters(nearest['dist'])}."
    return reply


def answer_dist(detections, name):
    matches = [d for d in detections if d["name"] == name and d["dist"] is not None]
    if not matches:
        return f"I don't see {_count_phrase(name, 1)}."
    nearest = min(matches, key=lambda d: d["dist"])
    return f"The {name} is about {_meters(nearest['dist'])} away."


# ---------- camera plumbing ----------

def _ensure_camera():
    """Lazily import YOLO + open the camera. Returns True when ready."""
    global _cam, _load_error
    if _cam is not None:
        return True
    try:
        import ClickClass as CC  # heavy: loads YOLO / mediapipe
        _cam = CC.Click()
        _load_error = None
        return True
    except Exception as e:
        _load_error = e
        print("VISION: camera init failed:", e)
        return False


def preload():
    """Warm up YOLO + camera in the background so the first question is fast."""
    def run():
        with _lock:
            _ensure_camera()
    threading.Thread(target=run, daemon=True).start()


def _run_query(kind, name):
    with _lock:
        if not _ensure_camera():
            _speak("My camera isn't working right now.")
            return
        try:
            detections = _cam.detectAll(MIN_CONF)
            import ClickClass as CC
            if name is not None and name not in CC.model.names.values():
                _speak(f"I don't know how to recognize a {name}.")
                return
        except Exception as e:
            print("VISION: query failed:", e)
            _speak("Something went wrong with my eyes.")
            return
    if detections is None:
        _speak("I couldn't get an image from my camera.")
        return
    print("VISION:", [(d["name"], round(d["conf"], 2), None if d["dist"] is None else round(d["dist"], 2))
                       for d in detections])
    if kind == "list":
        _speak(answer_list(detections))
    elif kind == "see":
        _speak(answer_see(detections, name))
    else:
        _speak(answer_dist(detections, name))


def handle_vision_command(text, speak):
    """
    text: the transcribed command. speak: callable(str) that says something out loud.
    Returns True if this was a camera question (handled here, on a worker thread).
    """
    global _speak
    kind = classify(text)
    if kind is None:
        return False
    _speak = speak
    name = None
    if kind in ("see", "dist"):
        name = find_object_name(text)
        if name is None:
            speak("I didn't catch which object you meant.")
            return True
    threading.Thread(target=_run_query, args=(kind, name), daemon=True).start()
    return True
