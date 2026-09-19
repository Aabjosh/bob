r"""Run the exported text-to-motor-intent model on a laptop.

Examples:
  .venv\Scripts\python.exe run_intent_classifier.py
  .venv\Scripts\python.exe run_intent_classifier.py --text "Turn left"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

INTENTS = (
    "ARM_UP",
    "LEFT_ARM",
    "RIGHT_ARM",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "HEAD_SHAKE",
    "HEAD_NOD",
    "HEAD_LEFT",
    "HEAD_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "DANCE",
    "STOP",
)
WORD_PATTERN = re.compile(r"[a-z0-9]+")
VOCAB_ENTRY_PATTERN = re.compile(r'^\s*"([^"\\]*)",\s*$')
NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?\b")
WAIT_PATTERN = re.compile(
    r"^(?:please\s+)?(?:wait|pause)(?:\s+(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>milliseconds?|ms|seconds?|secs?|s)?|\s+(?P<preset>a little|a lot))?\s*$",
    re.IGNORECASE,
)
SEQUENCE_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|\band then\b|\band\b|\bthen\b)\s*|"
    r"\s+(?=(?:turn|go|move|walk|drive|rotate|look|tilt|shake|nod|dance|stop|wait|pause)\b)",
    re.IGNORECASE,
)
STORY_REQUEST_PATTERN = re.compile(
    r"\b(?:tell|read|make|write|start|give)\b.*\b(?:story|tale)\b|\b(?:story|tale)\b.*\b(?:please|now)\b",
    re.IGNORECASE,
)
DEFAULT_WAIT_MS = 300
MAX_WAIT_MS = 30_000


@dataclass
class MotorCommand:
    intent: str
    value: float = 90.0
    duration_ms: int = 500
    confidence: float = 0.0
    source: str = ""


def parse_wait_duration(text: str) -> int | None:
    match = WAIT_PATTERN.match(text.strip())
    if not match:
        return None
    preset = (match.group("preset") or "").strip().lower()
    if preset == "a little":
        return 300
    if preset == "a lot":
        return 2_000
    amount_text = match.group("amount")
    if amount_text is None:
        return DEFAULT_WAIT_MS
    amount = float(amount_text)
    unit = (match.group("unit") or "").lower()
    if unit.startswith("s") or (not unit and amount < 20):
        amount *= 1_000
    duration_ms = int(round(amount))
    return min(max(duration_ms, 0), MAX_WAIT_MS)


def normalize_transcript(text: str) -> str:
    replacements = {
        "lef": "left",
        "rite": "right",
        "bakward": "backward",
        "bakword": "backward",
        "forword": "forward",
        "pleese": "please",
    }
    words = text.lower().split()
    return " ".join(replacements.get(word.strip(".,!?;:"), word.strip(".,!?;:")) for word in words)


def command_for_segment(classifier: "IntentClassifier", segment: str) -> list[MotorCommand]:
    lowered = segment.lower()
    wait_duration = parse_wait_duration(segment)
    if wait_duration is not None:
        return [MotorCommand("WAIT", 0.0, wait_duration, 1.0, segment)]
    arm_intent = "LEFT_ARM" if "left" in lowered else "RIGHT_ARM" if "right" in lowered else None
    if "wave" in lowered and arm_intent is not None:
        return [
            MotorCommand(arm_intent, 160.0, 350, 1.0, segment),
            MotorCommand(arm_intent, 20.0, 350, 1.0, segment),
        ]

    intent, confidence, _ = classifier.classify(segment)
    value, duration_ms = parse_parameters(segment)
    if arm_intent is not None and intent in ("LEFT_ARM", "RIGHT_ARM"):
        if any(word in lowered for word in ("raise", "lift", "up", "higher")):
            value = 160.0
        elif any(word in lowered for word in ("lower", "drop", "down")):
            value = 20.0
    return [MotorCommand(intent, value, duration_ms, confidence, segment)]


def tokenize(text: str) -> list[str]:
    """Match the tokenizer used by train_intent_model.py and the firmware."""
    return WORD_PATTERN.findall(text.lower())


def load_vocabulary(path: Path) -> dict[str, int]:
    words: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        match = VOCAB_ENTRY_PATTERN.match(line)
        if match:
            words.append(match.group(1))
    if len(words) < 2 or words[0] != "<PAD>" or words[1] != "<UNK>":
        raise ValueError(f"Invalid generated vocabulary header: {path}")
    return {word: index for index, word in enumerate(words)}


def is_story_request(text: str) -> bool:
    lowered = text.lower().strip()
    return bool(STORY_REQUEST_PATTERN.search(lowered) or lowered in {"story", "a story", "tell me a story"})


def load_story_starters(path: Path) -> list[str]:
    starters = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(starters, list) or not starters or not all(isinstance(item, str) for item in starters):
        raise ValueError(f"Invalid story starters file: {path}")
    return starters


def choose_story_starter(path: Path) -> str:
    return random.SystemRandom().choice(load_story_starters(path))


def rule_based_intent(text: str) -> str | None:
    """Resolve unambiguous safety-critical phrases before model inference."""
    lowered = text.lower()
    if "arm" in lowered and "left" in lowered:
        return "LEFT_ARM"
    if "arm" in lowered and "right" in lowered:
        return "RIGHT_ARM"
    has_head = "head" in lowered
    if has_head and ("shake" in lowered or ("left" in lowered and "right" in lowered)):
        return "HEAD_SHAKE"
    if has_head and ("nod" in lowered or "bob" in lowered or "yes" in lowered):
        return "HEAD_NOD"
    if has_head and "left" in lowered:
        return "HEAD_LEFT"
    if has_head and "right" in lowered:
        return "HEAD_RIGHT"
    if any(word in lowered for word in ("stop", "halt", "freeze", "hold position")):
        return "STOP"
    return None


class IntentClassifier:
    def __init__(self, model_path: Path, vocab_path: Path) -> None:
        self.vocabulary = load_vocabulary(vocab_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        input_shape = tuple(int(value) for value in self.input_details["shape"])
        self.max_tokens = input_shape[1]
        output_shape = tuple(int(value) for value in self.output_details["shape"])
        if len(input_shape) != 2 or input_shape[0] != 1 or self.input_details["dtype"] != np.int32:
            raise ValueError(f"Unexpected model input contract: {input_shape}, {self.input_details['dtype']}")
        if output_shape != (1, len(INTENTS)):
            raise ValueError(f"Unexpected model output contract: {output_shape}")

    def classify(self, text: str) -> tuple[str, float, list[str]]:
        normalized = normalize_transcript(text)
        words = tokenize(normalized)
        rule_intent = rule_based_intent(normalized)
        if rule_intent is not None:
            return rule_intent, 1.0, words
        token_ids = [self.vocabulary.get(word, 1) for word in words[:self.max_tokens]]
        token_ids.extend([0] * (self.max_tokens - len(token_ids)))
        input_values = np.asarray([token_ids], dtype=np.int32)
        self.interpreter.set_tensor(self.input_details["index"], input_values)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details["index"])[0]
        scale, zero_point = self.output_details["quantization"]
        if self.output_details["dtype"] == np.int8 and scale:
            probabilities = (output.astype(np.float32) - zero_point) * scale
        elif self.output_details["dtype"] == np.uint8 and scale:
            probabilities = (output.astype(np.float32) - zero_point) * scale
        else:
            probabilities = output.astype(np.float32)
        intent_id = int(np.argmax(probabilities))
        return INTENTS[intent_id], float(probabilities[intent_id]), words

    def plan(self, text: str, threshold: float = 0.55, max_commands: int = 8) -> list[MotorCommand]:
        """Classify a bounded sequence and extract safe numeric/default values."""
        commands: list[MotorCommand] = []
        for segment in SEQUENCE_SPLIT_PATTERN.split(text):
            segment = segment.strip()
            if not segment or len(commands) >= max_commands:
                continue
            for command in command_for_segment(self, segment):
                if len(commands) >= max_commands:
                    break
                commands.append(command)
                if command.intent == "STOP":
                    return [item for item in commands if item.confidence >= threshold]
        return [command for command in commands if command.confidence >= threshold]


def parse_parameters(text: str) -> tuple[float, int]:
    explicit_value = NUMBER_PATTERN.search(text)
    if explicit_value:
        value = float(explicit_value.group(1))
        return min(value, 180.0), 500
    lowered = text.lower()
    if "little" in lowered or "slightly" in lowered:
        return 30.0, 300
    if "lot" in lowered or "far" in lowered or "large" in lowered:
        return 150.0, 800
    return 90.0, 500


def print_result(classifier: IntentClassifier, text: str, threshold: float) -> None:
    intent, confidence, words = classifier.classify(text)
    accepted = confidence >= threshold
    result = {
        "text": text,
        "tokens": words[:8],
        "intent": intent,
        "confidence": round(confidence, 4),
        "accepted": accepted,
    }
    print(json.dumps(result))


def print_plan(classifier: IntentClassifier, text: str, threshold: float) -> None:
    commands = classifier.plan(text, threshold=threshold)
    print(json.dumps([asdict(command) for command in commands]))


def print_esp_plan(classifier: IntentClassifier, text: str, threshold: float) -> None:
    commands = classifier.plan(text, threshold=threshold)
    wire_commands = ";".join(f"{command.intent},{round(command.value):d},{command.duration_ms}" for command in commands)
    print(f"PLAN {wire_commands}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify text into an ESP32 motor intent.")
    parser.add_argument("--text", help="Classify one prompt and exit.")
    parser.add_argument("--plan", action="store_true", help="Interpret the text as a sequence of motor commands.")
    parser.add_argument("--esp-plan", action="store_true", help="Print a PLAN line ready for the ESP32 serial monitor.")
    parser.add_argument("--story", action="store_true", help="Treat the input as a story request and print a random starter.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum confidence to accept a motor command.")
    parser.add_argument("--model", type=Path, default=Path("generated/intent_model.tflite"))
    parser.add_argument("--vocab", type=Path, default=Path("generated/vocab.h"))
    parser.add_argument("--story-starters", type=Path, default=Path("generated/story_starters.json"))
    args = parser.parse_args()

    classifier = IntentClassifier(args.model, args.vocab)
    if args.text is not None:
        if args.story or is_story_request(args.text):
            print(json.dumps({"type": "STORY_STARTER", "starter": choose_story_starter(args.story_starters)}))
        elif args.esp_plan:
            print_esp_plan(classifier, args.text, args.threshold)
        elif args.plan:
            print_plan(classifier, args.text, args.threshold)
        else:
            print_result(classifier, args.text, args.threshold)
        return

    print("Intent classifier ready. Enter a command, or press Ctrl+C to exit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            return
        if text:
            if args.story or is_story_request(text):
                print(json.dumps({"type": "STORY_STARTER", "starter": choose_story_starter(args.story_starters)}))
            elif args.esp_plan:
                print_esp_plan(classifier, text, args.threshold)
            elif args.plan:
                print_plan(classifier, text, args.threshold)
            else:
                print_result(classifier, text, args.threshold)


if __name__ == "__main__":
    main()
