"""Behavior tests for the exported intent model and command planner."""

from pathlib import Path

from run_intent_classifier import (
    IntentClassifier,
    choose_story_starter,
    is_story_request,
    load_story_starters,
    parse_wait_duration,
)

MODEL = Path(__file__).parent / "generated" / "intent_model.tflite"
VOCAB = Path(__file__).parent / "generated" / "vocab.h"
STORY_STARTERS = Path(__file__).parent / "generated" / "story_starters.json"

# Deliberately varied: casing, punctuation, polite wording, synonyms, and short forms.
SINGLE_CASES = [
    ("ARM_UP", "RAISE the arm!"),
    ("ARM_UP", "Could you move my arm higher, please?"),
    ("ARM_UP", "lift hand up"),
    ("ARM_UP", "push arm up"),
    ("LEFT_ARM", "Please move the arm on the left."),
    ("LEFT_ARM", "wave with my left hand"),
    ("LEFT_ARM", "LEFT ARM ACTION"),
    ("LEFT_ARM", "activate left arm"),
    ("RIGHT_ARM", "Please move the arm on the right."),
    ("RIGHT_ARM", "wave with my right hand"),
    ("RIGHT_ARM", "RIGHT ARM ACTION!"),
    ("RIGHT_ARM", "engage right arm"),
    ("MOVE_FORWARD", "GO FORWARD!"),
    ("MOVE_FORWARD", "Please travel straight ahead."),
    ("MOVE_FORWARD", "move ahead"),
    ("MOVE_FORWARD", "march forward"),
    ("MOVE_BACKWARD", "GO BACKWARD!"),
    ("MOVE_BACKWARD", "Please reverse direction."),
    ("MOVE_BACKWARD", "back up"),
    ("MOVE_BACKWARD", "retreat backward"),
    ("HEAD_SHAKE", "Shake your head, please."),
    ("HEAD_SHAKE", "make a no motion"),
    ("HEAD_SHAKE", "head left and right"),
    ("HEAD_SHAKE", "waggle your head"),
    ("HEAD_NOD", "NOD your head!"),
    ("HEAD_NOD", "make a yes motion"),
    ("HEAD_NOD", "bob your head"),
    ("HEAD_NOD", "agree with your head"),
    ("HEAD_LEFT", "Turn your head toward the left."),
    ("HEAD_LEFT", "look left with your head"),
    ("HEAD_LEFT", "tilt head left"),
    ("HEAD_LEFT", "aim head left"),
    ("HEAD_RIGHT", "Turn your head toward the right."),
    ("HEAD_RIGHT", "look right with your head"),
    ("HEAD_RIGHT", "tilt head right"),
    ("HEAD_RIGHT", "face right"),
    ("TURN_LEFT", "TURN LEFT!"),
    ("TURN_LEFT", "Please rotate the body left."),
    ("TURN_LEFT", "make a left turn"),
    ("TURN_LEFT", "spin left"),
    ("TURN_RIGHT", "TURN RIGHT!"),
    ("TURN_RIGHT", "Please rotate the body right."),
    ("TURN_RIGHT", "make a right turn"),
    ("TURN_RIGHT", "veer right"),
    ("DANCE", "DANCE!"),
    ("DANCE", "Please perform a dance routine."),
    ("DANCE", "start dancing now"),
    ("DANCE", "boogie"),
    ("STOP", "STOP!!!"),
    ("STOP", "Please hold position."),
    ("STOP", "freeze immediately"),
    ("STOP", "stand still"),
]

SEQUENCE_CASES = [
    (["TURN_LEFT", "MOVE_FORWARD"], "turn left and go forward"),
    (["LEFT_ARM", "HEAD_SHAKE"], "move my left arm, then shake my head"),
    (["RIGHT_ARM", "HEAD_NOD", "STOP"], "right arm and nod, then stop"),
    (["MOVE_BACKWARD", "TURN_RIGHT"], "go back; rotate right"),
    (["HEAD_LEFT", "HEAD_RIGHT", "DANCE"], "look left then look right and dance"),
    (["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"], "turn left turn right go forward"),
    (["TURN_LEFT", "WAIT", "TURN_RIGHT"], "turn left, wait 2 seconds, turn right"),
]

NOISY_CASES = [
    ("LEFT_ARM", "uh move the lef arm please"),
    ("RIGHT_ARM", "move the rite arm now"),
    ("MOVE_FORWARD", "go forword please"),
    ("MOVE_BACKWARD", "uh go bakword"),
    ("TURN_LEFT", "turn lef now"),
    ("TURN_RIGHT", "please turn rite"),
    ("STOP", "pleese stop now"),
]

WAIT_CASES = [
    (300, "wait"),
    (2000, "wait 2"),
    (500, "wait 500"),
    (2000, "wait 2 seconds"),
    (500, "pause 500 ms"),
    (300, "wait a little"),
    (2000, "pause a lot"),
]

# Canonical natural phrases used to exercise every ordered pair. STOP is a
# safety barrier: when it is first, later commands must not be executed.
PAIR_PHRASES = {
    "ARM_UP": "raise your arm",
    "LEFT_ARM": "move your left arm",
    "RIGHT_ARM": "move your right arm",
    "MOVE_FORWARD": "go forward",
    "MOVE_BACKWARD": "go backward",
    "HEAD_SHAKE": "shake your head",
    "HEAD_NOD": "nod your head",
    "HEAD_LEFT": "look left with your head",
    "HEAD_RIGHT": "look right with your head",
    "TURN_LEFT": "turn left",
    "TURN_RIGHT": "turn right",
    "DANCE": "dance",
    "STOP": "stop",
}

PAIR_CASES = []
for first_intent, first_phrase in PAIR_PHRASES.items():
    for second_intent, second_phrase in PAIR_PHRASES.items():
        expected_pair = [first_intent] if first_intent == "STOP" else [first_intent, second_intent]
        if first_intent != "STOP" and second_intent == "STOP":
            expected_pair = [first_intent, "STOP"]
        PAIR_CASES.append((expected_pair, f"{first_phrase} and then {second_phrase}"))

HUMAN_CASES = [
    (["LEFT_ARM", "LEFT_ARM"], "Wave your left arm"),
    (["RIGHT_ARM", "RIGHT_ARM"], "could you wave the right arm?"),
    (["LEFT_ARM"], "raise your left arm"),
    (["LEFT_ARM"], "lower your left arm"),
    (["RIGHT_ARM"], "lift the right arm up"),
    (["RIGHT_ARM"], "drop the right arm down"),
    (["LEFT_ARM", "LEFT_ARM", "MOVE_FORWARD", "HEAD_SHAKE"], "wave left arm, go forward, then shake head"),
]

STORY_CASES = [
    "tell me a story",
    "Tell me a tale, please",
    "make up a story",
    "write a short tale",
    "start a story now",
]

PARAMETER_CASES = [
    (30.0, 300, "turn left a little"),
    (90.0, 500, "turn right"),
    (150.0, 800, "move forward a lot"),
    (45.0, 500, "turn left 45 degrees"),
    (180.0, 500, "turn right 360 degrees"),
]


def main() -> int:
    classifier = IntentClassifier(MODEL, VOCAB)
    failures = 0
    starters = load_story_starters(STORY_STARTERS)
    print(f"\nStory routing cases: {len(STORY_CASES)} ({len(starters)} starters)")
    for text in STORY_CASES:
        passed = is_story_request(text) and choose_story_starter(STORY_STARTERS) in starters
        status = "PASS" if passed else "FAIL"
        print(f"{status} story_request={is_story_request(text)} text={text!r}")
        failures += not passed
    print(f"Single-command cases: {len(SINGLE_CASES)}")
    for expected, text in SINGLE_CASES:
        actual, confidence, _ = classifier.classify(text)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} expected={expected:<14} actual={actual:<14} confidence={confidence:.3f} text={text!r}")
        failures += actual != expected

    print(f"\nSequence cases: {len(SEQUENCE_CASES)}")
    for expected, text in SEQUENCE_CASES:
        actual = [command.intent for command in classifier.plan(text)]
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} expected={expected} actual={actual} text={text!r}")
        failures += actual != expected

    print(f"\nNoisy transcript cases: {len(NOISY_CASES)}")
    for expected, text in NOISY_CASES:
        actual, confidence, _ = classifier.classify(text)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} expected={expected:<14} actual={actual:<14} confidence={confidence:.3f} text={text!r}")
        failures += actual != expected

    print(f"\nWait cases: {len(WAIT_CASES)}")
    for expected, text in WAIT_CASES:
        actual = parse_wait_duration(text)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} expected={expected} actual={actual} text={text!r}")
        failures += actual != expected

    print(f"\nOrdered pair cases: {len(PAIR_CASES)}")
    for expected, text in PAIR_CASES:
        actual = [command.intent for command in classifier.plan(text)]
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            print(f"{status} expected={expected} actual={actual} text={text!r}")
        failures += actual != expected

    print(f"\nHuman command cases: {len(HUMAN_CASES)}")
    for expected, text in HUMAN_CASES:
        commands = classifier.plan(text)
        actual = [command.intent for command in commands]
        status = "PASS" if actual == expected else "FAIL"
        values = [command.value for command in commands]
        print(f"{status} expected={expected} actual={actual} values={values} text={text!r}")
        if text.lower().startswith("raise your left"):
            failures += not (values == [160.0])
        elif text.lower().startswith("lower your left"):
            failures += not (values == [20.0])
        failures += actual != expected

    print(f"\nParameter cases: {len(PARAMETER_CASES)}")
    for expected_value, expected_duration, text in PARAMETER_CASES:
        commands = classifier.plan(text)
        actual = commands[0] if commands else None
        passed = (
            actual is not None
            and actual.value == expected_value
            and actual.duration_ms == expected_duration
        )
        status = "PASS" if passed else "FAIL"
        actual_values = None if actual is None else (actual.value, actual.duration_ms, actual.intent)
        print(f"{status} expected={(expected_value, expected_duration)} actual={actual_values} text={text!r}")
        failures += not passed

    total = (len(STORY_CASES) + len(SINGLE_CASES) + len(SEQUENCE_CASES) + len(PAIR_CASES) +
             len(NOISY_CASES) + len(WAIT_CASES) + len(HUMAN_CASES) + len(PARAMETER_CASES))
    print(f"\nResult: {total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
