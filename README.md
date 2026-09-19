# Offline ESP32-S3 Intent Classifier

This project trains a small, offline command classifier and runs it on an ESP32-S3
with TensorFlow Lite Micro. The model accepts eight integer word-token IDs and
returns thirteen intent scores:

`ARM_UP`, `LEFT_ARM`, `RIGHT_ARM`, `MOVE_FORWARD`, `MOVE_BACKWARD`,
`HEAD_SHAKE`, `HEAD_NOD`, `HEAD_LEFT`, `HEAD_RIGHT`, `TURN_LEFT`, `TURN_RIGHT`,
`DANCE`, `STOP`.

## 1. Train and export on a laptop

Use Python 3.10 or newer in a virtual environment:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install tensorflow numpy
python train_intent_model.py
```

The script creates `generated/intent_model.tflite`, `generated/model_data.h`,
`generated/vocab.h`, and `generated/model_metadata.h`. It writes the model byte
array itself, so `xxd` is optional.
The conversion keeps the input as `int32` token IDs (required by the Embedding
layer) and exports the output as `float32` scores. The script prints the
actual TensorFlow Lite tensor types, dimensions, held-out accuracy, confusion
matrix, and model size after conversion.

Add more representative phrases to `TRAIN_DATASET` and reserve genuinely new
wording for `TEST_DATASET`. The script rejects exact train/test phrase leakage.
Rerun it whenever the command vocabulary changes. Copy the complete `generated`
directory to the Arduino sketch directory; it is intentionally generated rather
than checked in.

The current training run includes noisy transcript variants and produces a model
with `int32[1,8]` input, 13 classes, and `float32[1,13]` output. Common speech
recognition misspellings such as `lef`, `rite`, `bakword`, and `forword` are
normalized before inference. Keep adding field phrases to the held-out tests
before relying on this model for safety-critical motion.

## 2. Build the firmware

Open `esp32_intent_classifier.ino` in Arduino IDE or use it as an ESP-IDF Arduino
component. Install the `TensorFlowLite_ESP32` library and select an ESP32-S3
board with PSRAM enabled. The firmware allocates a 256 KiB tensor arena from
PSRAM and falls back to internal RAM if PSRAM is unavailable.

The default mock servo pins are GPIO 4 (arm), GPIO 5 (head), and GPIO 6 (turn).
Change them for the board wiring. Servo outputs use ESP32 LEDC at 50 Hz. Replace
the calls in `dispatchMotorCommand()` with PCA9685 or motor-driver calls when
the real mechanism is connected.

Send one ASCII command per line at 115200 baud, for example:

```text
Move your arm up
```

The tokenizer lowercases ASCII letters, splits on non-alphanumeric characters,
looks up each word in `vocab.h`, pads to eight tokens, and maps unknown words to
`INTENT_UNK_ID`. The firmware checks the model schema, input type, and tensor
dimensions at boot, then rejects predictions below 0.55 confidence.

## Model contract

| Item | Value |
| --- | --- |
| Input | `int32[1][8]` token IDs |
| Vocabulary | Generated `vocab.h`, ID 0 is PAD, ID 1 is UNK |
| Network | Embedding(10) -> GlobalAveragePooling1D -> Dense(24, ReLU) -> Dense(13, softmax) |
| Output | `float32[1][13]` intent scores |
| Runtime arena | 256 KiB, PSRAM first |

## Laptop prompt runner

Classify one prompt with:

```text
.venv\Scripts\python.exe run_intent_classifier.py --text "Move your arm up"
```

Or start an interactive prompt session:

```text
.venv\Scripts\python.exe run_intent_classifier.py
```

Each result is printed as JSON with the normalized tokens, intent, confidence,
and whether it passes the default `0.55` motor-command threshold. Change the
threshold with `--threshold 0.70`.

## ESP32 prompt runner

Flash `esp32_intent_classifier.ino` with the generated directory beside the
sketch. Open the serial monitor at **115200 baud**, set line ending to **Newline**,
and send a prompt such as:

```text
Turn left
```

The sketch prints the selected intent and confidence, then dispatches the servo
macro only when confidence is at least `0.55`. Serial intake is nonblocking and
bounded to 128 characters, so incomplete input does not pause the main loop.

For a sequence, the ESP32 can plan natural-language text directly. It also
accepts the laptop runner's ESP format:

```text
.venv\Scripts\python.exe run_intent_classifier.py --esp-plan --text "Move your left arm and then shake your head"
```

Copy the resulting line into the serial monitor:

```text
PLAN LEFT_ARM,90,500;WAIT,0,1000;HEAD_SHAKE,90,500
```

Each item is `INTENT,value,duration_ms`; values are clamped to 0-180. `WAIT`
is a planner command, not a learned intent. The ESP32 inserts a default 300 ms
gap between commands, without adding a gap after the final command. Explicit
waits can be written as `wait 2 seconds`, `wait 500 ms`, `wait 2`, `wait 500`,
`wait a little`, or `wait a lot`. Bare values below 20 are seconds; values 20
or greater are milliseconds. Waits are capped at 30 seconds.

Future audio-to-text software can pass its generated transcript into the same
text command path; audio capture and speech recognition are not part of this
firmware.

The laptop planner understands explicit values such as `turn left 45 degrees`,
and presets such as `a little` (30), default (90), and `a lot` (150). These are
currently generic motor parameters; the real motor mapping should be calibrated
per mechanism.

## Behavior tests

Run the broad phrase, sequence, and parameter test corpus with:

```text
.venv\Scripts\python.exe test_intent_model.py
```

The current corpus contains 225 cases: casing, punctuation, polite phrasing,
synonyms, short commands, ambiguous head/body wording, all 169 ordered pairs of
the 13 atomic intents, human wave/raise/lower commands, sequences, modifiers,
and explicit numeric values. The latest run passed 225/225. The test script is a
regression check; add new field phrases there when hardware testing finds a
real-world failure.

Human-command planning includes these expansions:

- `wave your left arm` -> `LEFT_ARM(160)` then `LEFT_ARM(20)`
- `raise your left arm` -> `LEFT_ARM(160)`
- `lower your left arm` -> `LEFT_ARM(20)`

The same arm-direction rules are applied in the laptop planner and ESP32
inferencer, so raising and lowering do not collapse into the same motor value.

## TinyStories starter routing

Story requests are handled separately from motor intents. The trainer exports ten
shared starters to `generated/story_starters.json` and
`generated/story_starters.h`; the existing TinyStories runtime can consume one
starter as its prompt prefix.

On the laptop:

```text
.venv\Scripts\python.exe run_intent_classifier.py --text "tell me a story"
```

Output:

```json
{"type": "STORY_STARTER", "starter": "..."}
```

Use `--story` to force story mode for an arbitrary input. On the ESP32, send a
line such as `tell me a story` at 115200 baud. The Arduino sketch responds with:

```text
STORY_STARTER At sunrise, a small robot found a locked door beneath the old garden.
```

Your existing TinyStories `run.cpp` should listen for the `STORY_STARTER ` prefix,
copy the remainder into its prompt buffer, and begin generation. Story requests
never enter the motor classifier or servo command queue.
