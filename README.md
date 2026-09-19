# Offline ESP32-S3 Intent Classifier

This project trains a small, offline command classifier and runs it on an ESP32-S3
with TensorFlow Lite Micro. The model accepts eight integer word-token IDs and
returns six intent scores:

`0 ARM_UP`, `1 HEAD_SHAKE`, `2 HEAD_NOD`, `3 TURN_LEFT`, `4 TURN_RIGHT`, `5 STOP`.

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
actual TensorFlow Lite tensor types, dimensions, held-out accuracy, confusion
matrix, and model size after conversion.

Add more representative phrases to `TRAIN_DATASET` and reserve genuinely new
wording for `TEST_DATASET`. The script rejects exact train/test phrase leakage.
Rerun it whenever the command vocabulary changes. Copy the complete `generated`
directory to the Arduino sketch directory; it is intentionally generated rather
than checked in.

The current training run uses 69 training phrases, 13 validation phrases, and 24
held-out test phrases. On the development machine it produced a 4,816-byte
TFLite model with 71 vocabulary entries, `int32[1,8]` input, and `int8[1,6]`
output. Held-out accuracy was 95.8%; exact size and accuracy can vary with
TensorFlow versions and training results.

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
| Network | Embedding(8) -> GlobalAveragePooling1D -> Dense(16, ReLU) -> Dense(6, softmax) |
| Output | Six quantized `int8` scores, dequantized using the tensor scale/zero point |
| Runtime arena | 256 KiB, PSRAM first |
