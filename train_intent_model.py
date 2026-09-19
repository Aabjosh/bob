"""Train and export a tiny offline command-intent classifier.

Outputs:
  generated/vocab.h
  generated/model_data.h
  generated/intent_model.tflite

Install: python -m pip install tensorflow
Run:     python train_intent_model.py
"""

from pathlib import Path
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 7
MAX_TOKENS = 8
EMBEDDING_DIM = 8
HIDDEN_UNITS = 16
EPOCHS = 120
OUTPUT_DIR = Path(__file__).parent / "generated"

INTENTS = [
    "ARM_UP",
    "HEAD_SHAKE",
    "HEAD_NOD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
]

# These phrases intentionally include synonyms, polite prefixes, and different
# word orders. Keep the held-out phrases below separate from this set.
TRAIN_DATASET = {
    "ARM_UP": [
        "move your arm up", "raise arm", "lift the arm", "arm up",
        "please raise your arm", "move arm upward", "lift your hand",
        "bring your arm higher", "raise your hand up", "put arm up",
        "can you lift your arm", "move the arm upwards", "arm raise",
        "please move arm up", "up with your arm", "elevate your arm",
    ],
    "HEAD_SHAKE": [
        "shake your head", "shake head", "move head side to side",
        "say no", "head left right", "turn head back and forth",
        "move your head left and right", "shake from side to side",
        "please shake your head", "make a no motion", "head side to side",
        "look left then right with your head", "shake head left right",
    ],
    "HEAD_NOD": [
        "nod your head", "nod head", "move head up and down",
        "say yes", "head nod", "bob your head", "move head vertically",
        "please nod your head", "make a yes motion", "head up and down",
        "nod your head twice", "bob head up down", "nod forward and back",
    ],
    "TURN_LEFT": [
        "turn left", "rotate left", "go to the left", "steer left",
        "move left", "turn your body left", "please turn to the left",
        "take a left", "head toward the left", "left turn now",
        "rotate your body left", "go left please", "make a left turn",
    ],
    "TURN_RIGHT": [
        "turn right", "rotate right", "go to the right", "steer right",
        "move right", "turn your body right", "please turn to the right",
        "take a right", "head toward the right", "right turn now",
        "rotate your body right", "go right please", "make a right turn",
    ],
    "STOP": [
        "stop", "halt", "freeze", "do not move", "all stop",
        "stop moving", "emergency stop", "please stop", "stop right now",
        "cease movement", "hold position", "do not move now",
        "bring everything to a stop", "stop immediately",
    ],
}

# These are not used for fitting. They test paraphrases and short commands that
# do not occur verbatim in TRAIN_DATASET.
TEST_DATASET = {
    "ARM_UP": ["lift your arm higher", "raise the hand", "arm move upward", "move arm high"],
    "HEAD_SHAKE": ["shake from left to right", "move head left right", "move head back and forth", "head shake please"],
    "HEAD_NOD": ["nod head up and down", "move the head up down", "head bob twice", "nod forward"],
    "TURN_LEFT": ["make the body turn left", "move your body left", "leftward move", "steer to the left"],
    "TURN_RIGHT": ["make the body turn right", "move your body right", "rightward move", "steer to the right"],
    "STOP": ["stop moving immediately", "freeze and hold", "stop all movement", "freeze in place"],
}


def tokenize(text: str) -> list[str]:
    """Match the firmware tokenizer: lowercase ASCII words and digits."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary() -> dict[str, int]:
    all_phrases = list(TRAIN_DATASET.values()) + list(TEST_DATASET.values())
    words = sorted({word for phrases in all_phrases for text in phrases for word in tokenize(text)})
    return {"<PAD>": 0, "<UNK>": 1, **{word: index + 2 for index, word in enumerate(words)}}


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    ids = [vocabulary.get(word, vocabulary["<UNK>"]) for word in tokenize(text)]
    return (ids + [vocabulary["<PAD>"]] * MAX_TOKENS)[:MAX_TOKENS]


def write_vocab_header(vocabulary: dict[str, int]) -> None:
    words = list(vocabulary.keys())
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define INTENT_VOCAB_SIZE {len(words)}",
        f"#define INTENT_MAX_TOKENS {MAX_TOKENS}",
        "#define INTENT_PAD_ID 0",
        "#define INTENT_UNK_ID 1",
        "",
        "static const char *const kIntentVocabulary[INTENT_VOCAB_SIZE] = {",
    ]
    lines.extend(f'    "{word}",' for word in words)
    lines.extend(["};", ""])
    (OUTPUT_DIR / "vocab.h").write_text("\n".join(lines), encoding="ascii")


def write_model_header(model_path: Path) -> None:
    data = model_path.read_bytes()
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"static const unsigned int g_intent_model_len = {len(data)};",
        "static const unsigned char g_intent_model[] = {",
    ]
    for offset in range(0, len(data), 12):
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in data[offset : offset + 12]) + ",")
    lines.extend(["};", ""])
    (OUTPUT_DIR / "model_data.h").write_text("\n".join(lines), encoding="ascii")


def write_metadata_header(vocabulary: dict[str, int], model_path: Path) -> None:
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define INTENT_METADATA_CLASS_COUNT {len(INTENTS)}",
        f"#define INTENT_METADATA_VOCAB_SIZE {len(vocabulary)}",
        f"#define INTENT_METADATA_MAX_TOKENS {MAX_TOKENS}",
        f"#define INTENT_METADATA_MODEL_BYTES {model_path.stat().st_size}",
        "#define INTENT_METADATA_INPUT_TYPE_INT32 1",
        "#define INTENT_METADATA_OUTPUT_TYPE_INT8 1",
        "",
        "static const char *const kIntentClassNames[INTENT_METADATA_CLASS_COUNT] = {",
    ]
    lines.extend(f'    "{intent}",' for intent in INTENTS)
    lines.extend(["};", ""])
    (OUTPUT_DIR / "model_metadata.h").write_text("\n".join(lines), encoding="ascii")


def make_examples(dataset: dict[str, list[str]], vocabulary: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    texts = [text for intent in INTENTS for text in dataset[intent]]
    labels = [intent_id for intent_id, intent in enumerate(INTENTS) for _ in dataset[intent]]
    return (
        np.asarray([encode(text, vocabulary) for text in texts], dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
    )


def assert_no_split_leakage() -> None:
    train_phrases = {tuple(tokenize(text)) for phrases in TRAIN_DATASET.values() for text in phrases}
    test_phrases = {tuple(tokenize(text)) for phrases in TEST_DATASET.values() for text in phrases}
    overlap = train_phrases.intersection(test_phrases)
    if overlap:
        raise ValueError(f"Train/test phrase leakage detected: {sorted(overlap)}")


def report_evaluation(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=len(INTENTS)).numpy()
    accuracy = float(np.mean(predictions == y_test))
    print(f"Held-out accuracy: {accuracy:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion)
    for intent_id, intent in enumerate(INTENTS):
        true_positive = confusion[intent_id, intent_id]
        actual = confusion[intent_id].sum()
        predicted = confusion[:, intent_id].sum()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        print(f"  {intent}: precision={precision:.3f} recall={recall:.3f}")


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    assert_no_split_leakage()

    vocabulary = build_vocabulary()
    write_vocab_header(vocabulary)

    x, y = make_examples(TRAIN_DATASET, vocabulary)
    x_test, y_test = make_examples(TEST_DATASET, vocabulary)
    order = np.random.permutation(len(x))
    x, y = x[order], y[order]
    validation_size = max(len(INTENTS), len(x) // 6)
    x_train, y_train = x[:-validation_size], y[:-validation_size]
    x_validation, y_validation = x[-validation_size:], y[-validation_size:]

    model = keras.Sequential([
        keras.layers.Input(shape=(MAX_TOKENS,), dtype="int32"),
        keras.layers.Embedding(len(vocabulary), EMBEDDING_DIM, mask_zero=False),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
        keras.layers.Dense(len(INTENTS), activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=8,
        verbose=0,
        validation_data=(x_validation, y_validation),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
        shuffle=True,
    )
    report_evaluation(model, x_test, y_test)

    def representative_dataset():
        for row in x_train:
            yield [row[np.newaxis, :].astype(np.int32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Embedding indices remain int32 automatically; TensorFlow 2.21 rejects
    # explicitly assigning int32 to inference_input_type.
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    model_path = OUTPUT_DIR / "intent_model.tflite"
    model_path.write_bytes(tflite_model)
    write_model_header(model_path)
    write_metadata_header(vocabulary, model_path)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    print(f"Training examples: {len(x_train)}; validation: {len(x_validation)}; test: {len(x_test)}")
    print(f"Vocabulary: {len(vocabulary)} words; classes: {len(INTENTS)}")
    print(f"Input:  shape={input_info['shape'].tolist()} type={input_info['dtype']}")
    print(f"Output: shape={output_info['shape'].tolist()} type={output_info['dtype']}")
    print(f"Model bytes: {len(tflite_model)}")
    print(f"Wrote:  {model_path}, {OUTPUT_DIR / 'model_data.h'}, {OUTPUT_DIR / 'vocab.h'}, {OUTPUT_DIR / 'model_metadata.h'}")


if __name__ == "__main__":
    main()
