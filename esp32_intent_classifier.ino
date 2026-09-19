#include <Arduino.h>
#include <cctype>
#include <cstring>
#include <TensorFlowLite_ESP32.h>
#include "generated/model_data.h"
#include "generated/vocab.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"

namespace {
constexpr int kIntentCount = 6;
constexpr int kServoFrequency = 50;
constexpr int kServoResolution = 16;
constexpr int kArmServoChannel = 0;
constexpr int kHeadServoChannel = 1;
constexpr int kTurnServoChannel = 2;
constexpr size_t kTensorArenaSize = 256 * 1024;

uint8_t *tensor_arena = nullptr;
tflite::ErrorReporter *error_reporter = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

const char *const kIntentNames[kIntentCount] = {
    "ARM_UP", "HEAD_SHAKE", "HEAD_NOD", "TURN_LEFT", "TURN_RIGHT", "STOP",
};

uint8_t *allocateTensorArena() {
  uint8_t *arena = static_cast<uint8_t *>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (arena == nullptr) {
    arena = static_cast<uint8_t *>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT));
  }
  return arena;
}

int lookupToken(const char *word) {
  for (int id = 0; id < INTENT_VOCAB_SIZE; ++id) {
    if (strcmp(word, kIntentVocabulary[id]) == 0) {
      return id;
    }
  }
  return INTENT_UNK_ID;
}

void tokenizeToInput(const String &raw_text, TfLiteTensor *input) {
  for (int index = 0; index < INTENT_MAX_TOKENS; ++index) {
    input->data.i32[index] = INTENT_PAD_ID;
  }

  String word;
  int token_index = 0;
  for (size_t index = 0; index <= raw_text.length() && token_index < INTENT_MAX_TOKENS; ++index) {
    const char character = index < raw_text.length() ? raw_text[index] : ' ';
    const bool is_word_character =
        (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9');
    if (is_word_character) {
      word += static_cast<char>(tolower(static_cast<unsigned char>(character)));
    } else if (word.length() > 0) {
      char word_buffer[32];
      word.toCharArray(word_buffer, sizeof(word_buffer));
      input->data.i32[token_index++] = lookupToken(word_buffer);
      word = "";
    }
  }
}

float outputProbability(int index) {
  if (output_tensor->type == kTfLiteInt8) {
    return (static_cast<float>(output_tensor->data.int8[index]) - output_tensor->params.zero_point) *
           output_tensor->params.scale;
  }
  if (output_tensor->type == kTfLiteUInt8) {
    return (static_cast<float>(output_tensor->data.uint8[index]) - output_tensor->params.zero_point) *
           output_tensor->params.scale;
  }
  return output_tensor->data.f[index];
}

int classify(const String &raw_text, float *confidence) {
  tokenizeToInput(raw_text, input_tensor);
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    *confidence = 0.0f;
    return -1;
  }

  int best_intent = 0;
  float best_score = outputProbability(0);
  for (int intent = 1; intent < kIntentCount; ++intent) {
    const float score = outputProbability(intent);
    if (score > best_score) {
      best_score = score;
      best_intent = intent;
    }
  }
  *confidence = best_score;
  return best_intent;
}

void writeServoAngle(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  // 500-2500 us pulse range at 50 Hz, suitable for common hobby servos.
  const uint32_t duty = static_cast<uint32_t>(
      (500.0f + (2000.0f * angle / 180.0f)) * 65535.0f / 20000.0f);
  ledcWrite(channel, duty);
}

void dispatchMotorCommand(int intent_id) {
  switch (intent_id) {
    case 0:  // ARM_UP
      writeServoAngle(kArmServoChannel, 160);
      Serial.println("ARM_UP");
      break;
    case 1:  // HEAD_SHAKE
      writeServoAngle(kHeadServoChannel, 60);
      delay(180);
      writeServoAngle(kHeadServoChannel, 120);
      Serial.println("HEAD_SHAKE");
      break;
    case 2:  // HEAD_NOD
      writeServoAngle(kHeadServoChannel, 70);
      delay(180);
      writeServoAngle(kHeadServoChannel, 110);
      Serial.println("HEAD_NOD");
      break;
    case 3:  // TURN_LEFT
      writeServoAngle(kTurnServoChannel, 45);
      Serial.println("TURN_LEFT");
      break;
    case 4:  // TURN_RIGHT
      writeServoAngle(kTurnServoChannel, 135);
      Serial.println("TURN_RIGHT");
      break;
    case 5:  // STOP
      writeServoAngle(kArmServoChannel, 90);
      writeServoAngle(kHeadServoChannel, 90);
      writeServoAngle(kTurnServoChannel, 90);
      Serial.println("STOP");
      break;
    default:
      Serial.println("Unknown intent");
      break;
  }
}

bool initializeClassifier() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  const tflite::Model *model = tflite::GetModel(g_intent_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  tensor_arena = allocateTensorArena();
  if (tensor_arena == nullptr) {
    Serial.println("Tensor arena allocation failed");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    return false;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  if (input_tensor->type != kTfLiteInt32 || input_tensor->dims->size != 2 ||
      input_tensor->dims->data[0] != 1 || input_tensor->dims->data[1] != INTENT_MAX_TOKENS ||
      output_tensor->dims->size != 2 || output_tensor->dims->data[0] != 1 ||
      output_tensor->dims->data[1] != kIntentCount) {
    Serial.println("Unexpected model tensor contract");
    return false;
  }
  return true;
}
}  // namespace

void setup() {
  Serial.begin(115200);
  ledcSetup(kArmServoChannel, kServoFrequency, kServoResolution);
  ledcSetup(kHeadServoChannel, kServoFrequency, kServoResolution);
  ledcSetup(kTurnServoChannel, kServoFrequency, kServoResolution);
  ledcAttachPin(4, kArmServoChannel);
  ledcAttachPin(5, kHeadServoChannel);
  ledcAttachPin(6, kTurnServoChannel);

  if (!initializeClassifier()) {
    Serial.println("Classifier initialization failed");
    while (true) delay(1000);
  }
  Serial.println("Ready. Send one command per line.");
}

void loop() {
  if (!Serial.available()) {
    return;
  }
  const String command = Serial.readStringUntil('\n');
  if (command.length() == 0) {
    return;
  }

  float confidence = 0.0f;
  const int intent = classify(command, &confidence);
  Serial.printf("intent=%s confidence=%.3f\n",
                intent >= 0 ? kIntentNames[intent] : "ERROR", confidence);
  if (intent >= 0 && confidence >= 0.55f) {
    dispatchMotorCommand(intent);
  } else {
    Serial.println("Command rejected: low confidence");
  }
}
