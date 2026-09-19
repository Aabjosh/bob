#include <Arduino.h>
#include <cctype>
#include <cstring>
#include <TensorFlowLite_ESP32.h>
#include "generated/model_data.h"
#include "generated/vocab.h"
#include "generated/story_starters.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"
#include "esp_system.h"

namespace {
constexpr int kIntentCount = 13;
constexpr int kMaxQueuedCommands = 8;
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
    "ARM_UP", "LEFT_ARM", "RIGHT_ARM", "MOVE_FORWARD", "MOVE_BACKWARD",
    "HEAD_SHAKE", "HEAD_NOD", "HEAD_LEFT", "HEAD_RIGHT", "TURN_LEFT",
    "TURN_RIGHT", "DANCE", "STOP",
};
String serial_buffer;

struct MotorCommand {
  int intent_id;
  int value;
  uint32_t duration_ms;
};

MotorCommand command_queue[kMaxQueuedCommands];
int queue_head = 0;
int queue_tail = 0;
int queued_command_count = 0;

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

int ruleBasedIntent(const String &raw_text) {
  String lowered = raw_text;
  lowered.toLowerCase();
  if (lowered.indexOf("arm") >= 0 && lowered.indexOf("left") >= 0) {
    return 1;  // LEFT_ARM
  }
  if (lowered.indexOf("arm") >= 0 && lowered.indexOf("right") >= 0) {
    return 2;  // RIGHT_ARM
  }
  const bool has_head = lowered.indexOf("head") >= 0;
  if (has_head && (lowered.indexOf("shake") >= 0 ||
                   (lowered.indexOf("left") >= 0 && lowered.indexOf("right") >= 0))) {
    return 5;  // HEAD_SHAKE
  }
  if (has_head && (lowered.indexOf("nod") >= 0 || lowered.indexOf("bob") >= 0 ||
                   lowered.indexOf("yes") >= 0)) {
    return 6;  // HEAD_NOD
  }
  if (has_head && lowered.indexOf("left") >= 0) {
    return 7;  // HEAD_LEFT
  }
  if (has_head && lowered.indexOf("right") >= 0) {
    return 8;  // HEAD_RIGHT
  }
  if (lowered.indexOf("stop") >= 0 || lowered.indexOf("halt") >= 0 ||
      lowered.indexOf("freeze") >= 0 || lowered.indexOf("hold position") >= 0) {
    return 12;  // STOP
  }
  return -1;
}

bool isStoryRequest(const String &raw_text) {
  String lowered = raw_text;
  lowered.toLowerCase();
  const bool asks_for_story = lowered.indexOf("story") >= 0 || lowered.indexOf("tale") >= 0;
  const bool asks_to_generate = lowered.indexOf("tell") >= 0 || lowered.indexOf("read") >= 0 ||
                                lowered.indexOf("make") >= 0 || lowered.indexOf("write") >= 0 ||
                                lowered.indexOf("start") >= 0 || lowered.indexOf("give") >= 0;
  return asks_for_story && asks_to_generate;
}

void emitRandomStoryStarter() {
  const uint32_t index = esp_random() % STORY_STARTER_COUNT;
  Serial.print("STORY_STARTER ");
  Serial.println(kStoryStarters[index]);
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
  const int rule_intent = ruleBasedIntent(raw_text);
  if (rule_intent >= 0) {
    *confidence = 1.0f;
    return rule_intent;
  }
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

void dispatchMotorCommand(int intent_id, int value, uint32_t duration_ms) {
  const int bounded_value = constrain(value, 0, 180);
  switch (intent_id) {
    case 0:  // ARM_UP
      writeServoAngle(kArmServoChannel, bounded_value);
      Serial.println("ARM_UP");
      break;
    case 1:  // LEFT_ARM
      writeServoAngle(kArmServoChannel, bounded_value);
      Serial.println("LEFT_ARM");
      break;
    case 2:  // RIGHT_ARM
      writeServoAngle(kArmServoChannel, bounded_value);
      Serial.println("RIGHT_ARM");
      break;
    case 3:  // MOVE_FORWARD
      writeServoAngle(kTurnServoChannel, 90 + bounded_value / 2);
      Serial.println("MOVE_FORWARD");
      break;
    case 4:  // MOVE_BACKWARD
      writeServoAngle(kTurnServoChannel, 90 - bounded_value / 2);
      Serial.println("MOVE_BACKWARD");
      break;
    case 5:  // HEAD_SHAKE
      writeServoAngle(kHeadServoChannel, 60);
      delay(duration_ms / 2);
      writeServoAngle(kHeadServoChannel, 120);
      Serial.println("HEAD_SHAKE");
      break;
    case 6:  // HEAD_NOD
      writeServoAngle(kHeadServoChannel, 70);
      delay(duration_ms / 2);
      writeServoAngle(kHeadServoChannel, 110);
      Serial.println("HEAD_NOD");
      break;
    case 7:  // HEAD_LEFT
      writeServoAngle(kHeadServoChannel, 90 - bounded_value / 2);
      Serial.println("HEAD_LEFT");
      break;
    case 8:  // HEAD_RIGHT
      writeServoAngle(kHeadServoChannel, 90 + bounded_value / 2);
      Serial.println("HEAD_RIGHT");
      break;
    case 9:  // TURN_LEFT
      writeServoAngle(kTurnServoChannel, 90 - bounded_value / 2);
      Serial.println("TURN_LEFT");
      break;
    case 10:  // TURN_RIGHT
      writeServoAngle(kTurnServoChannel, 90 + bounded_value / 2);
      Serial.println("TURN_RIGHT");
      break;
    case 11:  // DANCE
      writeServoAngle(kArmServoChannel, 60);
      writeServoAngle(kHeadServoChannel, 120);
      delay(duration_ms / 2);
      writeServoAngle(kArmServoChannel, 120);
      writeServoAngle(kHeadServoChannel, 60);
      Serial.println("DANCE");
      break;
    case 12:  // STOP
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

int intentIdFromName(const String &name) {
  for (int intent_id = 0; intent_id < kIntentCount; ++intent_id) {
    if (name.equals(kIntentNames[intent_id])) {
      return intent_id;
    }
  }
  return -1;
}

void clearCommandQueue() {
  queue_head = 0;
  queue_tail = 0;
  queued_command_count = 0;
}

bool enqueueCommand(int intent_id, int value, uint32_t duration_ms) {
  if (queued_command_count >= kMaxQueuedCommands) {
    return false;
  }
  const int next_tail = (queue_tail + 1) % kMaxQueuedCommands;
  command_queue[queue_tail] = {intent_id, constrain(value, 0, 180), duration_ms};
  queue_tail = next_tail;
  ++queued_command_count;
  return true;
}

bool parsePlan(String plan) {
  clearCommandQueue();
  plan.trim();
  if (!plan.startsWith("PLAN ")) {
    return false;
  }
  plan.remove(0, 5);
  while (plan.length() > 0) {
    const int separator = plan.indexOf(';');
    String item = separator >= 0 ? plan.substring(0, separator) : plan;
    plan = separator >= 0 ? plan.substring(separator + 1) : "";
    item.trim();
    const int first_comma = item.indexOf(',');
    const int second_comma = item.indexOf(',', first_comma + 1);
    if (first_comma < 1 || second_comma < 0) {
      clearCommandQueue();
      return false;
    }
    const int intent_id = intentIdFromName(item.substring(0, first_comma));
    const int value = item.substring(first_comma + 1, second_comma).toInt();
    const uint32_t duration_ms = static_cast<uint32_t>(item.substring(second_comma + 1).toInt());
    if (intent_id < 0 || duration_ms == 0 || !enqueueCommand(intent_id, value, duration_ms)) {
      clearCommandQueue();
      return false;
    }
    if (intent_id == 12) {
      break;
    }
  }
  return queue_head != queue_tail;
}

void processCommandQueue() {
  if (queued_command_count == 0) {
    return;
  }
  const MotorCommand command = command_queue[queue_head];
  queue_head = (queue_head + 1) % kMaxQueuedCommands;
  --queued_command_count;
  dispatchMotorCommand(command.intent_id, command.value, command.duration_ms);
}

bool readSerialCommand(String *command) {
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\n') {
      *command = serial_buffer;
      serial_buffer = "";
      command->trim();
      return command->length() > 0;
    }
    if (character != '\r' && serial_buffer.length() < 128) {
      serial_buffer += character;
    }
  }
  return false;
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
  Serial.println("Ready. Send one command per line at 115200 baud.");
}

void loop() {
  processCommandQueue();
  String command;
  if (!readSerialCommand(&command)) {
    return;
  }

  if (command.startsWith("PLAN ")) {
    if (parsePlan(command)) {
      Serial.println("Plan accepted");
    } else {
      Serial.println("Plan rejected");
    }
    return;
  }

  if (isStoryRequest(command)) {
    emitRandomStoryStarter();
    return;
  }

  float confidence = 0.0f;
  const int intent = classify(command, &confidence);
  Serial.printf("intent=%s confidence=%.3f\n",
                intent >= 0 ? kIntentNames[intent] : "ERROR", confidence);
  if (intent >= 0 && confidence >= 0.55f) {
    enqueueCommand(intent, 90, 500);
  } else {
    Serial.println("Command rejected: low confidence");
  }
}
