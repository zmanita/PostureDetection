#include <TensorFlowLite.h>

#include "Arduino_BMI270_BMM150.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

#define WINDOW_SIZE 20
#define TENSOR_ARENA_SIZE 8 * 1024

const char* POSTURE_CLASSES[] = {"supine", "prone", "side", "sitting", "unknown"};
const int NUM_CLASSES = 5;

// Sensor selection enum
enum SensorType {
  ACCELEROMETER,
  GYROSCOPE,
  MAGNETOMETER,
  ALL_SENSORS
};

// ============================================================================
// NORMALIZATION PARAMETERS
// ============================================================================

const float ACCEL_MEANS[3] = {
  0.03924891548242334f,
  -0.01963246073298429f,
  0.15387240089753182f
};

const float ACCEL_STDS[3] = {
  0.46838459864325566f,
  0.3950920103471127f,
  0.6539748567927205f
};

const float GYRO_MEANS[3] = {
  0.3399690351533283f,
  0.33855317875841434f,
  -0.003573074046372476f
};

const float GYRO_STDS[3] = {
  5.29962428551057f,
  2.478528024927504f,
  1.0609240032968694f
};

const float MAG_MEANS[3] = {
  -0.6629917726252805f,
  -1.7085265519820494f,
  -1.7814061331338817f
};

const float MAG_STDS[3] = {
  28.614663937222847f,
  27.753577689171596f,
  32.02987942321389f
};

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float accelX_buffer[WINDOW_SIZE], accelY_buffer[WINDOW_SIZE], accelZ_buffer[WINDOW_SIZE];
float gyroX_buffer[WINDOW_SIZE], gyroY_buffer[WINDOW_SIZE], gyroZ_buffer[WINDOW_SIZE];
float magX_buffer[WINDOW_SIZE], magY_buffer[WINDOW_SIZE], magZ_buffer[WINDOW_SIZE];

int bufferIndex = 0;
bool isCollecting = false;
unsigned long collectionStartTime = 0;
bool waitingForSensorSelection = false;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("=== TFLite Posture Detection - Arduino Nano 33 BLE ===");
  Serial.println("Initializing IMU...");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  // Load TFLite model
  Serial.println("Loading TFLite model...");
  model = tflite::GetModel(model_tflite);
  
  if (model == nullptr) {
    Serial.println("Failed to load model!");
    while (1);
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("Failed to get input/output tensors!");
    while (1);
  }

  Serial.println("✓ Model loaded successfully!");
  Serial.println();
  printMainMenu();
}

// ============================================================================
// MENU FUNCTIONS
// ============================================================================

void printMainMenu() {
  Serial.println("Commands:");
  Serial.println("  1 - Collect data and predict posture");
  Serial.println("  2 - Collect raw data (no inference)");
  Serial.println("  3 - Print normalization parameters");
  Serial.println();
}

void printSensorMenu() {
  Serial.println("\n=== SELECT SENSOR FOR PREDICTION ===");
  Serial.println("  a - Prediction using Accelerometer");
  Serial.println("  b - Prediction using Gyroscope");
  Serial.println("  c - Prediction using Magnetometer");
  Serial.println("  d - Prediction using All Sensors");
  Serial.println("====================================");
  Serial.print("Enter your choice: ");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    // Skip newline and carriage return characters
    if (command == '\n' || command == '\r') {
      return;
    }

    if (waitingForSensorSelection) {
      handleSensorSelection(command);
      waitingForSensorSelection = false;
      // Clear any remaining characters in serial buffer
      while(Serial.available() > 0) {
        Serial.read();
      }
    } else {
      if (command == '1') {
        waitingForSensorSelection = true;
        printSensorMenu();
        // Clear any remaining characters in serial buffer
        while(Serial.available() > 0) {
          Serial.read();
        }
      } else if (command == '2') {
        collectRawData();
      } else if (command == '3') {
        printNormalizationParams();
      }
    }
  }
}

// ============================================================================
// SENSOR SELECTION HANDLER
// ============================================================================

void handleSensorSelection(char choice) {
  SensorType selectedSensor;
  
  switch(choice) {
    case 'a':
    case 'A':
      Serial.println("a");
      Serial.println("Selected: Accelerometer");
      selectedSensor = ACCELEROMETER;
      break;
    case 'b':
    case 'B':
      Serial.println("b");
      Serial.println("Selected: Gyroscope");
      selectedSensor = GYROSCOPE;
      break;
    case 'c':
    case 'C':
      Serial.println("c");
      Serial.println("Selected: Magnetometer");
      selectedSensor = MAGNETOMETER;
      break;
    case 'd':
    case 'D':
      Serial.println("d");
      Serial.println("Selected: All Sensors");
      selectedSensor = ALL_SENSORS;
      break;
    default:
      Serial.println("\nInvalid selection! Please try again.");
      printMainMenu();
      return;
  }
  
  collectAndPredict(selectedSensor);
  printMainMenu();
}

// ============================================================================
// STEP 1: COLLECT IMU DATA
// ============================================================================

void collectAndPredict(SensorType sensorType) {
  Serial.println("\nStarting 2-second data collection...");
  bufferIndex = 0;
  collectionStartTime = millis();
  isCollecting = true;

  while (isCollecting && bufferIndex < WINDOW_SIZE) {
    float accelX, accelY, accelZ;
    float gyroX, gyroY, gyroZ;
    float magX, magY, magZ;

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
      IMU.readAcceleration(accelX, accelY, accelZ);
      IMU.readGyroscope(gyroX, gyroY, gyroZ);
      IMU.readMagneticField(magX, magY, magZ);

      accelX_buffer[bufferIndex] = accelX;
      accelY_buffer[bufferIndex] = accelY;
      accelZ_buffer[bufferIndex] = accelZ;
      gyroX_buffer[bufferIndex] = gyroX;
      gyroY_buffer[bufferIndex] = gyroY;
      gyroZ_buffer[bufferIndex] = gyroZ;
      magX_buffer[bufferIndex] = magX;
      magY_buffer[bufferIndex] = magY;
      magZ_buffer[bufferIndex] = magZ;

      bufferIndex++;
      Serial.print(".");
    }

    if (millis() - collectionStartTime > 3000) {
      isCollecting = false;
    }
  }

  isCollecting = false;
  Serial.println();
  Serial.print("Collected ");
  Serial.print(bufferIndex);
  Serial.println(" samples");

  if (bufferIndex < WINDOW_SIZE) {
    Serial.println("ERROR: Not enough samples collected!");
    return;
  }

  extractFeaturesAndPredict(sensorType);
}

// ============================================================================
// STEP 2: EXTRACT FEATURES
// ============================================================================

void extractFeaturesAndPredict(SensorType sensorType) {
  Serial.println("Extracting and normalizing features...");

  float features[9];
  int featureCount = 0;

  // Calculate means based on selected sensor
  if (sensorType == ACCELEROMETER || sensorType == ALL_SENSORS) {
    float accel_x_mean = calculateMean(accelX_buffer, WINDOW_SIZE);
    float accel_y_mean = calculateMean(accelY_buffer, WINDOW_SIZE);
    float accel_z_mean = calculateMean(accelZ_buffer, WINDOW_SIZE);

    features[0] = (accel_x_mean - ACCEL_MEANS[0]) / ACCEL_STDS[0];
    features[1] = (accel_y_mean - ACCEL_MEANS[1]) / ACCEL_STDS[1];
    features[2] = (accel_z_mean - ACCEL_MEANS[2]) / ACCEL_STDS[2];
    featureCount = 3;
  } else {
    features[0] = 0.0f;
    features[1] = 0.0f;
    features[2] = 0.0f;
  }

  if (sensorType == GYROSCOPE || sensorType == ALL_SENSORS) {
    float gyro_x_mean = calculateMean(gyroX_buffer, WINDOW_SIZE);
    float gyro_y_mean = calculateMean(gyroY_buffer, WINDOW_SIZE);
    float gyro_z_mean = calculateMean(gyroZ_buffer, WINDOW_SIZE);

    features[3] = (gyro_x_mean - GYRO_MEANS[0]) / GYRO_STDS[0];
    features[4] = (gyro_y_mean - GYRO_MEANS[1]) / GYRO_STDS[1];
    features[5] = (gyro_z_mean - GYRO_MEANS[2]) / GYRO_STDS[2];
    featureCount = (sensorType == ALL_SENSORS) ? 6 : 3;
  } else {
    features[3] = 0.0f;
    features[4] = 0.0f;
    features[5] = 0.0f;
  }

  if (sensorType == MAGNETOMETER || sensorType == ALL_SENSORS) {
    float mag_x_mean = calculateMean(magX_buffer, WINDOW_SIZE);
    float mag_y_mean = calculateMean(magY_buffer, WINDOW_SIZE);
    float mag_z_mean = calculateMean(magZ_buffer, WINDOW_SIZE);

    features[6] = (mag_x_mean - MAG_MEANS[0]) / MAG_STDS[0];
    features[7] = (mag_y_mean - MAG_MEANS[1]) / MAG_STDS[1];
    features[8] = (mag_z_mean - MAG_MEANS[2]) / MAG_STDS[2];
    featureCount = (sensorType == ALL_SENSORS) ? 9 : 3;
  } else {
    features[6] = 0.0f;
    features[7] = 0.0f;
    features[8] = 0.0f;
  }

  // Run inference
  runInference(features, sensorType);
}

// ============================================================================
// STEP 3: RUN TFLITE INFERENCE
// ============================================================================

void runInference(float* features, SensorType sensorType) {
  Serial.println("Running TFLite inference...");

  // Display which sensor is being used
  Serial.print("Using sensor(s): ");
  switch(sensorType) {
    case ACCELEROMETER:
      Serial.println("Accelerometer only");
      break;
    case GYROSCOPE:
      Serial.println("Gyroscope only");
      break;
    case MAGNETOMETER:
      Serial.println("Magnetometer only");
      break;
    case ALL_SENSORS:
      Serial.println("All sensors (Accelerometer + Gyroscope + Magnetometer)");
      break;
  }

  // Copy features to input tensor
  float* input_data = input->data.f;
  for (int i = 0; i < 9; i++) {
    input_data[i] = features[i];
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  // Get output probabilities
  float* output_data = output->data.f;

  Serial.println("\n=== INFERENCE RESULTS ===");
  Serial.println("Class Probabilities:");

  int predicted_class = 0;
  float max_probability = output_data[0];

  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(POSTURE_CLASSES[i]);
    Serial.print(": ");
    Serial.println(output_data[i], 4);

    if (output_data[i] > max_probability) {
      max_probability = output_data[i];
      predicted_class = i;
    }
  }

  Serial.println();
  Serial.print("✓ PREDICTED POSTURE: ");
  Serial.println(POSTURE_CLASSES[predicted_class]);
  Serial.print("  Confidence: ");
  Serial.println(max_probability, 4);
  Serial.println("========================\n");
}

// ============================================================================
// UTILITY: Math functions
// ============================================================================

float calculateMean(float* data, int size) {
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += data[i];
  }
  return sum / size;
}

// ============================================================================
// UTILITY: COLLECT RAW DATA
// ============================================================================

void collectRawData() {
  Serial.println("Collecting 20 raw samples...");
  bufferIndex = 0;
  isCollecting = true;

  while (isCollecting && bufferIndex < WINDOW_SIZE) {
    float accelX, accelY, accelZ;
    float gyroX, gyroY, gyroZ;
    float magX, magY, magZ;

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
      IMU.readAcceleration(accelX, accelY, accelZ);
      IMU.readGyroscope(gyroX, gyroY, gyroZ);
      IMU.readMagneticField(magX, magY, magZ);

      Serial.print("Sample ");
      Serial.print(bufferIndex);
      Serial.print(": A(");
      Serial.print(accelX, 3);
      Serial.print(",");
      Serial.print(accelY, 3);
      Serial.print(",");
      Serial.print(accelZ, 3);
      Serial.print(") G(");
      Serial.print(gyroX, 3);
      Serial.print(",");
      Serial.print(gyroY, 3);
      Serial.print(",");
      Serial.print(gyroZ, 3);
      Serial.print(") M(");
      Serial.print(magX, 1);
      Serial.print(",");
      Serial.print(magY, 1);
      Serial.print(",");
      Serial.print(magZ, 1);
      Serial.println(")");

      bufferIndex++;
    }
  }

  Serial.println("Collection complete.\n");
  printMainMenu();
}

// ============================================================================
// UTILITY: PRINT NORMALIZATION PARAMETERS
// ============================================================================

void printNormalizationParams() {
  Serial.println("\n=== NORMALIZATION PARAMETERS ===");
  Serial.println("Accelerometer Means: [0.039, -0.020, 0.154]");
  Serial.println("Accelerometer Stds:  [0.468, 0.395, 0.654]");
  Serial.println("Gyroscope Means:     [0.340, 0.339, -0.004]");
  Serial.println("Gyroscope Stds:      [5.300, 2.479, 1.061]");
  Serial.println("Magnetometer Means:  [-0.663, -1.709, -1.781]");
  Serial.println("Magnetometer Stds:   [28.615, 27.754, 32.030]");
  Serial.println("================================\n");
  printMainMenu();
}