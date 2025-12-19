# PostureDetection
This project implements an end-to-end posture detection system using IMU sensor data collected from the Arduino Nano 33 BLE Sense. A lightweight neural network is trained offline and deployed on the microcontroller using TensorFlow Lite Micro to perform real-time posture classification across multiple body orientations and sensor configurations.
# Real-Time Posture Detection Using IMU Sensors (Arduino Nano 33 BLE)

This repository contains the full implementation of a real-time posture detection system using embedded IMU sensor data and a lightweight neural network deployed on the Arduino Nano 33 BLE Sense. The project demonstrates the complete TinyML workflow—from data collection and model training to model conversion and on-device inference.

---

## 📌 Project Overview

The goal of this project is to classify human postures in real time using inertial sensor data collected from the Arduino Nano 33 BLE Sense. The system is designed to:

- Collect raw IMU data (accelerometer, gyroscope, magnetometer)
- Train a neural network to classify postures offline
- Convert and deploy the trained model onto a microcontroller
- Perform real-time posture inference directly on the Arduino board

The project emphasizes **cross-platform integration**, **sensor robustness**, and **embedded deployment constraints**, making it a practical example of TinyML in action.

---

## 🧍‍♂️ Target Postures

The system classifies the following postures:

- Supine  
- Prone  
- Side (Left / Right)  
- Sitting  
- Unknown / Transitional  

To improve robustness, data was collected across multiple sensor orientations and natural transitions between postures.

---

## 📊 Data Collection & Processing

- **Hardware:** Arduino Nano 33 BLE Sense  
- **Sensors:** Accelerometer, Gyroscope, Magnetometer  
- **Sampling Rate:** ~10 Hz (limited by magnetometer)  
- **Windowing:** Feature extraction over fixed time windows  
- **Features:** Mean X, Y, Z values from a single selected sensor  
- **Normalization:** Z-score normalization applied during training  

Raw sensor data was initially logged to `.rtf` files, parsed using Python, and consolidated into a single CSV dataset for training.

---

## 🧠 Model Architecture

The posture classifier is a lightweight fully connected neural network designed for embedded deployment:

- **Input Layer:** 3 neurons (X, Y, Z sensor channels)
- **Hidden Layers:** 2 layers with ReLU activation
- **Output Layer:** Softmax activation for multi-class classification
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy

The model was intentionally kept simple to ensure compatibility with microcontroller memory and compute constraints.

---

## 🔁 Training & Evaluation

- **Dataset Split:** 60% training / 20% validation / 20% testing
- **Final Test Accuracy:** ~93.5%
- **Test Loss:** 0.2046

Training and validation curves show stable convergence and minimal overfitting in the offline environment.

---

## 🚀 Model Deployment

The trained model was converted using **TensorFlow Lite Micro** and deployed on the Arduino Nano 33 BLE Sense:

1. Trained model exported from Python
2. Converted to `.tflite` format
3. Transformed into a C byte array (`.h` file)
4. Integrated into the Arduino project
5. Real-time inference performed on streaming IMU data

A menu-driven interface allows users to:
- Select posture prediction mode
- Choose which sensor to use
- Run real-time inference on demand

---

## ⚠️ Deployment Challenges

Several real-world issues were encountered during deployment:

- Magnetometer’s low sampling frequency limited data diversity
- Dataset bias toward controlled board orientations
- Mismatch between training-time and inference-time normalization
- Cross-platform compatibility issues with Arduino IDE and TFLite Micro
- Occasional corruption of generated `.h` model files

These challenges led to a **significant accuracy drop on-device (~15%)**, highlighting the gap between offline evaluation and real-world embedded inference.

---

## 📈 Results & Discussion

While the trained model performed well in a controlled testing environment, real-time deployment revealed key limitations related to data diversity, sensor behavior, and system integration. Despite this, the project successfully demonstrates:

- End-to-end TinyML workflow
- Embedded neural network deployment
- Real-time posture inference on microcontrollers
- Practical constraints of IMU-based posture recognition

---

## 🔮 Future Improvements

- Collect larger and more diverse datasets
- Incorporate sensor fusion instead of single-sensor inputs
- Improve real-time normalization consistency
- Handle asynchronous sensor availability more robustly
- Explore temporal models (e.g., RNNs) for sequence modeling

---


## 📚 References

- Arduino Nano 33 BLE Sense ML Documentation  
- TensorFlow Lite Micro  
- IMU-based Posture Recognition Literature  

---

## 👤 Author

**Zarin Musarrat Manita**  
Arizona State University

---

This project was developed as part of a graduate-level embedded machine learning course and serves as a practical demonstration of TinyML system design.
