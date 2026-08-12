# 🖐️ Real-Time EMG Gesture Recognition & Control System

[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intelligent, real-time Electromyography (EMG) gesture recognition and IoT control platform. The system acquires simulated/live raw biological muscle signals, extracts 15 time-domain features (Hudgins, Hjorth parameters, DASDV, MYOP), classifies human gestures via a weighted multi-model machine learning ensemble, and interfaces with smart IoT appliances with dynamic calibration and fatigue compensation.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Feature Extraction Pipeline](#-feature-extraction-pipeline)
- [Machine Learning Ensemble](#-machine-learning-ensemble)
- [Interactive Dashboard](#-interactive-dashboard)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Web Dashboard Installation](#web-dashboard-installation)
  - [ML Training Pipeline](#ml-training-pipeline)
- [Gesture Mapping & IoT Controls](#-gesture-mapping--iot-controls)
- [Model Evaluation](#-model-evaluation)
- [License](#-license)

---

## ⚡ Overview

Electromyography (EMG) signals record the electrical activity produced by skeletal muscles during contraction. This project provides a complete end-to-end framework to:
1. **Simulate & Filter Raw EMG**: Synthesize realistic multi-component biopotential signals (burst frequencies, electrode noise floor, baseline DC drift, and random motion artifacts).
2. **Extract Comprehensive Time-Domain Features**: Calculate 15 distinct statistical, morphology, and frequency-domain surrogate metrics.
3. **Classify Hand & Wrist Gestures**: Leverage ensemble learning across Random Forest, Support Vector Machines (SVM), Gradient Boosting, k-NN, and Gaussian Naive Bayes.
4. **Real-Time Interactive Telemetry Dashboard**: Provide a responsive React dashboard with live oscilloscopes, radar charts, confusion matrices, gesture maps, and IoT home automation simulation.
5. **Dynamic User Calibration**: Personalize recognition thresholds per user to mitigate muscle fatigue and electrode shift.

---

## 🏗️ System Architecture

```
   +-----------------------------------------------------------+
   |                Surface EMG Signal Acquisition             |
   |   (Sampling Rate: 500 Hz | Window: 256 samples / ~512ms)  |
   +-----------------------------+-----------------------------+
                                 |
                                 v
   +-----------------------------------------------------------+
   |             Pre-Processing & Feature Extraction           |
   |  • Amplitude & Power (MAV, MMAV, RMS, VAR, STD, IEMG)    |
   |  • Waveform Complexity (WL, AAC, DASDV, ZC, SSC)         |
   |  • Hjorth Parameters (Activity, Mobility, Complexity)     |
   |  • Activation Thresholds (MYOP)                          |
   +-----------------------------+-----------------------------+
                                 |
                                 v
   +-----------------------------------------------------------+
   |           Multi-Model Machine Learning Ensemble           |
   |  • Random Forest (200 trees)   • Gradient Boosting        |
   |  • Support Vector Machine (RBF) • Gaussian Naive Bayes     |
   |  • k-Nearest Neighbors (k=7)   • Soft-Voting Ensemble     |
   +-----------------------------+-----------------------------+
                                 |
                                 v
   +-----------------------------------------------------------+
   |         Real-Time Dashboard & IoT Device Controller        |
   |  • Live Signal Oscilloscope    • Dynamic Calibration      |
   |  • Feature Radar Visualizer    • Rejection Thresholding   |
   |  • Appliance Trigger Matrix    • Latency Telemetry        |
   +-----------------------------------------------------------+
```

---

## 🌟 Key Features

- **6 Gesture Classes**:
  - `FIST` ✊ — Hand clenched firmly
  - `OPEN_HAND` 🖐 — Full finger extension
  - `WRIST_UP` ☝️ — Wrist dorsiflexion
  - `WRIST_DOWN` 👇 — Wrist palmar flexion
  - `DOUBLE_FLEX` 💪 — Simultaneous wrist and forearm contraction
  - `RELAX` ✋ — Resting baseline
- **15 Time-Domain EMG Features**: Combined Hudgins set, Hjorth parameters, DASDV, and Myopulse percentage.
- **Adaptive Calibration Protocol**: 6-step guided calibration recording to adapt baseline RMS and per-gesture scales.
- **Low-Latency Streaming**: Visualizes real-time biological streams at 60 FPS with configurable temporal smoothing.
- **IoT Appliance Control**: Interactive smart home dashboard triggering Lights, Fans, Smart Doors, Motors, TV, and AC units based on gesture triggers.
- **Explainability & Telemetry**: Feature importance charts, real-time confidence scores, and confusion matrix breakdowns.

---

## 🔬 Feature Extraction Pipeline

| Feature | Category | Description | Formula / Principle |
|:---|:---|:---|:---|
| **MAV** | Amplitude | Mean Absolute Value | $\frac{1}{N}\sum |x_i|$ |
| **MMAV** | Amplitude | Modified Mean Absolute Value | Weighted window prioritizing core samples |
| **RMS** | Power | Root Mean Square | $\sqrt{\frac{1}{N}\sum x_i^2}$ |
| **VAR** | Power | Variance of signal | $\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$ |
| **STD** | Power | Standard Deviation | $\sigma$ |
| **IEMG** | Power | Integrated EMG | $\sum |x_i|$ |
| **WL** | Morphology | Waveform Length | $\sum |x_i - x_{i-1}|$ |
| **AAC** | Morphology | Average Amplitude Change | $\frac{1}{N}\sum |x_i - x_{i-1}|$ |
| **DASDV** | Morphology | Difference Absolute Standard Deviation | $\sqrt{\frac{1}{N-1}\sum (x_{i+1} - x_i)^2}$ |
| **ZC** | Frequency | Zero Crossing Count | Count of sign transitions exceeding noise threshold |
| **SSC** | Frequency | Slope Sign Change | Count of turn points exceeding threshold |
| **Hjorth Activity** | Spectral | Signal Variance / Power | $\text{Var}(x(t))$ |
| **Hjorth Mobility** | Spectral | Mean Frequency Estimate | $\sqrt{\text{Var}(x'(t)) / \text{Var}(x(t))}$ |
| **Hjorth Complexity** | Spectral | Spectral Bandwidth | $\text{Mobility}(x'(t)) / \text{Mobility}(x(t))$ |
| **MYOP** | Threshold | Myopulse Percentage Rate | Fraction of samples exceeding $3 \times \text{noise}$ |

---

## 🧠 Machine Learning Ensemble

The pipeline trains and benchmarks 5 primary classifiers using `scikit-learn`:

```python
# Model Evaluation Pipeline (5-Fold Stratified Cross-Validation)
VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=10)),
        ('gb',  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)),
        ('svm', Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, C=10, gamma='scale'))])),
        ('knn', Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=7))])),
        ('gnb', GaussianNB())
    ],
    voting='soft'
)
```

---

## 📊 Interactive Dashboard

The dashboard (`EMGDashboard_v5.jsx`) features:
1. **Live Signal Scope**: Oscilloscope displaying real-time raw biopotentials and moving RMS energy envelopes.
2. **Predicted State & Confidence Meter**: High-visibility gesture cards with real-time class probability distribution.
3. **Feature Radar & Metric Gauges**: Polar radar displaying normalized feature vectors alongside live scalar readings.
4. **Smart Appliance Control Grid**: Interactive state toggles for home automation devices triggered by recognized gestures.
5. **Interactive Calibration Wizard**: Guided 6-gesture calibration workflow with baseline drift subtraction.
6. **Model Telemetry & Confusion Matrix**: Visual accuracy charts, cross-validation metrics, and feature importances.

---

## 📁 Project Structure

```
Real-Time-EMG-Gesture-Control/
├── src/
│   ├── EMGDashboard_v5.jsx      # Core React interactive dashboard component
│   └── main.jsx                 # React DOM mount entry point
├── index.html                   # HTML5 document template & Google fonts
├── vite.config.js               # Vite build configuration
├── package.json                 # Project dependencies & scripts
├── train_model_v2.py            # Python ML training & feature extraction pipeline
├── emg_model_v2.pkl             # Serialized trained model weights
├── model_meta_v2.json           # Model metadata, CV benchmarks & confusion matrix
├── feature_weights.json         # Feature importance weights & normalization bounds
├── .gitignore                   # Ignored files (node_modules, caches, builds)
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js**: v18.0 or higher
- **npm**: v9.0 or higher
- **Python**: v3.9 or higher (for retraining models)

### Web Dashboard Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prathameshmowade/Real-Time-EMG-Gesture-Control.git
   cd Real-Time-EMG-Gesture-Control
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   Navigate to `http://localhost:5173/` to view the live dashboard.

### ML Training Pipeline

To re-run the dataset generation, feature extraction, and ensemble model training:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy pandas scikit-learn joblib

# Run training pipeline
python train_model_v2.py
```

---

## 🎮 Gesture Mapping & IoT Controls

| Device | Action ON | Action OFF | Status Indicator |
|:---|:---|:---|:---:|
| 💡 **Smart Light** | `FIST` ✊ | `OPEN_HAND` 🖐 | Active / Inactive |
| 🌀 **Ceiling Fan** | `WRIST_UP` ☝️ | `WRIST_DOWN` 👇 | RPM Speed Gauge |
| 🚪 **Smart Door** | `WRIST_DOWN` 👇 | `RELAX` ✋ | Locked / Unlocked |
| ⚙️ **Stepper Motor** | `DOUBLE_FLEX` 💪 | `OPEN_HAND` 🖐 | Running / Stopped |
| 📺 **Smart TV** | `OPEN_HAND` 🖐 | `RELAX` ✋ | On / Off |
| ❄️ **AC Unit** | `DOUBLE_FLEX` 💪 | `WRIST_UP` ☝️ | Cooling / Standby |

---

## 📈 Model Evaluation

- **Cross-Validation**: 5-Fold Stratified Cross-Validation on multi-user dataset.
- **Top Predictive Features**: Hjorth Activity (11.5%), MMAV (11.1%), MAV (11.0%), RMS (10.5%), IEMG (9.7%).
- **Inference Latency**: < 15ms per 256-sample window.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
