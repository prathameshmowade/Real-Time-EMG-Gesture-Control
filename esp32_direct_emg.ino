/*
======================================================================
  REAL-TIME EMG GESTURE CONTROL: DIRECT SINGLE-BOARD ESP32 FIRMWARE
  Directly connects EMG Sensor V3.0 to ESP32 DevKit V1
  Actuates 1 Servo Motor (GPIO 13) + 1 LED (GPIO 2)
  Streams live serial telemetry to Web Dashboard & Python
======================================================================
  Hardware Pin Connections on ESP32 DevKit V1:
  1. EMG Sensor V3.0:
     - VCC (+)  -> ESP32 3V3 (or VIN 5V)
     - GND (-)  -> ESP32 GND
     - SIG (A)  -> ESP32 GPIO 34 (ADC1_CH6)

  2. Servo Motor (SG90 / MG90S):
     - Brown / Black -> ESP32 GND
     - Red           -> ESP32 VIN (5V)
     - Orange/Yellow -> ESP32 GPIO 13 (PWM)

  3. Indicator LED:
     - Built-in LED -> ESP32 GPIO 2
======================================================================
*/

#include <ESP32Servo.h>

#define EMG_PIN     34     // Analog ADC1 pin for EMG Sensor V3.0
#define SERVO_PIN   13     // PWM pin for 1 Servo Motor
#define LED_PIN      2     // Built-in Blue LED

Servo myServo;

// Sampling parameters (500 Hz)
const int SAMPLE_RATE = 500;
const int SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE; // 2000 us (2 ms)
const int WINDOW_SIZE = 50; // 100ms window

float buffer[WINDOW_SIZE];
int bufIdx = 0;
float baseline = 1.65f;
float noiseFloor = 0.05f;

unsigned long lastSampleTime = 0;
unsigned long lastPrintTime = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Initialize Servo
  ESP32PWM::allocateTimer(0);
  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(0); // Start relaxed at 0 deg

  // Configure ADC (12-bit: 0 - 4095, 0.0 - 3.3V)
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Full 0.0V - 3.3V range

  Serial.println("\n=======================================================");
  Serial.println("  [+] ESP32 DIRECT EMG ACQUISITION & CONTROL STARTED");
  Serial.println("=======================================================");

  // 1. Calibrate resting baseline for 1 second (Keep arm relaxed)
  Serial.println("[*] Calibrating resting baseline on GPIO 34 (Relax arm)...");
  float sum = 0.0f;
  for (int i = 0; i < 500; i++) {
    int raw = analogRead(EMG_PIN);
    sum += (raw / 4095.0f) * 3.3f;
    delay(2);
  }
  baseline = sum / 500.0f;

  // Calculate resting noise floor
  float noiseSum = 0.0f;
  for (int i = 0; i < 200; i++) {
    int raw = analogRead(EMG_PIN);
    float v = (raw / 4095.0f) * 3.3f;
    noiseSum += pow(v - baseline, 2);
    delay(2);
  }
  noiseFloor = sqrt(noiseSum / 200.0f);
  if (noiseFloor < 0.03f) noiseFloor = 0.03f;

  Serial.printf("[+] Baseline DC Offset: %.3f V | Noise Floor: %.3f V\n", baseline, noiseFloor);
  Serial.println("[+] System Ready! Reading Muscle Biopotentials...\n");
}

void loop() {
  unsigned long now = micros();

  // Enforce precise 500 Hz sampling interval (every 2000 microseconds)
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;

    int rawADC = analogRead(EMG_PIN);
    float voltage = (rawADC / 4095.0f) * 3.3f;
    float signal = voltage - baseline; // Zero-centered biopotential

    buffer[bufIdx++] = signal;

    // Window processing every 50 samples (10 times per second)
    if (bufIdx >= WINDOW_SIZE) {
      bufIdx = 0;

      // Calculate RMS energy
      float sumSq = 0.0f;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        sumSq += buffer[i] * buffer[i];
      }
      float rms = sqrt(sumSq / WINDOW_SIZE);

      // Classify basic gestures
      String gesture = "RELAX";
      if (rms > 0.45f) {
        gesture = "FIST";
      } else if (rms > (noiseFloor * 2.5f) && rms <= 0.45f) {
        gesture = "OPEN_HAND";
      } else {
        gesture = "RELAX";
      }

      // Actuate Servo & LED
      if (gesture == "FIST") {
        myServo.write(180);
        digitalWrite(LED_PIN, HIGH);
      } else if (gesture == "OPEN_HAND") {
        myServo.write(90);
        digitalWrite(LED_PIN, HIGH);
      } else {
        myServo.write(0);
        digitalWrite(LED_PIN, LOW);
      }

      // Print live telemetry to Serial Monitor & Web Dashboard
      Serial.printf("[EMG] Gesture: >> %-10s << | RMS: %.3f V | Raw: %.3f V\n", gesture.c_str(), rms, voltage);
    }
  }
}
