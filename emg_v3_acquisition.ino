/*
 ======================================================================
  EMG SENSOR V3.0 (DFRobot / OYMotion / Grove / CJMCU) FIRMWARE
  Tailored specifically for EMG Sensor V3.0 analog modules.
  Includes DC-bias baseline auto-centering, 500 Hz precise sampling,
  and USB Serial streaming to the Real-Time ML Gesture Classifier.
 ======================================================================
 Hardware Wiring for EMG Sensor V3.0:
   EMG V3.0 Pin    ->  Arduino Uno / Nano Pin  ->  ESP32 / Pico W Pin
   * + / VCC       ->  5V (or 3.3V)            ->  3.3V (VBUS)
   * - / GND       ->  GND                     ->  GND
   * A / SIG / OUT ->  Analog Pin A0           ->  GPIO 34 (Pico: GP26)

 3-Lead Electrode Placement:
   * RED   (Positive Lead)  -> Center of Target Muscle Belly (e.g., Forearm)
   * BLUE  (Negative Lead)  -> 2 cm along the same muscle fiber
   * BLACK (Reference/GND)  -> Bony area (Elbow bone or Wrist bone)
 ======================================================================
*/

const int EMG_PIN = A0;             // Analog input from EMG Sensor V3.0
const int SAMPLE_RATE_HZ = 500;     // 500 Samples per second (Standard EMG)
const unsigned long SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ; // 2000 us

// Baseline DC-Offset Tracking (EMG V3.0 centers biopotentials at VCC/2)
float baselineOffset = 2.5f;        // 2.5V on 5V Arduino, 1.65V on 3.3V ESP32
float alpha = 0.002f;               // Adaptive baseline high-pass filter constant

unsigned long previousMicros = 0;
bool isCalibrated = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    ; // Wait for USB Serial connection
  }

  #if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
    analogReadResolution(12);       // 12-bit ADC (0 - 4095) for ESP32/Pico
    baselineOffset = 1.65f;
  #else
    baselineOffset = 2.5f;          // 10-bit ADC (0 - 1023) for 5V Arduino Uno/Nano
  #endif

  // Initial 1-second resting baseline calibration
  long sumADC = 0;
  int numSamples = 500;
  for (int i = 0; i < numSamples; i++) {
    sumADC += analogRead(EMG_PIN);
    delay(2);
  }
  
  #if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
    baselineOffset = ((float)sumADC / numSamples / 4095.0f) * 3.3f;
  #else
    baselineOffset = ((float)sumADC / numSamples / 1023.0f) * 5.0f;
  #endif

  isCalibrated = true;
}

void loop() {
  unsigned long currentMicros = micros();

  // Enforce rigid 500 Hz sampling timer
  if (currentMicros - previousMicros >= SAMPLE_INTERVAL_US) {
    previousMicros = currentMicros;

    // 1. Read Raw ADC from EMG V3.0
    int rawADC = analogRead(EMG_PIN);

    // 2. Convert ADC to physical voltage
    #if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
      float voltage = ((float)rawADC / 4095.0f) * 3.3f;
    #else
      float voltage = ((float)rawADC / 1023.0f) * 5.0f;
    #endif

    // 3. Remove DC-bias baseline (Zero-centered biopotential action signal)
    baselineOffset = (1.0f - alpha) * baselineOffset + alpha * voltage;
    float emgSignal = voltage - baselineOffset;

    // 4. Stream zero-centered biopotential to PC over USB Serial
    Serial.println(emgSignal, 4);
  }
}
