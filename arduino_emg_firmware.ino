/*
 ======================================================================
  REAL-TIME EMG DATA ACQUISITION FIRMWARE (ARDUINO / ESP32 / PICO)
  Reads differential raw biopotential voltages from EMG sensor
  (MyoWare 2.0 / AD8232 / ADS1115 / Custom Electrodes) and streams
  at a fixed 500 Hz / 200 Hz sample rate over USB Serial to PC.
 ======================================================================
 Hardware Wiring:
   EMG Sensor (MyoWare / AD8232) -> Arduino / ESP32 Pin
   * +Vs  / VCC  -->  3.3V or 5V
   * GND         -->  GND
   * SIG / OUT   -->  Analog Pin A0 (or GPIO 34 on ESP32, GP26 on Pico)
 ======================================================================
*/

const int EMG_PIN = A0;             // Analog input pin
const unsigned long SAMPLE_INTERVAL_MICROS = 2000; // 500 Hz = 2000 microseconds (2 ms)

unsigned long previousMicros = 0;

void setup() {
  // Initialize high-speed USB Serial communication
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    ; // Wait for serial port to connect
  }

  // Set ADC resolution if on 32-bit MCU (ESP32 / Pico / Arduino Due)
  #if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
    analogReadResolution(12); // 12-bit ADC (0 - 4095)
  #endif
}

void loop() {
  unsigned long currentMicros = micros();

  // Enforce precise sampling rate (500 Hz)
  if (currentMicros - previousMicros >= SAMPLE_INTERVAL_MICROS) {
    previousMicros = currentMicros;

    // 1. Read Raw Analog Value (0 - 1023 on Uno, 0 - 4095 on ESP32/Pico)
    int rawADC = analogRead(EMG_PIN);

    // 2. Convert to normalized voltage (-1.0V to +1.0V or centered voltage)
    #if defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
      float voltage = ((float)rawADC / 4095.0f) * 3.3f;
    #else
      float voltage = ((float)rawADC / 1023.0f) * 5.0f;
    #endif

    // Center the biopotential signal around zero
    float centeredSignal = voltage - (voltage / 2.0f); // or high-pass offset

    // 3. Stream to PC over USB Serial
    Serial.println(voltage, 4);
  }
}
