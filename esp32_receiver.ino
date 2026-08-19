/*
 ======================================================================
  RECEIVER END: ESP32 DEVKIT V1 (1 LED LIGHT + 2 SERVO MOTORS)
  Receives gesture commands wirelessly from the Pico W transmitter
  over UDP WiFi and controls 1 LED Light and 2 Servo Motors in real-time!
 ======================================================================
 Hardware Wiring for Receiver End:
   * 1x LED Light:
     - Long Leg (+) (Anode)   ->  GPIO 4 (with 220 Ohm Resistor)
     - Short Leg (-) (Cathode)->  GND
     (Or use Onboard Blue LED ->  GPIO 2)

   * 2x Servo Motors (SG90 / MG996R):
     - Servo 1 (Gripper / Hand):
       * Signal Wire (Orange/Yellow) ->  GPIO 13
       * Power (+)   (Red)           ->  5V (VIN or External 5V)
       * Ground (-)  (Brown/Black)   ->  GND
     - Servo 2 (Wrist / Arm Rotation):
       * Signal Wire (Orange/Yellow) ->  GPIO 12
       * Power (+)   (Red)           ->  5V (VIN or External 5V)
       * Ground (-)  (Brown/Black)   ->  GND
 ======================================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

// --- 1. NETWORK CONFIGURATION ---
const char* AP_SSID = "ESP32_EMG_GATEWAY";   // Standalone WiFi Access Point
const char* AP_PASS = "emgpassword123";      // Password
const int UDP_PORT = 4210;

WiFiUDP udp;
char packetBuffer[255];

// --- 2. HARDWARE PIN DEFINITIONS ---
const int PIN_LED     = 4;    // External LED Light (also mirrors to GPIO 2)
const int PIN_LED_ONB = 2;    // Onboard Blue LED
const int PIN_SERVO_1 = 13;   // Servo 1: Gripper / Hand Open-Close
const int PIN_SERVO_2 = 12;   // Servo 2: Wrist Flexion / Arm Angle

Servo servo1_Gripper;
Servo servo2_Wrist;

void setup() {
  Serial.begin(115200);

  // 1. Initialize LED Light Pins
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_LED_ONB, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_LED_ONB, LOW);

  // 2. Initialize 2x Servo Motors
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);

  servo1_Gripper.setPeriodHertz(50); // Standard 50 Hz PWM
  servo2_Wrist.setPeriodHertz(50);

  servo1_Gripper.attach(PIN_SERVO_1, 500, 2400);
  servo2_Wrist.attach(PIN_SERVO_2, 500, 2400);

  // Initial Relaxed State: Gripper Open (0 deg), Wrist Center (90 deg)
  servo1_Gripper.write(0);
  servo2_Wrist.write(90);

  Serial.println("\n=======================================================");
  Serial.println("   RECEIVER: ESP32 (1 LED + 2 SERVO MOTORS) READY");
  Serial.println("=======================================================");

  // 3. Start Standalone Access Point (No Router Required)
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("[+] WiFi AP Started: ");
  Serial.println(AP_SSID);
  Serial.print("[+] ESP32 Gateway IP: ");
  Serial.println(WiFi.softAPIP());

  // 4. Start High-Speed UDP Listener
  udp.begin(UDP_PORT);
  Serial.printf("[+] Listening on UDP port %d for Pico W packets...\n", UDP_PORT);
  Serial.println("=======================================================\n");
}

void loop() {
  int packetSize = udp.parsePacket();

  if (packetSize) {
    int len = udp.read(packetBuffer, 254);
    if (len > 0) {
      packetBuffer[len] = '\0';
    }

    String msg = String(packetBuffer);
    msg.trim();

    // Parse packet: "GESTURE,RMS,VOLTAGE"
    int firstComma = msg.indexOf(',');
    int secondComma = msg.indexOf(',', firstComma + 1);

    String gesture = "RELAX";
    float rms = 0.0f;
    float voltage = 0.0f;

    if (firstComma > 0) {
      gesture = msg.substring(0, firstComma);
      if (secondComma > 0) {
        rms = msg.substring(firstComma + 1, secondComma).toFloat();
        voltage = msg.substring(secondComma + 1).toFloat();
      }
    } else {
      gesture = msg;
    }

    // 1. Actuate 1 LED Light and 2 Servo Motors
    actuateHardware(gesture);

    // 2. Stream Live Telemetry to PC over USB Serial
    Serial.printf("[PICO_W -> ESP32] Gesture: %-12s | RMS: %-5.3f V | LED: %s | S1: %3d deg | S2: %3d deg\n",
                  gesture.c_str(), rms, 
                  (gesture == "FIST" || gesture == "DOUBLE_PULSE") ? "ON " : "OFF",
                  servo1_Gripper.read(), servo2_Wrist.read());
  }
}

// Controls 1 LED Light & 2 Servo Motors based on Hand Gesture
void actuateHardware(String gesture) {
  if (gesture == "FIST") {
    // Squeeze / Power Grip:
    // * LED Light: ON
    // * Servo 1 (Gripper): 180° (Fully Clamped / Closed)
    // * Servo 2 (Wrist):   45°  (Flexed)
    digitalWrite(PIN_LED, HIGH);
    digitalWrite(PIN_LED_ONB, HIGH);
    servo1_Gripper.write(180);
    servo2_Wrist.write(45);
  }
  else if (gesture == "OPEN_HAND") {
    // Stretch Fingers:
    // * LED Light: OFF
    // * Servo 1 (Gripper): 0°   (Fully Open)
    // * Servo 2 (Wrist):   135° (Extended Up)
    digitalWrite(PIN_LED, LOW);
    digitalWrite(PIN_LED_ONB, LOW);
    servo1_Gripper.write(0);
    servo2_Wrist.write(135);
  }
  else if (gesture == "DOUBLE_PULSE") {
    // Quick Double Pulse Action:
    // * LED Light: Double Flash
    // * Servo 1 & 2: Rapid Toggle Wave
    digitalWrite(PIN_LED, HIGH);
    digitalWrite(PIN_LED_ONB, HIGH);
    servo1_Gripper.write(90);
    servo2_Wrist.write(180);
    delay(80);
    digitalWrite(PIN_LED, LOW);
    digitalWrite(PIN_LED_ONB, LOW);
  }
  else { // RELAX
    // Resting State:
    // * LED Light: OFF
    // * Servo 1 (Gripper): 15° (Neutral Relaxed Open)
    // * Servo 2 (Wrist):   90° (Straight Center)
    digitalWrite(PIN_LED, LOW);
    digitalWrite(PIN_LED_ONB, LOW);
    servo1_Gripper.write(15);
    servo2_Wrist.write(90);
  }
}
