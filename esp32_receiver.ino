/*
 ======================================================================
  RECEIVER END: ESP32 DEVKIT V1 (WIRELESS BASE STATION & ACTUATOR)
  Receives real-time EMG biopotential packets & hand gesture commands
  wirelessly from the Raspberry Pi Pico W over high-speed UDP WiFi.
  Controls bionic robotic hand servos and streams to PC / Web Dashboard!
 ======================================================================
 Optional Hardware:
   * 5x SG90 / MG996R Servos for Robotic Hand / Fingers:
     - Servo Thumb  -> GPIO 13
     - Servo Index  -> GPIO 12
     - Servo Middle -> GPIO 14
     - Servo Ring   -> GPIO 27
     - Servo Pinky  -> GPIO 26
   * Status LED     -> GPIO 2 (Onboard Blue LED)
 ======================================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

// --- 1. CONFIGURATION ---
const char* AP_SSID = "ESP32_EMG_GATEWAY";   // Standalone WiFi AP created by ESP32
const char* AP_PASS = "emgpassword123";      // Password (min 8 chars)
const int UDP_PORT = 4210;

WiFiUDP udp;
char packetBuffer[255];

// --- 2. SERVO MOTORS SETUP ---
Servo servoThumb;
Servo servoIndex;
Servo servoMiddle;
Servo servoRing;
Servo servoPinky;

const int PIN_THUMB  = 13;
const int PIN_INDEX  = 12;
const int PIN_MIDDLE = 14;
const int PIN_RING   = 27;
const int PIN_PINKY  = 26;
const int LED_PIN    = 2;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("\n=======================================================");
  Serial.println("   RECEIVER: ESP32 DEVKIT V1 EMG BASE STATION");
  Serial.println("=======================================================");

  // 1. Initialize Servos (50 Hz standard PWM)
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servoThumb.setPeriodHertz(50);
  servoIndex.setPeriodHertz(50);
  servoMiddle.setPeriodHertz(50);
  servoRing.setPeriodHertz(50);
  servoPinky.setPeriodHertz(50);

  servoThumb.attach(PIN_THUMB, 500, 2400);
  servoIndex.attach(PIN_INDEX, 500, 2400);
  servoMiddle.attach(PIN_MIDDLE, 500, 2400);
  servoRing.attach(PIN_RING, 500, 2400);
  servoPinky.attach(PIN_PINKY, 500, 2400);

  // Set initial relaxed position (Fingers Open)
  setRoboticHand(0, 0, 0, 0, 0);

  // 2. Start WiFi in Access Point (AP) mode (No router required!)
  Serial.print("[*] Starting Standalone Access Point: ");
  Serial.println(AP_SSID);
  WiFi.softAP(AP_SSID, AP_PASS);

  IPAddress myIP = WiFi.softAPIP();
  Serial.print("[+] ESP32 Gateway IP: ");
  Serial.println(myIP);

  // 3. Start UDP Listener
  udp.begin(UDP_PORT);
  Serial.printf("[+] Listening for Pico W packets on UDP port: %d\n", UDP_PORT);
  Serial.println("=======================================================\n");
}

void loop() {
  int packetSize = udp.parsePacket();

  if (packetSize) {
    digitalWrite(LED_PIN, HIGH); // Flash LED on packet received
    int len = udp.read(packetBuffer, 254);
    if (len > 0) {
      packetBuffer[len] = '\0';
    }

    String msg = String(packetBuffer);
    msg.trim();

    // Parse CSV format: "GESTURE,RMS,VOLTAGE"
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

    // 1. Actuate Robotic Hand Servos based on Gesture
    actuateServos(gesture);

    // 2. Stream formatted telemetry to PC / Web Dashboard over USB Serial
    Serial.printf("[PICO_W -> ESP32] Gesture: %-12s | RMS: %-6.3f V | Raw: %-6.3f V\n", 
                  gesture.c_str(), rms, voltage);

    digitalWrite(LED_PIN, LOW);
  }
}

// Actuates 5-finger servos based on simplified basic gestures
void actuateServos(String gesture) {
  if (gesture == "FIST") {
    // All fingers tightly closed (180 degrees)
    setRoboticHand(180, 180, 180, 180, 180);
  } 
  else if (gesture == "OPEN_HAND") {
    // All fingers fully extended open (0 degrees)
    setRoboticHand(0, 0, 0, 0, 0);
  } 
  else if (gesture == "DOUBLE_PULSE") {
    // Wave action / Point gesture
    setRoboticHand(180, 0, 180, 180, 180);
    delay(100);
  } 
  else { // RELAX
    // Resting position (25 degrees)
    setRoboticHand(25, 25, 25, 25, 25);
  }
}

void setRoboticHand(int t, int i, int m, int r, int p) {
  servoThumb.write(t);
  servoIndex.write(i);
  servoMiddle.write(m);
  servoRing.write(r);
  servoPinky.write(p);
}
