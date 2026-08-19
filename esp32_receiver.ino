/*
 ======================================================================
  RECEIVER END: ESP32 DEVKIT V1 (1 SERVO MOTOR + 1 LED LIGHT)
  Receives wireless EMG gesture commands from Raspberry Pi Pico W.
  Actuates 1 Servo Motor (0° to 180°) and 1 Indicator LED Light!
 ======================================================================
 Hardware Wiring for Receiver End:
   ESP32 DevKit V1 Pin  ->  Component & Wire Connection
   * GPIO 13            ->  Servo PWM Signal Wire (Orange / Yellow)
   * 5V (or VIN)        ->  Servo VCC (Red Wire) - Or external 5V
   * GND                ->  Servo GND (Brown / Black Wire)
   * GPIO 2 (or GPIO 4) ->  LED Anode (+) via 220 Ohm Resistor
   * GND                ->  LED Cathode (-) Ground
 ======================================================================
 Behavior:
   * RELAX        ->  Servo: 0°   (Open)    |  LED: OFF
   * FIST         ->  Servo: 180° (Closed)  |  LED: ON
   * OPEN_HAND    ->  Servo: 0°   (Open)    |  LED: Slow Blink
   * DOUBLE_PULSE ->  Servo: Sweep (0°-180°) |  LED: Fast Double Flash
 ======================================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

// --- 1. CONFIGURATION ---
const char* AP_SSID = "ESP32_EMG_GATEWAY";   // Standalone WiFi AP name
const char* AP_PASS = "emgpassword123";      // WiFi Password
const int UDP_PORT = 4210;

// --- 2. PIN DEFINITIONS ---
const int SERVO_PIN = 13;   // GPIO 13 for 1 Servo Motor
const int LED_PIN   = 2;    // GPIO 2 (Built-in Blue LED / External LED)

Servo myServo;
WiFiUDP udp;
char packetBuffer[255];

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("\n=======================================================");
  Serial.println("   RECEIVER: ESP32 (1 SERVO MOTOR + 1 LED LIGHT)");
  Serial.println("=======================================================");

  // 1. Initialize Servo (50 Hz standard PWM)
  ESP32PWM::allocateTimer(0);
  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);

  // Initial Relax Position
  myServo.write(0);
  digitalWrite(LED_PIN, LOW);
  Serial.println("[+] Servo attached on GPIO 13 (Position: 0 deg)");
  Serial.println("[+] LED attached on GPIO 2");

  // 2. Start WiFi in Standalone Access Point Mode
  WiFi.softAP(AP_SSID, AP_PASS);
  IPAddress myIP = WiFi.softAPIP();
  Serial.print("[+] ESP32 Gateway IP: ");
  Serial.println(myIP);

  // 3. Start UDP Listener
  udp.begin(UDP_PORT);
  Serial.printf("[+] Listening for Pico W on UDP Port: %d\n", UDP_PORT);
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

    if (firstComma > 0) {
      gesture = msg.substring(0, firstComma);
      if (secondComma > 0) {
        rms = msg.substring(firstComma + 1, secondComma).toFloat();
      }
    } else {
      gesture = msg;
    }

    // 1. Control Servo & LED based on Gesture
    actuateHardware(gesture);

    // 2. Print Live Serial Monitor Telemetry
    Serial.printf("[PICO_W -> ESP32] Gesture: >> %-12s << | RMS: %.3f V\n", gesture.c_str(), rms);
  }
}

// Actuates 1 Servo Motor & 1 LED Light
void actuateHardware(String gesture) {
  if (gesture == "FIST") {
    // Muscle Squeeze -> Close Servo & Turn ON LED
    myServo.write(180);
    digitalWrite(LED_PIN, HIGH);
  } 
  else if (gesture == "OPEN_HAND") {
    // Hand Open -> Open Servo & Turn OFF LED
    myServo.write(0);
    digitalWrite(LED_PIN, LOW);
  } 
  else if (gesture == "DOUBLE_PULSE") {
    // Rapid Double Flex -> Sweep Servo & Flash LED twice
    for (int k = 0; k < 2; k++) {
      digitalWrite(LED_PIN, HIGH);
      delay(80);
      digitalWrite(LED_PIN, LOW);
      delay(80);
    }
    myServo.write(180);
    delay(200);
    myServo.write(0);
  } 
  else { // RELAX
    // Resting -> Open Servo & LED OFF
    myServo.write(0);
    digitalWrite(LED_PIN, LOW);
  }
}
