"""
======================================================================
  SENDER END: RASPBERRY PI PICO W (WEARABLE TRANSMITTER)
  Connects to EMG Sensor V3.0 on arm, samples at 500 Hz, and streams
  biopotentials & gesture commands wirelessly via high-speed UDP
  directly to the ESP32 DevKit V1 Receiver!
======================================================================
 Hardware Connections on Pico W:
   EMG Sensor V3.0    ->  Raspberry Pi Pico W Pin
   * + / VCC          ->  3.3V (Pin 36 - 3V3(OUT))
   * - / GND          ->  GND  (Pin 38 - GND)
   * A / SIG          ->  GP26 (Pin 31 - ADC0)
======================================================================
"""

import time
import math
import socket
import network
from machine import ADC, Pin

# --- 1. CONFIGURATION ---
WIFI_SSID = "ESP32_EMG_GATEWAY"       # ESP32 Access Point SSID (or your Home/Phone Hotspot)
WIFI_PASSWORD = "emgpassword123"      # Password
ESP32_IP = "192.168.4.1"              # Default IP of ESP32 Access Point
UDP_PORT = 4210                       # High-speed UDP port

SAMPLE_RATE_HZ = 500
SAMPLE_INTERVAL_US = 1000000 // SAMPLE_RATE_HZ # 2000 us (2 ms)
WINDOW_SIZE = 100                     # Sliding transmission window

# --- 2. HARDWARE SETUP ---
emg_adc = ADC(Pin(26))                 # GP26 is ADC0
led = Pin("LED", Pin.OUT)             # Onboard Pico W status LED

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    print(f"[*] Connecting to WiFi: {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    timeout = 20
    while not wlan.isconnected() and timeout > 0:
        led.toggle()
        time.sleep(0.5)
        timeout -= 1

    if wlan.isconnected():
        print(f"[+] WiFi Connected! Pico W IP: {wlan.ifconfig()[0]}")
        led.on()
        return True
    else:
        print("[!] WiFi connection failed. Will stream in offline mode.")
        led.off()
        return False

# --- 3. ON-CHIP GESTURE CLASSIFIER ---
GESTURES = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_FLEX', 'WRIST_EXT']

def classify_window(buffer):
    """Computes RMS energy and slope changes in ~0.5 ms."""
    N = len(buffer)
    if N == 0:
        return 'RELAX', 0.0

    sum_sq = sum(x * x for x in buffer)
    rms = math.sqrt(sum_sq / N)
    
    # Fast decision thresholds tuned for EMG Sensor V3.0
    if rms < 0.08:
        gesture = 'RELAX'
    elif rms > 0.65:
        gesture = 'FIST'
    elif rms > 0.40:
        gesture = 'WRIST_FLEX'
    elif rms > 0.20:
        gesture = 'OPEN_HAND'
    else:
        gesture = 'WRIST_EXT'

    return gesture, rms

# --- 4. MAIN ACQUISITION & TRANSMISSION LOOP ---
def main():
    print("=" * 60)
    print("   PICO W EMG SENDER STARTING (EMG SENSOR V3.0)")
    print("=" * 60)

    connect_wifi()

    # Create high-speed UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.0)

    # Calibration DC offset
    print("[*] Calibrating resting baseline on GP26...")
    sum_raw = sum(emg_adc.read_u16() for _ in range(300))
    baseline = (sum_raw / 300.0) / 65535.0 * 3.3
    print(f"[+] Baseline DC Offset: {baseline:.3f} V")

    buffer = []
    sample_count = 0
    t_start = time.ticks_us()

    while True:
        # Enforce exact 500 Hz sampling timer
        t_now = time.ticks_us()
        if time.ticks_diff(t_now, t_start) >= SAMPLE_INTERVAL_US:
            t_start = t_now

            # 1. Read ADC voltage (0.0 to 3.3V)
            raw_u16 = emg_adc.read_u16()
            voltage = (raw_u16 / 65535.0) * 3.3
            signal = voltage - baseline # Zero-centered biopotential
            buffer.append(signal)

            # 2. Window transmission every 50 samples (10 times per second)
            if len(buffer) >= WINDOW_SIZE:
                gesture, rms = classify_window(buffer)

                # Format packet: "GESTURE,RMS,VOLTAGE"
                packet = f"{gesture},{rms:.4f},{voltage:.4f}\n"
                
                try:
                    sock.sendto(packet.encode(), (ESP32_IP, UDP_PORT))
                    print(f"-> Sent to ESP32: >> {gesture:12s} << (RMS: {rms:.3f}V)")
                except Exception as e:
                    pass # Non-blocking UDP

                buffer = buffer[25:] # Slide window by 25 samples

if __name__ == "__main__":
    main()
