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

# --- 3. ROBUST BASIC MOVEMENT CLASSIFIER (EMG SENSOR V3.0) ---
# Simplified to 3 core unmistakable muscle states + double-pulse trigger
GESTURES = ['RELAX', 'FIST', 'OPEN_HAND', 'DOUBLE_PULSE']

# State tracking for double-pulse detection
last_peak_time = 0
pulse_count = 0

def classify_basic_movement(buffer, baseline_noise):
    """
    Robust Envelope & Energy Classifier tailored for single-channel EMG V3.0.
    Eliminates complex multi-muscle crosstalk and focuses on clean, distinct gestures.
    """
    global last_peak_time, pulse_count
    N = len(buffer)
    if N == 0:
        return 'RELAX', 0.0

    # Calculate Root Mean Square (RMS) & Rectified Mean (MAV)
    mav = sum(abs(x) for x in buffer) / N
    rms = math.sqrt(sum(x * x for x in buffer) / N)

    # Dynamic noise floor threshold (auto-adjusts to resting baseline)
    noise_gate = max(baseline_noise * 1.5, 0.05)
    
    # 1. Check if resting
    if rms < noise_gate:
        return 'RELAX', rms

    # 2. Check for Double Pulse (quick contraction twice within 700 ms)
    now = time.ticks_ms()
    if rms > 0.35:
        if time.ticks_diff(now, last_peak_time) > 200 and time.ticks_diff(now, last_peak_time) < 700:
            last_peak_time = now
            return 'DOUBLE_PULSE', rms
        else:
            last_peak_time = now

    # 3. High Force vs Moderate Contraction
    if rms >= 0.45 or mav >= 0.35:
        return 'FIST', rms          # Strong muscle squeeze / clench
    elif rms >= 0.12 or mav >= 0.09:
        return 'OPEN_HAND', rms     # Moderate finger stretch / hand open
    else:
        return 'RELAX', rms

# --- 4. MAIN ACQUISITION & TRANSMISSION LOOP ---
def main():
    print("=" * 60)
    print("   PICO W EMG SENDER STARTING (EMG SENSOR V3.0)")
    print("=" * 60)

    connect_wifi()

    # Create high-speed UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.0)

    # Calibration DC offset & resting noise level
    print("[*] Calibrating resting baseline on GP26 (Keep Arm Relaxed)...")
    samples_cal = []
    for _ in range(300):
        v = (emg_adc.read_u16() / 65535.0) * 3.3
        samples_cal.append(v)
        time.sleep_ms(2)

    baseline = sum(samples_cal) / len(samples_cal)
    # Estimate baseline resting RMS noise
    baseline_noise = math.sqrt(sum((x - baseline)**2 for x in samples_cal) / len(samples_cal))
    print(f"[+] Baseline DC Offset: {baseline:.3f} V | Noise Floor: {baseline_noise:.3f} V")

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
                gesture, rms = classify_basic_movement(buffer, baseline_noise)

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
