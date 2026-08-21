"""
======================================================================
  HARDWARE CONNECTION DIAGNOSTIC & DEBUGGER TOOL
  Tests direct serial communication with ESP32 / Pico W / Arduino
  and identifies exact connection bottlenecks.
======================================================================
"""

import sys
import time
import serial
import serial.tools.list_ports

def scan_and_test():
    print("=" * 70)
    print("   [+] EMG HARDWARE DIAGNOSTIC & PORT TESTER")
    print("=" * 70)

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("\n[!] ERROR: No USB Serial COM ports found!")
        print("    -> Make sure your ESP32 / Arduino USB cable is plugged in firmly.")
        print("    -> Ensure you are using a DATA USB cable (not charge-only).")
        return

    print(f"\n[+] Detected {len(ports)} Serial Port(s):")
    for i, p in enumerate(ports):
        print(f"    [{i+1}] Port: {p.device} | Description: {p.description} | HWID: {p.hwid}")

    # Test each port
    for p in ports:
        port_name = p.device
        print(f"\n--- Testing Port: {port_name} ({p.description}) ---")
        try:
            ser = serial.Serial(port_name, 115200, timeout=1.0)
            print(f"[OK] Port {port_name} OPENED SUCCESSFULLY!")
            print(f"[*] Listening for incoming data for 4 seconds (115200 baud)...")
            
            lines_received = 0
            t_end = time.time() + 4.0
            while time.time() < t_end:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    lines_received += 1
                    print(f"    -> [RAW DATA]: {line}")

            if lines_received > 0:
                print(f"[OK] SUCCESS! Received {lines_received} data lines from {port_name}.")
            else:
                print(f"[!] Warning: Port opened, but NO data received.")
                print("    -> Check if firmware is uploaded and running on the board.")
                print("    -> Press the 'EN / RESET' button on your ESP32.")

            ser.close()
        except serial.SerialException as e:
            print(f"[X] FAILED to open {port_name}: {e}")
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                print("    -> CAUSE: Port is BUSY / LOCKED by another software!")
                print("    -> ACTION: Close Arduino IDE Serial Monitor, Thonny, or any other serial terminal.")
        except Exception as e:
            print(f"[X] Unexpected Error on {port_name}: {e}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    scan_and_test()
