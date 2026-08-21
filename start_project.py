"""
======================================================================
  ONE-CLICK PROJECT LAUNCHER: EMG GESTURE RECOGNITION SYSTEM
  1. Detects connected ESP32 / Arduino hardware on COM ports
  2. Starts the WebSocket Bridge Server
  3. Opens the interactive Web Dashboard in your browser
======================================================================
"""

import os
import sys
import time
import webbrowser
import subprocess

try:
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
except Exception:
    ports = []

print("=" * 70)
print("   🚀 LAUNCHING REAL-TIME EMG GESTURE RECOGNITION SYSTEM")
print("=" * 70)

print("\n[1/3] Checking connected Hardware...")
if ports:
    for p in ports:
        print(f"  * Detected Device: {p.device} -> {p.description}")
else:
    print("  * No USB COM ports detected. (Running in Wireless UDP / Simulated mode)")

print("\n[2/3] Opening Web Dashboard in browser...")
dashboard_url = "http://localhost:3000"
webbrowser.open(dashboard_url)
print(f"  * Dashboard URL: {dashboard_url}")

print("\n[3/3] Starting Hardware WebSocket Bridge Server on ws://localhost:8765...")
print("----------------------------------------------------------------------")
print(" IMPORTANT: If Arduino IDE Serial Monitor is open, CLOSE IT now")
print("            so Python can communicate with your ESP32 on COM3.")
print("----------------------------------------------------------------------\n")

# Run hardware dashboard server
import hardware_dashboard_server
import asyncio

try:
    asyncio.run(hardware_dashboard_server.main())
except KeyboardInterrupt:
    print("\n[!] Project stopped.")
