"""
======================================================================
  HARDWARE-TO-WEB DASHBOARD WEBSOCKET BRIDGE SERVER
  Connects physical hardware (Pico W WiFi UDP or ESP32 USB Serial)
  to the React Web Dashboard (http://localhost:3000) over WebSockets!
======================================================================
Usage:
  # 1. Listen to ESP32 / Arduino on USB Serial and broadcast to Web Dashboard:
  python hardware_dashboard_server.py --source serial --port COM3

  # 2. Listen to Pico W wireless UDP packets and broadcast to Web Dashboard:
  python hardware_dashboard_server.py --source udp --udp_port 4210

  # 3. Auto-detect everything:
  python hardware_dashboard_server.py
======================================================================
"""

import os
import sys
import time
import json
import socket
import asyncio
import argparse
import joblib
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    import websockets
except ImportError:
    websockets = None

from predict_emg import extract_features_standalone

# Global connected WebSocket clients (browser dashboards)
CONNECTED_CLIENTS = set()

# Pre-load trained model for instant feature-to-gesture classification
MODEL = None
GESTURES = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX']

def load_default_model():
    global MODEL, GESTURES
    if os.path.exists("ninapro_model.pkl"):
        MODEL = joblib.load("ninapro_model.pkl")
        GESTURES = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_FLEX', 'WRIST_EXT', 'RADIAL_DEV', 'ULNAR_DEV']
        print("[+] Loaded NinaPro Model for Web Dashboard Bridge.")
    elif os.path.exists("kaggle_emg_model.pkl"):
        MODEL = joblib.load("kaggle_emg_model.pkl")
        GESTURES = ['RELAX', 'FIST', 'WRIST_DOWN', 'WRIST_UP', 'RADIAL_DEV', 'ULNAR_DEV']
        print("[+] Loaded Kaggle Model for Web Dashboard Bridge.")
    elif os.path.exists("emg_model_v2.pkl"):
        loaded = joblib.load("emg_model_v2.pkl")
        MODEL = loaded.get("rf", loaded.get("ensemble", next(iter(loaded.values()))))
        GESTURES = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']
        print("[+] Loaded Random Forest Model for Web Dashboard Bridge.")

async def broadcast_to_dashboards(data_dict):
    """Sends JSON telemetry packet to all connected browser tabs."""
    if not CONNECTED_CLIENTS:
        return
    message = json.dumps(data_dict)
    disconnected = set()
    for client in CONNECTED_CLIENTS:
        try:
            await client.send(message)
        except Exception:
            disconnected.add(client)
    CONNECTED_CLIENTS.difference_update(disconnected)

async def websocket_handler(websocket):
    """Registers new browser dashboard connection."""
    CONNECTED_CLIENTS.add(websocket)
    print(f"[+] Web Dashboard connected from: {websocket.remote_address}")
    try:
        # Send initial handshake
        await websocket.send(json.dumps({
            "type": "handshake",
            "status": "connected",
            "message": "Hardware Bridge Online",
            "gestures": GESTURES
        }))
        async for _ in websocket:
            pass # Keep alive
    except Exception:
        pass
    finally:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"[-] Web Dashboard disconnected.")

async def serial_reader_loop(port, baud):
    """Reads live biopotentials from USB Serial (ESP32 / Arduino / Pico)."""
    if serial is None:
        print("[!] PySerial is required for USB serial reading.")
        return

    while True:
        try:
            ser_ports = [p.device for p in serial.tools.list_ports.comports()]
            use_port = port if port else (ser_ports[0] if ser_ports else None)

            if not use_port:
                print("\r[*] Waiting for ESP32 / Arduino USB connection...", end="", flush=True)
                await asyncio.sleep(1.0)
                continue

            print(f"\n[+] Connecting to USB Serial: {use_port} @ {baud} baud...")
            try:
                ser = serial.Serial(use_port, baud, timeout=0.2, write_timeout=0.2)
                ser.dtr = False
                ser.rts = False
                time.sleep(0.5)
                ser.dtr = True
                ser.rts = True
                print(f"[+] Connected to {use_port}! Streaming to Web Dashboard...")
            except Exception as port_err:
                print(f"\n[!] Could not access {use_port}: {port_err}")
                print("    --> TIP: If Arduino Serial Monitor, Thonny, or another app is open, CLOSE IT first so Python can access the port!")
                await asyncio.sleep(2.5)
                continue

            buffer = []
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        # Handles both pure voltages ("1.650") and tagged packets ("[PICO_W -> ESP32] Gesture: FIST...")
                        if "Gesture:" in line:
                            # Parse ESP32 text log
                            parts = line.split("Gesture:")
                            g_part = parts[1].split("|")[0].strip().replace(">>", "").replace("<<", "").strip()
                            await broadcast_to_dashboards({
                                "type": "gesture_event",
                                "gesture": g_part,
                                "timestamp": time.time()
                            })
                        else:
                            val = float(line.split(",")[0])
                            buffer.append(val)

                            # Stream real-time data point to browser
                            if len(buffer) % 4 == 0:
                                await broadcast_to_dashboards({
                                    "type": "emg_sample",
                                    "voltage": val,
                                    "timestamp": time.time()
                                })

                            if len(buffer) >= 200:
                                win = np.array(buffer[-200:], dtype=np.float32)
                                feats = extract_features_standalone(win)
                                raw_pred = MODEL.predict(feats.reshape(1, -1))[0] if MODEL else 0
                                
                                if isinstance(raw_pred, (str, np.str_)):
                                    pred_label = str(raw_pred)
                                else:
                                    pred_idx = int(raw_pred)
                                    pred_label = GESTURES[pred_idx] if pred_idx < len(GESTURES) else f"Class_{pred_idx}"

                                rms_val = float(np.sqrt(np.mean(win ** 2)))
                                await broadcast_to_dashboards({
                                    "type": "classification",
                                    "gesture": pred_label,
                                    "rms": rms_val,
                                    "features": {name: float(feats[i]) for i, name in enumerate(["MAV","MMAV","RMS","VAR","STD","WL","AAC","DASDV","ZC","SSC","IEMG","HjActivity","HjMobility","HjComplexity","MYOP"])},
                                    "timestamp": time.time()
                                })
                                buffer = buffer[50:]
                    except ValueError:
                        pass

                await asyncio.sleep(0.002) # Yield execution

        except Exception as e:
            print(f"\n[!] Serial Disconnected: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2.0)

async def udp_reader_loop(udp_port):
    """Listens for wireless UDP packets from Raspberry Pi Pico W."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    sock.setblocking(False)
    print(f"[+] UDP Listener active on port {udp_port}. Ready for Pico W wireless packets...")

    loop = asyncio.get_event_loop()
    while True:
        try:
            data, addr = await loop.sock_recv(sock, 1024)
            msg = data.decode('utf-8', errors='ignore').strip()
            
            # Format: "GESTURE,RMS,VOLTAGE"
            parts = msg.split(",")
            gesture = parts[0]
            rms = float(parts[1]) if len(parts) > 1 else 0.0
            voltage = float(parts[2]) if len(parts) > 2 else 0.0

            await broadcast_to_dashboards({
                "type": "classification",
                "gesture": gesture,
                "rms": rms,
                "voltage": voltage,
                "timestamp": time.time()
            })
        except Exception:
            await asyncio.sleep(0.01)

async def main():
    parser = argparse.ArgumentParser(description="Hardware to Web Dashboard WebSocket Bridge")
    parser.add_argument("--source", type=str, default="auto", help="Data source: serial, udp, or auto")
    parser.add_argument("--port", type=str, default=None, help="COM port for Serial (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--udp_port", type=int, default=4210, help="UDP listener port")
    parser.add_argument("--ws_port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    args = parser.parse_args()

    print("=" * 70)
    print("   HARDWARE-TO-WEB DASHBOARD WEBSOCKET BRIDGE SERVER")
    print("=" * 70)
    load_default_model()

    print(f"[*] Starting WebSocket Server on ws://localhost:{args.ws_port}...")
    server = await websockets.serve(websocket_handler, "0.0.0.0", args.ws_port)
    print(f"[+] WebSocket Server running! Open http://localhost:3000 in your browser.")
    print("=" * 70)

    tasks = []
    if args.source in ["serial", "auto"]:
        tasks.append(asyncio.create_task(serial_reader_loop(args.port, args.baud)))
    if args.source in ["udp", "auto"]:
        tasks.append(asyncio.create_task(udp_reader_loop(args.udp_port)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Server stopped by user.")
