"""
======================================================================
  LIVE HARDWARE EMG SERIAL-TO-MODEL INFERENCE BRIDGE
  Connects directly to your microcontroller (Arduino / Pico / ESP32)
  over USB Serial, streams live ADC biopotential voltages, extracts
  features in real-time, and predicts hand gestures using your saved
  pre-trained models (NinaPro, Kaggle, Random Forest, or TCN)!
======================================================================
Usage:
  # 1. Auto-detect COM port and predict using NinaPro Model:
  python live_hardware_bridge.py --model ninapro

  # 2. Specify COM port and Baud rate (e.g. COM3 at 115200 baud):
  python live_hardware_bridge.py --port COM3 --baud 115200 --model kaggle

  # 3. Stream to WebSocket / Robot Arm Servo Output:
  python live_hardware_bridge.py --model ninapro --servo
======================================================================
"""

import os
import sys
import time
import json
import argparse
import joblib
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from predict_emg import extract_features_standalone

def detect_serial_ports():
    if serial is None:
        print("[!] PySerial not installed. Run: pip install pyserial")
        return []
    ports = list(serial.tools.list_ports.comports())
    return [p.device for p in ports]

def run_live_bridge(port=None, baud=115200, model_type="ninapro", window_size=200, step_size=50):
    print("=" * 70)
    print("   LIVE HARDWARE SERIAL-TO-MODEL INFERENCE BRIDGE")
    print("=" * 70)

    # 1. Load Pre-Trained Model
    if model_type.lower() in ["ninapro", "nina"]:
        model_path = "ninapro_model.pkl"
        meta_path = "ninapro_meta.json"
        default_gestures = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_FLEX', 'WRIST_EXT', 'RADIAL_DEV', 'ULNAR_DEV']
    elif model_type.lower() in ["kaggle", "kaggle_rf"]:
        model_path = "kaggle_emg_model.pkl"
        meta_path = "kaggle_emg_meta.json"
        default_gestures = ['RELAX', 'FIST', 'WRIST_DOWN', 'WRIST_UP', 'RADIAL_DEV', 'ULNAR_DEV']
    else:
        model_path = "emg_model_v2.pkl"
        meta_path = "model_meta_v2.json"
        default_gestures = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']

    if not os.path.exists(model_path):
        print(f"[!] Error: Model file '{model_path}' not found!")
        return

    print(f" Loaded Model: {model_path}")
    loaded_obj = joblib.load(model_path)
    if isinstance(loaded_obj, dict):
        model = loaded_obj.get("rf", loaded_obj.get("ensemble", next(iter(loaded_obj.values()))))
    else:
        model = loaded_obj

    gestures = default_gestures
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if "target_gestures" in meta:
                    gestures = meta["target_gestures"]
                elif "gestures" in meta:
                    gestures = meta["gestures"]
        except Exception:
            pass

    print(f" Target Gestures ({len(gestures)}): {', '.join(gestures)}")

    # 2. Resolve Serial Port
    if serial is None:
        print("\n[!] PySerial library is required to connect to hardware.")
        print("    Install via: pip install pyserial\n")
        return

    available_ports = detect_serial_ports()
    if not port:
        if len(available_ports) == 0:
            print("\n[!] No active USB Serial COM ports detected.")
            print("    Please connect your Arduino / Raspberry Pi Pico / ESP32 via USB.\n")
            print("--- RUNNING SIMULATED HARDWARE MODE FOR TESTING ---")
            simulate_hardware_stream(model, gestures, window_size, step_size)
            return
        port = available_ports[0]

    print(f" Connecting to Hardware on: {port} @ {baud} Baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0) # Allow microcontroller boot & auto-reset
        print(f" Connected successfully to {port}!\n")
    except Exception as e:
        print(f"[!] Could not open serial port {port}: {e}")
        print("--- RUNNING SIMULATED HARDWARE MODE FOR TESTING ---")
        simulate_hardware_stream(model, gestures, window_size, step_size)
        return

    # 3. Real-Time Sliding Window Buffer & Inference Loop
    buffer = []
    print("-" * 70)
    print("   STREAMING LIVE HARDWARE BIOPOTENTIAL SIGNALS")
    print("   Perform gestures with your hand/wrist to see instant predictions:")
    print("-" * 70)

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # Parse voltage / ADC value from microcontroller
            try:
                val = float(line.split(",")[0])
                buffer.append(val)
            except ValueError:
                continue

            # When sliding window is full, extract features and predict
            if len(buffer) >= window_size:
                window_data = np.array(buffer[-window_size:], dtype=np.float32)
                t0 = time.perf_counter()

                # Extract 15 time-domain features
                feats = extract_features_standalone(window_data).reshape(1, -1)

                # Model Prediction
                raw_pred = model.predict(feats)[0]
                if isinstance(raw_pred, (str, np.str_)):
                    pred_label = str(raw_pred)
                else:
                    pred_idx = int(raw_pred)
                    pred_label = gestures[pred_idx] if pred_idx < len(gestures) else f"Class_{pred_idx}"

                probs = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
                if probs is not None:
                    if hasattr(model, "classes_") and raw_pred in model.classes_:
                        idx = list(model.classes_).index(raw_pred)
                        conf = probs[idx] * 100.0
                    elif isinstance(raw_pred, (int, np.integer)) and int(raw_pred) < len(probs):
                        conf = probs[int(raw_pred)] * 100.0
                    else:
                        conf = float(np.max(probs)) * 100.0
                else:
                    conf = 100.0
                latency = (time.perf_counter() - t0) * 1000.0

                # Print clean live status bar
                rms_val = np.sqrt(np.mean(window_data ** 2))
                bar_len = int(min(rms_val * 40, 25))
                rms_bar = "#" * bar_len + "-" * (25 - bar_len)
                print(f" EMG Energy: [{rms_bar}] | Detected Gesture: >> {pred_label:12s} << (Conf: {conf:4.1f}%) | Latency: {latency:.2f} ms")

                # Slide the window by step_size
                buffer = buffer[step_size:]

    except KeyboardInterrupt:
        print("\n\n[!] Live inference stopped by user.")
    finally:
        ser.close()
        print(" Serial port closed.")

def simulate_hardware_stream(model, gestures, window_size=200, step_size=50):
    """Simulates a continuous hardware biopotential stream if no USB device is connected."""
    print(" Press Ctrl+C to stop simulation.\n")
    from train_model_v2 import simulate_emg

    sim_gestures = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX']
    g_idx = 0
    buffer = []

    try:
        while True:
            current_gest = sim_gestures[g_idx % len(sim_gestures)]
            # Generate continuous synthetic EMG chunk
            chunk = simulate_emg(current_gest, user_scale=1.0, fatigue=0.0)[:step_size]
            buffer.extend(chunk)

            if len(buffer) >= window_size:
                window_data = np.array(buffer[-window_size:], dtype=np.float32)
                t0 = time.perf_counter()

                feats = extract_features_standalone(window_data).reshape(1, -1)
                raw_pred = model.predict(feats)[0]
                if isinstance(raw_pred, (str, np.str_)):
                    pred_label = str(raw_pred)
                else:
                    pred_idx = int(raw_pred)
                    pred_label = gestures[pred_idx] if pred_idx < len(gestures) else f"Class_{pred_idx}"

                latency = (time.perf_counter() - t0) * 1000.0
                rms_val = np.sqrt(np.mean(window_data ** 2))
                bar_len = int(min(rms_val * 30, 25))
                rms_bar = "#" * bar_len + "-" * (25 - bar_len)
                print(f" [SIM HW] [{rms_bar}] | Gesture: >> {pred_label:12s} << | Latency: {latency:.2f} ms")

                buffer = buffer[step_size:]
                if np.random.rand() > 0.85:
                    g_idx += 1

            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\n Simulation stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Hardware EMG Inference Bridge")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--model", type=str, default="ninapro", help="Model type: ninapro, kaggle, rf")
    args = parser.parse_args()

    run_live_bridge(port=args.port, baud=args.baud, model_type=args.model)
