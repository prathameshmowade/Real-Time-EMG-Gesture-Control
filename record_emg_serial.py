"""
======================================================================
  REAL-TIME HARDWARE EMG DATASET RECORDER
  Connects to Arduino / Raspberry Pi Pico / ESP32 / MyoWare ADC via USB
  Serial and guides you through recording a labeled EMG dataset.
======================================================================
Usage:
  python record_emg_serial.py --port COM3 --baud 115200 --subject 1 --trials 10
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

GESTURES = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX']
WINDOW_SIZE = 256
SAMPLING_RATE = 500  # 500 Hz

def record_live_dataset(port="COM3", baud=115200, subject_id=1, trials_per_gesture=10, output_dir="my_emg_recordings"):
    try:
        import serial
    except ImportError:
        print("ERROR: 'pyserial' library not found.")
        print("Please install via: pip install pyserial")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("   REAL-TIME EMG HARDWARE DATASET ACQUISITION PROTOCOL")
    print("=" * 65)
    print(f" Port: {port} | Baud Rate: {baud} | Subject ID: {subject_id}")
    print(f" Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE*1000:.0f} ms @ {SAMPLING_RATE} Hz)")
    print(f" Gestures: {', '.join(GESTURES)}")
    print(f" Repetitions per gesture: {trials_per_gesture}")
    print("=" * 65)

    try:
        ser = serial.Serial(port, baud, timeout=2.0)
        time.sleep(2.0)  # wait for serial connection to stabilize
        print(f" Connected to {port} successfully!\n")
    except Exception as e:
        print(f" Could not open serial port {port}: {e}")
        print("Tip: Check Device Manager for your COM port or run simulated mode.")
        return

    all_windows = []
    all_labels = []
    all_gestures = []
    all_trials = []

    try:
        for g_idx, g_name in enumerate(GESTURES):
            print(f"\n─────────────────────────────────────────────────────────")
            print(f" [GESTURE {g_idx+1}/{len(GESTURES)}]: >> {g_name} <<")
            print(f"─────────────────────────────────────────────────────────")
            input(f"Get ready to perform '{g_name}'. Press [ENTER] to start recording...")

            for t in range(trials_per_gesture):
                print(f"  --> Trial {t+1}/{trials_per_gesture} for '{g_name}': HOLD GESTURE NOW!")
                
                # Buffer to collect WINDOW_SIZE samples
                buffer = []
                ser.reset_input_buffer()
                start_time = time.time()

                while len(buffer) < WINDOW_SIZE:
                    if time.time() - start_time > 5.0:
                        print("      [Warning] Timeout reading from serial. Padding remaining.")
                        while len(buffer) < WINDOW_SIZE:
                            buffer.append(0.0)
                        break

                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            # Parse voltage / analog value
                            val = float(line)
                            # If raw 12-bit or 10-bit ADC, convert to voltage
                            if val > 1023.0:
                                val = (val / 4095.0) * 3.3  # 12-bit ESP32 / Pico ADC
                            elif val > 10.0:
                                val = (val / 1023.0) * 5.0  # 10-bit Arduino Uno ADC
                            buffer.append(val)
                    except ValueError:
                        continue

                all_windows.append(buffer[:WINDOW_SIZE])
                all_labels.append(g_idx)
                all_gestures.append(g_name)
                all_trials.append(t + 1)
                print(f"      Recorded {WINDOW_SIZE} samples (RMS = {np.sqrt(np.mean(np.square(buffer))):.4f} V)")

                # Rest period between repetitions
                if t < trials_per_gesture - 1:
                    print("      [REST for 1.5 seconds]...", end="", flush=True)
                    time.sleep(1.5)
                    print(" Done.")

    except KeyboardInterrupt:
        print("\n Acquisition interrupted by user. Saving captured data...")
    finally:
        ser.close()

    # Save to CSV
    if len(all_windows) > 0:
        sample_cols = [f"sample_{i}" for i in range(WINDOW_SIZE)]
        df = pd.DataFrame(all_windows, columns=sample_cols)
        df.insert(0, "subject_id", subject_id)
        df.insert(1, "gesture_id", all_labels)
        df.insert(2, "gesture_name", all_gestures)
        df.insert(3, "trial_num", all_trials)

        csv_file = os.path.join(output_dir, f"emg_subject_{subject_id}_{int(time.time())}.csv")
        df.to_csv(csv_file, index=False)
        print("\n" + "=" * 65)
        print(f" SUCCESS: Recorded {len(df)} total gesture windows!")
        print(f" Saved dataset to: {csv_file}")
        print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Hardware EMG Dataset Recorder")
    parser.add_argument("--port", type=str, default="COM3", help="Serial COM port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID integer (default: 1)")
    parser.add_argument("--trials", type=int, default=10, help="Trials per gesture (default: 10)")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")

    args = parser.parse_args()
    record_live_dataset(
        port=args.port,
        baud=args.baud,
        subject_id=args.subject,
        trials_per_gesture=args.trials,
        output_dir=args.output
    )
