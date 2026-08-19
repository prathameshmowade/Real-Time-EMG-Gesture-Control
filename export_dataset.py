"""
======================================================================
  EMG DATASET EXPORTER
  Generates and exports multi-user EMG dataset to CSV, NPZ, and JSON
  formats for custom model training, Kaggle, Colab, or MATLAB.
======================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from train_model_v2 import simulate_emg, extract_features, GESTURES, FEATURE_NAMES, SR, WIN, N_USERS, N_TRIALS, RANDOM

def export_all_datasets(output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(RANDOM)
    print("=" * 60)
    print("   GENERATING & EXPORTING EMG DATASETS")
    print("=" * 60)

    # 1. Generate full dataset (20 subjects, 6 gestures, 60 trials = 7,200 samples)
    n_users = N_USERS
    trials_per_gesture = N_TRIALS
    window_size = WIN
    sampling_rate = SR

    print(f"\n[1/4] Synthesizing physiological sEMG biopotentials...")
    print(f"      Subjects: {n_users} | Gestures: {len(GESTURES)} | Trials: {trials_per_gesture}")
    print(f"      Window Size: {window_size} samples ({window_size/sampling_rate*1000:.1f} ms @ {sampling_rate} Hz)")

    user_scales = np.random.uniform(0.50, 1.60, n_users)
    user_scales[0] = 1.0  # reference subject

    raw_signals = []
    features_list = []
    gesture_labels = []
    user_ids = []

    for u in range(n_users):
        scale = user_scales[u]
        for g_idx, g_name in enumerate(GESTURES):
            for t in range(trials_per_gesture):
                fatigue = (t / trials_per_gesture) * 0.30
                sig = simulate_emg(g_name, user_scale=scale, fatigue=fatigue)
                feat = extract_features(sig)

                raw_signals.append(sig)
                features_list.append(feat)
                gesture_labels.append(g_idx)
                user_ids.append(u)

    raw_signals = np.array(raw_signals, dtype=np.float32)
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(gesture_labels, dtype=np.int64)
    users = np.array(user_ids, dtype=np.int64)
    n_samples = len(labels)

    print(f"      Total Synthesized Samples: {n_samples}")

    # 2. Extract 15 Time-Domain Features
    print(f"\n[2/4] Extracted 15-dimensional time-domain feature vectors across all {n_samples} samples.")

    # 3. Export to CSV Formats
    print(f"\n[3/4] Exporting tabular CSV files...")
    
    # 3a. Features CSV
    df_features = pd.DataFrame(features, columns=FEATURE_NAMES)
    df_features.insert(0, "user_id", users)
    df_features.insert(1, "gesture_id", labels)
    df_features.insert(2, "gesture_name", [GESTURES[l] for l in labels])
    
    feat_csv_path = os.path.join(output_dir, "emg_features_dataset.csv")
    df_features.to_csv(feat_csv_path, index=False)
    print(f"      Saved: {feat_csv_path} ({os.path.getsize(feat_csv_path) / 1024:.1f} KB)")

    # 3b. Raw Waveforms CSV (Signal samples t_0 to t_255)
    sample_cols = [f"sample_{t}" for t in range(window_size)]
    df_raw = pd.DataFrame(raw_signals, columns=sample_cols)
    df_raw.insert(0, "user_id", users)
    df_raw.insert(1, "gesture_id", labels)
    df_raw.insert(2, "gesture_name", [GESTURES[l] for l in labels])
    
    raw_csv_path = os.path.join(output_dir, "emg_raw_signals_dataset.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"      Saved: {raw_csv_path} ({os.path.getsize(raw_csv_path) / (1024*1024):.2f} MB)")

    # 4. Export to Compressed NumPy (.npz) format
    print(f"\n[4/4] Exporting compressed binary NumPy archive (.npz)...")
    npz_path = os.path.join(output_dir, "emg_dataset.npz")
    np.savez_compressed(
        npz_path,
        X_raw=raw_signals,
        X_features=features,
        y=labels,
        users=users,
        gesture_names=np.array(GESTURES),
        feature_names=np.array(FEATURE_NAMES),
        sampling_rate=sampling_rate,
        window_size=window_size
    )
    print(f"      Saved: {npz_path} ({os.path.getsize(npz_path) / (1024*1024):.2f} MB)")

    # 5. Export metadata JSON schema
    schema = {
        "dataset_name": "Multi-Subject Surface EMG Gesture Recognition Dataset",
        "sampling_rate_hz": sampling_rate,
        "window_size_samples": window_size,
        "window_duration_seconds": window_size / sampling_rate,
        "total_samples": n_samples,
        "num_subjects": n_users,
        "subject_ids": list(range(n_users)),
        "gestures": {idx: name for idx, name in enumerate(GESTURES)},
        "feature_names": FEATURE_NAMES,
        "train_split": "Users 0 to 16 (6,120 samples, 85%)",
        "test_split": "Users 17 to 19 (1,080 samples, 15%)",
        "files": {
            "features_csv": "emg_features_dataset.csv",
            "raw_csv": "emg_raw_signals_dataset.csv",
            "numpy_archive": "emg_dataset.npz"
        }
    }
    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"      Saved: {meta_path}")

    print("\n" + "=" * 60)
    print("   DATASET EXPORT COMPLETE!")
    print(f"   Directory: ./{output_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    export_all_datasets()

