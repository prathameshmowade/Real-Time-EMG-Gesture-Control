"""
======================================================================
  NINAPRO EMG BENCHMARK DATASET DOWNLOADER & TRAINER
  Automates downloading preprocessed subjects from the NinaPro Database
  (https://ninapro.hevs.ch/), parses multi-channel .mat files, extracts
  15 time-domain features, and trains gesture recognition models.
======================================================================
"""

import os
import sys
import glob
import json
import time
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import scipy.io
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from train_model_v2 import extract_features, FEATURE_NAMES

# Mapping of core NinaPro DB1 gestures (Exercise B: 17 basic hand/wrist gestures)
NINAPRO_GESTURE_MAP = {
    0: 'RELAX',
    1: 'THUMB_FLEX',
    2: 'THUMB_EXT',
    3: 'INDEX_FLEX',
    4: 'INDEX_EXT',
    5: 'MIDDLE_FLEX',
    6: 'MIDDLE_EXT',
    7: 'RING_LITTLE_FLEX',
    8: 'RING_LITTLE_EXT',
    9: 'FIST',
    10: 'OPEN_HAND',
    11: 'WRIST_FLEX',
    12: 'WRIST_EXT',
    13: 'RADIAL_DEV',
    14: 'ULNAR_DEV',
    15: 'PRONATION',
    16: 'SUPINATION',
    17: 'HAND_OPEN'
}

def download_ninapro_subject(subject_id=1, db="DB1", dest_dir="dataset/ninapro"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_filename = f"s{subject_id}.zip"
    zip_path = os.path.join(dest_dir, zip_filename)
    extract_folder = os.path.join(dest_dir, f"s{subject_id}")

    if os.path.exists(extract_folder) and len(glob.glob(os.path.join(extract_folder, "*.mat"))) > 0:
        print(f" Subject {subject_id} already downloaded and extracted at: {extract_folder}")
        return extract_folder

    if db.upper() == "DB1":
        url = f"https://ninapro.hevs.ch/files/DB1/Preprocessed/s{subject_id}.zip"
    else:
        url = f"https://ninapro.hevs.ch/files/DB5_Preproc/s{subject_id}.zip"

    print("=" * 70)
    print(f"   DOWNLOADING NINAPRO {db.upper()} - SUBJECT {subject_id}")
    print("=" * 70)
    print(f" URL: {url}")
    print(f" Destination: {zip_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, "wb") as out_file:
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            print(f"\r Downloaded: {downloaded / (1024*1024):.2f} MB...", end="", flush=True)

    print(f"\n Download complete! Total Size: {os.path.getsize(zip_path)/(1024*1024):.2f} MB")
    print(f" Extracting to: {extract_folder}...")
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(" Extraction complete!")
    return extract_folder

def load_ninapro_mat_files(subject_folders, window_size=200, step_size=100, target_gestures=None):
    print("\n" + "=" * 70)
    print("   PARSING & EXTRACTING FEATURES FROM NINAPRO (.MAT)")
    print("=" * 70)

    if target_gestures is None:
        target_gestures = [0, 9, 10, 11, 12, 13, 14] # RELAX, FIST, OPEN_HAND, WRIST_FLEX, WRIST_EXT, RADIAL, ULNAR

    target_names = [NINAPRO_GESTURE_MAP.get(g, f"G_{g}") for g in target_gestures]
    print(f" Target Gestures ({len(target_gestures)}): {', '.join(target_names)}")
    print(f" Window Size: {window_size} samples | Step: {step_size} samples")

    all_features = []
    all_labels = []
    all_subjects = []

    for s_idx, s_dir in enumerate(subject_folders):
        mat_files = glob.glob(os.path.join(s_dir, "*.mat")) + glob.glob(os.path.join(s_dir, "**", "*.mat"), recursive=True)
        sub_windows = 0

        for mat_file in mat_files:
            try:
                data = scipy.io.loadmat(mat_file)
                if 'emg' not in data:
                    continue

                emg = data['emg'] # shape (T, N_channels), e.g. (T, 10)
                # 'restimulus' contains refined movement ground truth labels
                stimulus = data.get('restimulus', data.get('stimulus', None))
                if stimulus is None:
                    continue
                stimulus = stimulus.flatten()

                # Choose primary electrode channel (channel 1) or mean of channels
                signal = emg[:, 0]

                # Slide windows across continuous acquisition
                for start in range(0, len(signal) - window_size, step_size):
                    end = start + window_size
                    win_stim = stimulus[start:end]

                    # Only take window if label is consistent throughout the window
                    if np.all(win_stim == win_stim[0]):
                        label = int(win_stim[0])
                        if label in target_gestures:
                            target_label_idx = target_gestures.index(label)
                            win = signal[start:end]

                            feat = extract_features(win)
                            all_features.append(feat)
                            all_labels.append(target_label_idx)
                            all_subjects.append(s_idx)
                            sub_windows += 1

            except Exception as e:
                print(f"  [Warning] Error loading {mat_file}: {e}")

        print(f"  * Subject {s_idx+1}: {sub_windows} valid gesture windows extracted.")

    X_feat = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)

    print(f"\nTotal Extracted Samples: {len(y)}")
    print(f"Feature Matrix Shape: {X_feat.shape}")
    print(f"Class Distribution: {np.bincount(y) if len(y)>0 else []}")

    return X_feat, y, subjects, target_names

def train_and_evaluate_ninapro(X_feat, y, target_names, output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("   TRAINING & BENCHMARKING MODELS ON NINAPRO EMG DATA")
    print("=" * 70)

    if len(y) < 50:
        print("Not enough samples to train.")
        return

    # 1. 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=10, gamma='scale', random_state=42))])

    print("Running 5-Fold Stratified Cross-Validation...")
    rf_scores = cross_val_score(rf, X_feat, y, cv=cv, scoring='accuracy', n_jobs=-1)
    svm_scores = cross_val_score(svm, X_feat, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"  * Random Forest (200 Trees): {np.mean(rf_scores)*100:.2f}% ± {np.std(rf_scores)*100:.2f}%")
    print(f"  * SVM (RBF Kernel):          {np.mean(svm_scores)*100:.2f}% ± {np.std(svm_scores)*100:.2f}%")

    # 2. Train final model on full dataset
    print("\nTraining final Random Forest model...")
    rf.fit(X_feat, y)

    y_pred = rf.predict(X_feat)
    report_dict = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    report_text = classification_report(y, y_pred, target_names=target_names)
    print("\nClassification Report (Full NinaPro Cohort):")
    print(report_text)

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(f"NinaPro DB1 EMG Confusion Matrix (5-Fold CV: {np.mean(rf_scores)*100:.2f}%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Gesture", fontweight='bold')
    ax.set_ylabel("True Gesture", fontweight='bold')
    plt.tight_layout()
    plot_path = "ninapro_results.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved evaluation plot: {plot_path}")

    # 4. Save Model & Metadata
    model_path = "ninapro_model.pkl"
    joblib.dump(rf, model_path)
    print(f"Saved trained NinaPro model: {model_path}")

    meta = {
        "dataset_name": "NinaPro Database (DB1 / DB5)",
        "total_samples": len(y),
        "target_gestures": target_names,
        "cv_accuracy_rf": {"mean": float(np.mean(rf_scores)), "std": float(np.std(rf_scores))},
        "cv_accuracy_svm": {"mean": float(np.mean(svm_scores)), "std": float(np.std(svm_scores))},
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
    meta_path = "ninapro_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    print("\n" + "=" * 70)
    print("   NINAPRO TRAINING & BENCHMARK COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NinaPro Dataset Downloader & Trainer")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2], help="Subject IDs to download (e.g. 1 2 3)")
    parser.add_argument("--db", type=str, default="DB1", help="NinaPro database version: DB1 or DB5")
    args = parser.parse_args()

    subject_folders = []
    for sid in args.subjects:
        s_folder = download_ninapro_subject(subject_id=sid, db=args.db)
        subject_folders.append(s_folder)

    X_feat, y, subjects, target_names = load_ninapro_mat_files(subject_folders)
    train_and_evaluate_ninapro(X_feat, y, target_names)
