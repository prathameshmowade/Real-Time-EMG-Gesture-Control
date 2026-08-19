"""
======================================================================
  TRAIN ON REAL-WORLD KAGGLE / UCI EMG DATASET (36 SUBJECTS)
  Loads real sEMG recordings from 36 human subjects, extracts 15 time-domain
  features, runs 5-Fold Stratified Cross-Validation and evaluates on
  completely unseen subjects.
======================================================================
"""

import os
import glob
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from train_model_v2 import extract_features, FEATURE_NAMES

KAGGLE_GESTURES = {
    1: 'RELAX',
    2: 'FIST',
    3: 'WRIST_DOWN',
    4: 'WRIST_UP',
    5: 'RADIAL_DEV',
    6: 'ULNAR_DEV',
    7: 'OPEN_HAND'
}
TARGET_GESTURES = ['RELAX', 'FIST', 'WRIST_DOWN', 'WRIST_UP', 'RADIAL_DEV', 'ULNAR_DEV']

def load_and_preprocess_kaggle_dataset(data_dir="dataset/uci_emg/EMG_data_for_gestures-master", window_size=256, max_subjects=36):
    print("=" * 70)
    print("   LOADING & PARSING REAL-WORLD KAGGLE / UCI EMG DATASET")
    print("=" * 70)
    print(f" Data Directory: {data_dir}")
    print(f" Window Size: {window_size} samples")
    print(f" Target Gesture Classes ({len(TARGET_GESTURES)}): {', '.join(TARGET_GESTURES)}")

    subject_folders = sorted([f for f in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(f)])
    if max_subjects:
        subject_folders = subject_folders[:max_subjects]

    all_features = []
    all_raw = []
    all_labels = []
    all_subjects = []

    print(f"\nProcessing {len(subject_folders)} human subjects...")

    for s_idx, s_folder in enumerate(subject_folders):
        sub_name = os.path.basename(s_folder)
        txt_files = glob.glob(os.path.join(s_folder, "*.txt"))
        sub_windows = 0

        for txt_file in txt_files:
            try:
                # Read tab-delimited text file
                df = pd.read_csv(txt_file, sep='\t')
                if 'class' not in df.columns or df.shape[1] < 10:
                    continue

                # Filter out unmarked class (0)
                df = df[df['class'].isin(KAGGLE_GESTURES.keys())]

                # Group by continuous gesture segments
                # Use Channel 1 or mean of channels for single-channel pipeline
                # (Channel 1: Flexor Carpi Radialis)
                sig_col = 'channel1' if 'channel1' in df.columns else df.columns[1]

                # Process segment per gesture class
                for g_id, g_df in df.groupby('class'):
                    g_name = KAGGLE_GESTURES.get(g_id, None)
                    if g_name not in TARGET_GESTURES:
                        continue
                    
                    target_label_idx = TARGET_GESTURES.index(g_name)
                    raw_signal = g_df[sig_col].values

                    # Sliding non-overlapping windowing
                    n_windows = len(raw_signal) // window_size
                    for w in range(n_windows):
                        win = raw_signal[w * window_size : (w + 1) * window_size]
                        
                        # Scale biopotentials to realistic volts if recorded in microvolts
                        if np.max(np.abs(win)) < 0.001:
                            win = win * 1000.0  # convert mV / V scaling
                        
                        feat = extract_features(win)
                        all_raw.append(win)
                        all_features.append(feat)
                        all_labels.append(target_label_idx)
                        all_subjects.append(s_idx)
                        sub_windows += 1

            except Exception as e:
                print(f"  [Warning] Error parsing {txt_file}: {e}")

        print(f"  * Subject {s_idx+1:02d} ({sub_name}): {sub_windows} gesture windows extracted.")

    X_feat = np.array(all_features, dtype=np.float32)
    X_raw = np.array(all_raw, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)

    print(f"\nTotal Dataset Samples: {len(y)}")
    print(f"Feature Matrix Shape: {X_feat.shape}")
    print(f"Raw Signal Matrix Shape: {X_raw.shape}")
    print(f"Class Distribution: {np.bincount(y)}")

    return X_feat, X_raw, y, subjects

def train_and_benchmark_kaggle(X_feat, y, subjects, output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("   BENCHMARKING CLASSIFIERS ON REAL KAGGLE EMG DATA")
    print("=" * 70)

    # Subject-independent split: First ~80% subjects for training, remaining 20% for testing
    unique_subjects = np.unique(subjects)
    n_train_sub = int(len(unique_subjects) * 0.80)
    train_subs = unique_subjects[:n_train_sub]
    test_subs = unique_subjects[n_train_sub:]

    train_mask = np.isin(subjects, train_subs)
    test_mask = np.isin(subjects, test_subs)

    X_train, y_train = X_feat[train_mask], y[train_mask]
    X_test, y_test = X_feat[test_mask], y[test_mask]

    print(f"Training Cohort: {len(train_subs)} subjects ({len(y_train)} samples)")
    print(f"Unseen Test Cohort: {len(test_subs)} subjects ({len(y_test)} samples)")

    # 1. 5-Fold Cross-Validation on Training Cohort
    print("\n--- Running 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest (200 Trees)": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "SVM (RBF Kernel)": Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=10, gamma='scale', probability=True, random_state=42))]),
        "Gaussian Naive Bayes": Pipeline([('scaler', StandardScaler()), ('gnb', GaussianNB())]),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        print(f"  {name:30s} | 5-Fold CV Accuracy: {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")

    # 2. Train Best Random Forest Model on Full Training Cohort
    print("\n--- Training Final Random Forest Model on Full Training Cohort ---")
    rf_best = RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_best.fit(X_train, y_train)

    # 3. Evaluate on Unseen Test Subjects
    y_pred = rf_best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> UNSEEN SUBJECT TEST ACCURACY: {test_acc*100:.2f}% <<<\n")

    report_dict = classification_report(y_test, y_pred, target_names=TARGET_GESTURES, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=TARGET_GESTURES)
    print("Classification Report (Unseen Test Subjects):")
    print(report_text)

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_GESTURES, yticklabels=TARGET_GESTURES, ax=ax)
    ax.set_title(f"Real-World Kaggle/UCI EMG Test Confusion Matrix (Acc: {test_acc*100:.2f}%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Gesture", fontweight='bold')
    ax.set_ylabel("True Gesture", fontweight='bold')
    plt.tight_layout()
    plot_path = "kaggle_emg_results.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved evaluation plot: {plot_path}")

    # 5. Save Trained Model & Metadata
    model_save_path = "kaggle_emg_model.pkl"
    joblib.dump(rf_best, model_save_path)
    print(f"Saved trained Kaggle model: {model_save_path}")

    meta = {
        "dataset_source": "Kaggle / UCI Machine Learning Repository - EMG Data for Gestures (36 Subjects)",
        "total_samples": len(y),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_subjects": [int(s) for s in train_subs],
        "test_subjects": [int(s) for s in test_subs],
        "gestures": TARGET_GESTURES,
        "cv_results": cv_results,
        "test_accuracy": float(test_acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    meta_path = "kaggle_emg_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    print("\n" + "=" * 70)
    print("   TRAINING ON REAL-WORLD KAGGLE DATASET COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    X_feat, X_raw, y, subjects = load_and_preprocess_kaggle_dataset()
    train_and_benchmark_kaggle(X_feat, y, subjects)
