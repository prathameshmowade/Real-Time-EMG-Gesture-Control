"""
======================================================================
  UNIVERSAL EMG GESTURE PREDICTOR (ZERO-TRAINING INFERENCE)
  Directly loads any pre-trained model file (.pkl or .pth) and runs
  instant predictions on new EMG signals or live streams on ANY PC.
======================================================================
Usage:
  # 1. Predict using the 36-Subject Kaggle Pre-Trained Model:
  python predict_emg.py --model kaggle

  # 2. Predict using the Tuned Random Forest / Ensemble:
  python predict_emg.py --model rf

  # 3. Predict using Deep Learning TCN / BiLSTM / DTSF-CNN:
  python predict_emg.py --model tcn
  python predict_emg.py --model bilstm
  python predict_emg.py --model cnn

  # 4. Predict on a custom CSV file:
  python predict_emg.py --model kaggle --input path/to/my_signal.csv
======================================================================
"""

import os
import sys
import json
import time
import argparse
import joblib
import numpy as np

# Feature extraction function (Standalone, zero dependencies outside standard numpy)
def extract_features_standalone(signal, sampling_rate=500):
    sig = np.array(signal, dtype=np.float64)
    N = len(sig)
    
    # 1-5. Amplitude & Energy
    mav = np.mean(np.abs(sig))
    w = np.ones(N)
    w[:N//4] = 0.5
    w[3*N//4:] = 0.5
    mmav = np.mean(w * np.abs(sig))
    rms = np.sqrt(np.mean(sig ** 2))
    var = np.var(sig)
    std = np.std(sig)
    
    # 6-8. Waveform Morphology
    diff = np.diff(sig)
    wl = np.sum(np.abs(diff))
    aac = np.mean(np.abs(diff))
    dasdv = np.sqrt(np.mean(diff ** 2))
    
    # 9-10. Frequency Surrogates
    zc = np.sum(((sig[:-1] * sig[1:]) < 0) & (np.abs(diff) > 0.01))
    d1, d2 = diff[:-1], diff[1:]
    ssc = np.sum((np.abs(d1 - d2) >= 0.003) & (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))))
    
    # 11. Integrated EMG
    iemg = np.sum(np.abs(sig))
    
    # 12-14. Hjorth Parameters
    h_act = var
    h_mob = np.sqrt(np.var(diff) / (var + 1e-12))
    diff2 = np.diff(diff)
    h_mob_diff = np.sqrt(np.var(diff2) / (np.var(diff) + 1e-12))
    h_comp = h_mob_diff / (h_mob + 1e-12)
    
    # 15. Myopulse Rate
    myop = np.mean(np.abs(sig) > 3.0 * (std + 1e-12))
    
    return np.array([
        mav, mmav, rms, var, std,
        wl, aac, dasdv, zc, ssc,
        iemg, h_act, h_mob, h_comp, myop
    ], dtype=np.float32)

def run_prediction(model_type="kaggle", input_file=None):
    print("=" * 70)
    print("   EMG REAL-TIME PREDICTION & INFERENCE ENGINE")
    print("=" * 70)

    # 1. Select Model
    if model_type.lower() in ["ninapro", "nina"]:
        model_path = "ninapro_model.pkl"
        meta_path = "ninapro_meta.json"
        is_torch = False
        default_gestures = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_FLEX', 'WRIST_EXT', 'RADIAL_DEV', 'ULNAR_DEV']
    elif model_type.lower() in ["kaggle", "kaggle_rf"]:
        model_path = "kaggle_emg_model.pkl"
        meta_path = "kaggle_emg_meta.json"
        is_torch = False
        default_gestures = ['RELAX', 'FIST', 'WRIST_DOWN', 'WRIST_UP', 'RADIAL_DEV', 'ULNAR_DEV']
    elif model_type.lower() in ["rf", "ensemble", "v2"]:
        model_path = "emg_model_v2.pkl"
        meta_path = "model_meta_v2.json"
        is_torch = False
        default_gestures = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']
    elif model_type.lower() == "tcn":
        model_path = "tcn_model.pth"
        meta_path = "tcn_meta.json"
        is_torch = True
        default_gestures = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']
    elif model_type.lower() == "bilstm":
        model_path = "bilstm_model.pth"
        meta_path = "bilstm_meta.json"
        is_torch = True
        default_gestures = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']
    elif model_type.lower() == "cnn":
        model_path = "cnn_model.pth"
        meta_path = "cnn_meta.json"
        is_torch = True
        default_gestures = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']
    else:
        print(f"Unknown model type: {model_type}. Options: kaggle, rf, tcn, bilstm, cnn")
        return

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    print(f" Loaded Pre-Trained Model: {model_path}")

    # Load Metadata if available
    gestures = default_gestures
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if "gestures" in meta and isinstance(meta["gestures"], list):
                    gestures = meta["gestures"]
        except Exception:
            pass

    print(f" Active Gesture Classes ({len(gestures)}): {', '.join(gestures)}")

    # 2. Load Model Object
    if not is_torch:
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict):
            # Prefer ensemble or rf if available in dictionary
            if "rf" in loaded_obj:
                model = loaded_obj["rf"]
            elif "ensemble" in loaded_obj:
                model = loaded_obj["ensemble"]
            else:
                model = next(iter(loaded_obj.values()))
        else:
            model = loaded_obj
    else:
        import torch
        if model_type.lower() == "tcn":
            from train_tcn_fast import TemporalConvNet
            model = TemporalConvNet(n_classes=len(gestures))
        elif model_type.lower() == "bilstm":
            from train_bilstm_model import CNN_BiLSTM
            model = CNN_BiLSTM(n_classes=len(gestures))
        elif model_type.lower() == "cnn":
            from train_cnn_model import DTSF_CNN, compute_welch_psd
            model = DTSF_CNN(n_classes=len(gestures))
        
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        model.eval()

    # 3. Obtain Input Data
    if input_file and os.path.exists(input_file):
        import pandas as pd
        df = pd.read_csv(input_file)
        # Check if raw signal or features
        if "sample_0" in df.columns:
            sample_cols = [c for c in df.columns if c.startswith("sample_")]
            signals = df[sample_cols].values
        elif df.shape[1] >= 256:
            signals = df.iloc[:, :256].values
        else:
            signals = df.values
    else:
        # Generate simulated demo windows for each class to demonstrate instant prediction
        print("\n Generating demonstration test signals...")
        from train_model_v2 import simulate_emg
        demo_gestures = ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX'] if 'OPEN_HAND' in gestures else gestures
        signals = [simulate_emg(g) if g in ['RELAX', 'FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX'] else np.random.randn(256)*0.5 for g in demo_gestures]

    # 4. Run Instant Inference
    print("\n" + "-" * 70)
    print("   RUNNING INSTANT PREDICTIONS")
    print("-" * 70)

    for i, sig in enumerate(signals):
        t0 = time.perf_counter()
        
        if not is_torch:
            features = extract_features_standalone(sig).reshape(1, -1)
            raw_pred = model.predict(features)[0]
            if isinstance(raw_pred, (str, np.str_)):
                pred_label = str(raw_pred)
                pred_idx = gestures.index(pred_label) if pred_label in gestures else 0
            else:
                pred_idx = int(raw_pred)
                pred_label = gestures[pred_idx] if pred_idx < len(gestures) else f"Class_{pred_idx}"
            
            probs = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
            confidence = probs[pred_idx] * 100.0 if probs is not None and pred_idx < len(probs) else 100.0
        else:
            import torch
            sig_arr = np.array(sig, dtype=np.float32)
            sig_norm = (sig_arr - sig_arr.mean()) / (sig_arr.std() + 1e-8)
            sig_t = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 256)
            feat_t = torch.tensor(extract_features_standalone(sig_arr), dtype=torch.float32).unsqueeze(0) # (1, 15)
            
            with torch.no_grad():
                if model_type.lower() == "cnn":
                    from train_cnn_model import compute_welch_psd
                    psd = compute_welch_psd(sig_arr)
                    psd_t = torch.tensor(psd, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 33)
                    out = model(sig_t, psd_t, feat_t)
                else:
                    out = model(sig_t, feat_t)
                prob_t = torch.softmax(out, dim=1)
                pred_idx = int(torch.argmax(prob_t, dim=1).item())
                probs = prob_t.squeeze(0).numpy()
            pred_label = gestures[pred_idx] if pred_idx < len(gestures) else f"Class_{pred_idx}"
            confidence = probs[pred_idx] * 100.0 if probs is not None else 100.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f" Sample {i+1:02d} | Predicted: >> {pred_label:12s} << (Confidence: {confidence:5.1f}%) | Latency: {latency_ms:.3f} ms")

    print("-" * 70)
    print(" Inference complete! You can deploy this file on any computer with Python.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal EMG Inference Predictor")
    parser.add_argument("--model", type=str, default="kaggle", help="Model type: kaggle, rf, tcn, bilstm, cnn")
    parser.add_argument("--input", type=str, default=None, help="Optional CSV file containing raw signals")
    args = parser.parse_args()
    run_prediction(model_type=args.model, input_file=args.input)
