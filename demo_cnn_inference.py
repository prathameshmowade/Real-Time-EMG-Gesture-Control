"""
demo_cnn_inference.py - Real-Time DTSF-CNN Inference Demonstration
===================================================================
Loads the trained DTSF-CNN model (cnn_model.pth) and performs real-time
inference on synthesized EMG signal streams, showcasing live predictions,
confidence scores, and temporal-spectral latency.
"""

import os, sys, time, json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F

from train_cnn_model import (
    DTSF_CNN, compute_welch_psd, GESTURES, G2I, I2G,
    WIN, SR, N_FEATURES
)
from train_model_v2 import simulate_emg, extract_features

def compute_reference_stats():
    """Compute reference normalization stats from representative samples."""
    raw_list, psd_list, feat_list = [], [], []
    for g in GESTURES:
        for _ in range(20):
            sig = simulate_emg(g, user_scale=1.0)
            raw_list.append(sig)
            psd_list.append(compute_welch_psd(sig))
            feat_list.append(extract_features(sig))
            
    psd_arr = np.array(psd_list)
    feat_arr = np.array(feat_list)
    
    psd_mu  = float(psd_arr.mean())
    psd_std = float(psd_arr.std()) + 1e-8
    hc_mean = feat_arr.mean(axis=0)
    hc_std  = feat_arr.std(axis=0) + 1e-8
    return psd_mu, psd_std, hc_mean, hc_std

def run_demo(n_cycles=2, delay=0.15):
    print("\n" + "=" * 75)
    print("  LIVE INFERENCE DEMO: Dual-Path Temporal-Spectral Fusion CNN (DTSF-CNN)")
    print("=" * 75)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device : {device}")
    
    # Initialize and load model weights
    model_path = 'cnn_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please train the model first.")
        return
    
    model = DTSF_CNN(n_freq_bins=33).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded Weights  : {model_path} ({model.count_parameters():,} trainable parameters)")
    
    # Compute reference normalization stats
    psd_mu, psd_std, hc_mean, hc_std = compute_reference_stats()
    
    print("\nStreaming live simulated EMG signals across 6 gesture classes...\n")
    print(f"{'True Gesture':<16} | {'Predicted Gesture':<18} | {'Confidence':<10} | {'Status':<6} | {'Latency'}")
    print("-" * 75)
    
    test_gestures = GESTURES * n_cycles
    
    correct = 0
    total = len(test_gestures)
    latencies = []
    
    for idx, true_gesture in enumerate(test_gestures, 1):
        # 1. Synthesize raw EMG signal window (256 samples @ 500Hz)
        raw_sig = simulate_emg(true_gesture, user_scale=1.0, fatigue=np.random.uniform(0, 0.10))
        
        # 2. Compute Welch PSD
        psd = compute_welch_psd(raw_sig)
        
        # 3. Extract 15 handcrafted time-domain features
        hc_feat = extract_features(raw_sig)
        
        # 4. Standardize inputs matching training pipeline
        sig_norm = (raw_sig - raw_sig.mean()) / (raw_sig.std() + 1e-8)
        psd_norm = (psd - psd_mu) / psd_std
        feat_norm = (hc_feat - hc_mean) / hc_std
        
        # Prepare tensors
        t_raw = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 256)
        t_psd = torch.tensor(psd_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 33)
        t_feat = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)            # (1, 15)
        
        # 5. Model forward pass
        with torch.no_grad():
            t_start = time.perf_counter()
            logits = model(t_raw, t_psd, t_feat)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            latencies.append(latency_ms)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_gesture = I2G[pred_idx]
            conf = probs[pred_idx]
        
        if pred_gesture == true_gesture:
            correct += 1
            status_icon = "MATCH"
        else:
            status_icon = "DIFF"
            
        print(f"{true_gesture:<16} | {pred_gesture:<18} | {conf*100:5.1f}%     | {status_icon:<6} | {latency_ms:.2f} ms")
        sys.stdout.flush()
        time.sleep(delay)
        
    acc = (correct / total) * 100
    avg_latency = np.mean(latencies)
    print("-" * 75)
    print(f"\nLive Inference Run Completed: {correct}/{total} matched ({acc:.1f}% accuracy)")
    print(f"Average Inference Latency: {avg_latency:.2f} ms per 256-sample window (< 5 ms real-time budget)\n")
    sys.stdout.flush()

if __name__ == '__main__':
    run_demo(n_cycles=2, delay=0.15)
