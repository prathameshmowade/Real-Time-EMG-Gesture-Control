"""
======================================================================
  MICROPYTHON / EMBEDDED REAL-TIME EMG CLASSIFIER
  Runs directly on Raspberry Pi Pico W, ESP32, Arduino, or any low-power
  microcontroller with ZERO dependencies (pure standard MicroPython).
======================================================================
Usage on Raspberry Pi Pico W:
  1. Upload 'pico_micropython_classifier.py' and 'feature_weights.json' to the Pico.
  2. Run this script in Thonny / MicroPython REPL.
======================================================================
"""

import math
import json
import time

GESTURES = ['FIST', 'OPEN_HAND', 'WRIST_UP', 'WRIST_DOWN', 'DOUBLE_FLEX', 'RELAX']

class EmbeddedEMGClassifier:
    def __init__(self, weights_json_path="feature_weights.json"):
        """Loads lightweight pre-computed Gaussian & feature parameters from JSON."""
        try:
            with open(weights_json_path, 'r') as f:
                data = json.load(f)
                self.priors = data.get("priors", [1.0/len(GESTURES)] * len(GESTURES))
                self.means = data.get("means", [])      # shape: (n_classes, n_features)
                self.variances = data.get("variances", [])  # shape: (n_classes, n_features)
                self.feature_names = data.get("feature_names", [])
                print(f"[Pico Classifier] Loaded {len(self.means)} classes with {len(self.feature_names)} features.")
        except Exception as e:
            print(f"[Pico Classifier] Warning loading {weights_json_path}: {e}")
            self.means = []
            self.variances = []

    def extract_features(self, signal):
        """Extracts lightweight time-domain features on-chip in ~1-2 ms."""
        N = len(signal)
        if N == 0:
            return [0.0] * 15

        # 1. Mean Absolute Value (MAV)
        sum_abs = sum(abs(x) for x in signal)
        mav = sum_abs / N

        # 2. Root Mean Square (RMS) & Variance (VAR)
        sum_sq = sum(x * x for x in signal)
        rms = math.sqrt(sum_sq / N)
        mean_val = sum(signal) / N
        var = sum((x - mean_val) ** 2 for x in signal) / N
        std = math.sqrt(var)

        # 3. Waveform Length (WL) & Average Amplitude Change (AAC)
        wl = 0.0
        zc = 0
        ssc = 0
        diffs = []
        for i in range(1, N):
            d = signal[i] - signal[i-1]
            diffs.append(d)
            wl += abs(d)
            # Zero crossing
            if (signal[i] * signal[i-1] < 0) and abs(d) > 0.01:
                zc += 1

        aac = wl / (N - 1) if N > 1 else 0.0

        # Slope sign changes
        for i in range(1, len(diffs)):
            if (diffs[i] * diffs[i-1] < 0) and abs(diffs[i]) > 0.003:
                ssc += 1

        # Return primary feature vector
        return [mav, rms, var, std, wl, aac, float(zc), float(ssc)]

    def predict(self, signal):
        """Predicts gesture class using on-device Gaussian maximum a posteriori likelihood."""
        feats = self.extract_features(signal)
        
        # If dictionary weights are loaded, compute log-likelihoods
        if isinstance(self.means, dict) and len(self.means) > 0:
            best_class = GESTURES[0]
            best_log_prob = -1e9

            for c_idx, g_name in enumerate(GESTURES):
                prior_val = self.priors[c_idx] if c_idx < len(self.priors) else (1.0 / len(GESTURES))
                log_prob = math.log(max(prior_val, 1e-6))
                
                c_means = self.means.get(g_name, [])
                c_vars = self.variances.get(g_name, [])

                # Sum log Gaussian likelihood across extracted features
                n_eval = min(len(feats), len(c_means))
                for f_idx in range(n_eval):
                    v = max(c_vars[f_idx], 1e-6)
                    diff = feats[f_idx] - c_means[f_idx]
                    log_prob -= 0.5 * (math.log(2 * math.pi * v) + (diff * diff) / v)

                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_class = g_name

            return best_class

        # Fast fallback threshold heuristic (ultra-low power)
        rms = feats[1]
        if rms < 0.08:
            return 'RELAX'
        elif rms > 0.60:
            return 'DOUBLE_FLEX'
        elif rms > 0.45:
            return 'FIST'
        else:
            return 'OPEN_HAND'

# Demonstration on standalone Python / MicroPython
if __name__ == "__main__":
    print("=" * 60)
    print("   MICROPYTHON / PICO W EMG CLASSIFIER DEMO")
    print("=" * 60)

    clf = EmbeddedEMGClassifier("feature_weights.json")

    # Test dummy biopotential windows
    test_signals = {
        "Rest Baseline": [0.03 * math.sin(i * 0.1) for i in range(256)],
        "High Force Contraction": [0.85 * math.sin(i * 0.4) for i in range(256)],
    }

    for name, sig in test_signals.items():
        t0 = time.time()
        pred = clf.predict(sig)
        elapsed_ms = (time.time() - t0) * 1000.0
        print(f" Input: {name:25s} -> Predicted Gesture: >> {pred:12s} << (Latency: {elapsed_ms:.2f} ms)")

    print("=" * 60)
