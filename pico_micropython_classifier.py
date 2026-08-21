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
                self.means = data.get("means", {})          # dict: {gesture: [15 means]}
                self.variances = data.get("variances", {})  # dict: {gesture: [15 vars]}
                self.scaler_mean = data.get("scaler_mean", [0.0] * 15)
                self.scaler_std = data.get("scaler_std", [1.0] * 15)
                self.feature_names = data.get("feature_names", [])
                print(f"[Pico Classifier] Loaded {len(self.means)} classes with {len(self.feature_names)} features.")
        except Exception as e:
            print(f"[Pico Classifier] Warning loading {weights_json_path}: {e}")
            self.means = {}
            self.variances = {}
            self.scaler_mean = [0.0] * 15
            self.scaler_std = [1.0] * 15

    def extract_features(self, signal):
        """Extracts 15 time-domain features on-chip in ~1-2 ms without external libraries."""
        N = len(signal)
        if N == 0:
            return [0.0] * 15

        # 1-5. Amplitude & Power features
        sum_abs = sum(abs(x) for x in signal)
        mav = sum_abs / N

        # MMAV (weighted window)
        sum_w_abs = 0.0
        q1, q3 = N // 4, (3 * N) // 4
        for i, x in enumerate(signal):
            w = 1.0 if (q1 <= i < q3) else 0.5
            sum_w_abs += w * abs(x)
        mmav = sum_w_abs / N

        sum_sq = sum(x * x for x in signal)
        rms = math.sqrt(sum_sq / N)
        mean_val = sum(signal) / N
        var = sum((x - mean_val) ** 2 for x in signal) / N
        std = math.sqrt(var)

        # 6-8. Morphology & Differences
        diff1 = []
        wl = 0.0
        diff_sq_sum = 0.0
        zc = 0
        for i in range(1, N):
            d = signal[i] - signal[i-1]
            diff1.append(d)
            wl += abs(d)
            diff_sq_sum += d * d
            # Zero crossings with noise threshold
            if (signal[i] * signal[i-1] < 0) and abs(d) > 0.01:
                zc += 1

        aac = wl / (N - 1) if N > 1 else 0.0
        dasdv = math.sqrt(diff_sq_sum / (N - 1)) if N > 1 else 0.0

        # 9-10. Slope Sign Changes (SSC)
        ssc = 0
        diff2 = []
        for i in range(1, len(diff1)):
            d1, d2 = diff1[i-1], diff1[i]
            diff2.append(d2 - d1)
            if (d1 * d2 < 0) and abs(d1 - d2) >= 0.003:
                ssc += 1

        # 11. Integrated EMG (IEMG)
        iemg = sum_abs

        # 12-14. Hjorth Parameters
        h_act = var
        var_d1 = (sum(d * d for d in diff1) / len(diff1)) if diff1 else 1e-12
        h_mob = math.sqrt(var_d1 / (var + 1e-12))
        var_d2 = (sum(d * d for d in diff2) / len(diff2)) if diff2 else 1e-12
        h_comp = (math.sqrt(var_d2 / (var_d1 + 1e-12)) / (h_mob + 1e-12)) if h_mob > 0 else 0.0

        # 15. Myopulse Percentage Rate (MYOP)
        myop_thresh = 3.0 * std
        myop = sum(1 for x in signal if abs(x) > myop_thresh) / N

        return [mav, mmav, rms, var, std, wl, aac, dasdv, float(zc), float(ssc), iemg, h_act, h_mob, h_comp, myop]

    def predict(self, signal):
        """Predicts gesture class using on-device Gaussian maximum a posteriori likelihood."""
        feats = self.extract_features(signal)
        
        # If dictionary weights are loaded, compute log-likelihoods
        if isinstance(self.means, dict) and len(self.means) > 0:
            best_class = GESTURES[0]
            best_log_prob = -1e9

            # Apply standard scaling
            norm_feats = []
            for i, f in enumerate(feats):
                mu = self.scaler_mean[i] if i < len(self.scaler_mean) else 0.0
                st = self.scaler_std[i] if i < len(self.scaler_std) else 1.0
                norm_feats.append((f - mu) / (st + 1e-8))

            for c_idx, g_name in enumerate(GESTURES):
                prior_val = self.priors[c_idx] if c_idx < len(self.priors) else (1.0 / len(GESTURES))
                log_prob = math.log(max(prior_val, 1e-6))
                
                c_means = self.means.get(g_name, [])
                c_vars = self.variances.get(g_name, [])

                # Sum log Gaussian likelihood across extracted features
                n_eval = min(len(norm_feats), len(c_means))
                for f_idx in range(n_eval):
                    v = max(c_vars[f_idx], 1e-6)
                    diff = norm_feats[f_idx] - c_means[f_idx]
                    log_prob -= 0.5 * (math.log(2 * math.pi * v) + (diff * diff) / v)

                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_class = g_name

            return best_class

        # Fast fallback threshold heuristic (ultra-low power)
        rms = feats[2]  # index 2 is RMS
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
