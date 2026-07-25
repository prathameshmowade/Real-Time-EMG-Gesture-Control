"""
train_model_v2.py — Enhanced EMG Gesture Recognition Pipeline
Features  : 15 time-domain features (Hudgins + Hjorth + DASDV + MYOP)
Models    : GNB, Random Forest, SVM, Gradient Boosting → Weighted Ensemble
Evaluation: 5-fold CV + user-independent test + calibration simulation
Output    : emg_model_v2.pkl + model_meta_v2.json + feature_weights.json
"""
import numpy as np, pandas as pd, json, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
SR         = 500
WIN        = 256
GESTURES   = ['FIST','OPEN_HAND','WRIST_UP','WRIST_DOWN','DOUBLE_FLEX','RELAX']
N_USERS    = 20       # simulated users for training
N_TRIALS   = 60       # trials per gesture per user
RANDOM     = 42
np.random.seed(RANDOM)

# Gesture EMG parameters (amplitude, burst Hz, noise, DC offset)
GP = {
    'FIST':        (0.82, 155, 0.14, 0.022),
    'OPEN_HAND':   (0.50, 102, 0.10, 0.011),
    'WRIST_UP':    (0.67, 128, 0.12, 0.016),
    'WRIST_DOWN':  (0.60, 113, 0.11, 0.013),
    'DOUBLE_FLEX': (1.02, 178, 0.20, 0.031),
    'RELAX':       (0.04,  28, 0.02, 0.000),
}
EXP_RMS = {'RELAX':0.040,'FIST':0.520,'OPEN_HAND':0.310,'WRIST_UP':0.420,'WRIST_DOWN':0.370,'DOUBLE_FLEX':0.670}

# ═══════════════════════════════════════════════════
# EMG SIGNAL SIMULATION  (realistic multi-component)
# ═══════════════════════════════════════════════════
def simulate_emg(gesture, user_scale=1.0, fatigue=0.0, artifacts=True):
    amp, hz, noise, dc = GP[gesture]
    t = np.linspace(0, WIN/SR, WIN)
    fatigue_factor = 1.0 - 0.20 * fatigue   # up to 20% amplitude drop

    # Primary EMG: amplitude-modulated Gaussian noise
    am = amp * user_scale * fatigue_factor * (
        1 + 0.14 * np.sin(2*np.pi*1.8*t) + 0.06 * np.sin(2*np.pi*0.4*t)
    )
    emg = am * np.random.randn(WIN)

    # Spectral content at characteristic frequencies
    emg += amp * user_scale * 0.07 * np.sin(2*np.pi*hz*t)
    emg += amp * user_scale * 0.04 * np.sin(2*np.pi*(hz*0.6)*t)

    # Electrode noise floor + DC
    emg += noise * np.random.randn(WIN)
    emg += dc + 0.005 * np.random.randn()

    # Motion artifact (random, 8% chance)
    if artifacts and np.random.rand() < 0.08:
        pos = np.random.randint(0, WIN-20)
        emg[pos:pos+np.random.randint(5,20)] += 0.35 * np.random.randn()

    return emg

# ═══════════════════════════════════════════════════
# FEATURE EXTRACTION  — 15 features
# ═══════════════════════════════════════════════════
FEATURE_NAMES = [
    'MAV','MMAV','RMS','VAR','STD',
    'WL','AAC','DASDV',
    'ZC','SSC',
    'IEMG',
    'HjorthActivity','HjorthMobility','HjorthComplexity',
    'MYOP'
]

def extract_features(sig, zc_thresh=0.01, ssc_thresh=0.003, myop_mult=3.0):
    n   = len(sig)
    mu  = np.mean(sig)
    std = np.std(sig)

    # ── Amplitude / power features ──────────────────
    mav  = np.mean(np.abs(sig))
    # Modified MAV: full weight in middle 50%, half weight outside
    w    = np.where((np.arange(n)>=n//4) & (np.arange(n)<3*n//4), 1.0, 0.5)
    mmav = np.sum(w * np.abs(sig)) / n
    rms  = np.sqrt(np.mean(sig**2))
    var  = np.var(sig)
    iemg = np.sum(np.abs(sig))                                  # Integrated EMG

    # ── Complexity / morphology features ─────────────
    diff1  = np.diff(sig)
    diff2  = np.diff(diff1)
    wl     = np.sum(np.abs(diff1))                              # Waveform Length
    aac    = np.mean(np.abs(diff1))                             # Average Amplitude Change
    dasdv  = np.sqrt(np.mean(diff1**2))                        # DASDV

    # ── Frequency-information features ───────────────
    # Zero crossings with hysteresis
    hi, lo = sig >  zc_thresh, sig < -zc_thresh
    zc     = int(np.sum((hi[1:] & lo[:-1]) | (lo[1:] & hi[:-1])))

    # Slope sign changes
    d1, d2 = diff1[:-1], diff1[1:]
    ssc    = int(np.sum(
        (np.abs(d1-d2) >= ssc_thresh) &
        (((d1>0)&(d2<0)) | ((d1<0)&(d2>0)))
    ))

    # ── Hjorth parameters ─────────────────────────────
    activity   = var                                            # Power
    var_d1     = np.var(diff1)
    mobility   = np.sqrt(var_d1 / (var + 1e-12))             # Frequency estimate
    var_d2     = np.var(diff2)
    complexity = (np.sqrt(var_d2/(var_d1+1e-12)) /
                  (mobility + 1e-12))                          # Waveform complexity

    # ── Myopulse Percentage Rate ──────────────────────
    threshold = myop_mult * std
    myop      = float(np.sum(np.abs(sig) > threshold)) / n

    return np.array([
        mav, mmav, rms, var, std,
        wl, aac, dasdv,
        zc, ssc,
        iemg,
        activity, mobility, complexity,
        myop
    ])


if __name__ == '__main__':
    sig = simulate_emg('DOUBLE_FLEX')
    feats = extract_features(sig)
    print(f"Extracted all {len(feats)} features including Hjorth parameters.")
