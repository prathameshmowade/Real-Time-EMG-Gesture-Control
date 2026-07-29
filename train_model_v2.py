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

# ═══════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════
def build_dataset():
    print(f"Building dataset: {N_USERS} users × {len(GESTURES)} gestures × {N_TRIALS} trials")
    X, y, users = [], [], []

    for uid in range(N_USERS):
        u_scale   = np.random.uniform(0.50, 1.60)
        noise_var = np.random.uniform(0.85, 1.15)

        for gesture in GESTURES:
            for trial in range(N_TRIALS):
                fatigue = (trial / N_TRIALS) * 0.30    # up to 30% drop
                sig  = simulate_emg(gesture, u_scale, fatigue)
                feat = extract_features(sig)
                X.append(feat)
                y.append(gesture)
                users.append(uid)

    return np.array(X), np.array(y), np.array(users)

# ═══════════════════════════════════════════════════
# CALIBRATION  (per-user normalization)
# ═══════════════════════════════════════════════════
def calibrate_user(u_scale, n_reps=7):
    ratios = []
    for gesture in GESTURES:
        for _ in range(n_reps):
            sig  = simulate_emg(gesture, u_scale)
            feat = extract_features(sig)
            rms  = feat[2]   # index 2 = RMS
            if rms > 0.001:
                ratios.append(rms / EXP_RMS[gesture])
    return float(np.median(ratios))

def apply_calibration(X_vec, cal_scale):
    v = X_vec.copy().astype(float)
    # Amplitude-based features: linear with scale
    v[[0,1,2,4,6,7,10]] /= cal_scale
    # Variance features: quadratic
    v[[3,11]]           /= cal_scale**2
    # WL: linear
    v[5]                /= cal_scale
    # Count / ratio features (ZC, SSC, MYOP): unchanged
    # Hjorth Mobility / Complexity: scale-invariant approximation
    return v

# ═══════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════
def train_and_evaluate():
    print("\n" + "="*65)
    print("  EMG GESTURE RECOGNITION v2 — ENHANCED ML PIPELINE")
    print("="*65)

    X, y, users = build_dataset()
    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features")
    for g in GESTURES:
        print(f"  {g:<14}: {np.sum(y==g)} samples")

    # User-independent split: last 3 users as test
    test_uids = np.unique(users)[-3:]
    mask      = np.isin(users, test_uids)
    X_tr, y_tr = X[~mask], y[~mask]
    X_te, y_te = X[mask],  y[mask]
    print(f"\nTrain: {X_tr.shape[0]} ({N_USERS-3} users) | Test: {X_te.shape[0]} ({len(test_uids)} unseen users)")

    # ── Feature scaling ──────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── Model zoo ────────────────────────────────────
    print("\n── 5-fold CV on training set ──")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

    models_raw = {
        "Gaussian NB":        Pipeline([('sc', StandardScaler()), ('m', GaussianNB())]),
        "K-NN (k=7)":         Pipeline([('sc', StandardScaler()), ('m', KNeighborsClassifier(n_neighbors=7, metric='euclidean'))]),
        "SVM (RBF, C=10)":    Pipeline([('sc', StandardScaler()), ('m', SVC(C=10, gamma='scale', probability=True))]),
        "Random Forest 200":  RandomForestClassifier(n_estimators=200, max_features='sqrt', class_weight='balanced', random_state=RANDOM, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=RANDOM),
    }
    cv_scores = {}
    for name, clf in models_raw.items():
        XX = X_tr if "Forest" in name or "Boost" in name else X_tr
        sc = cross_val_score(clf, XX, y_tr, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_scores[name] = sc
        print(f"  {name:<25}: {sc.mean():.4f} ± {sc.std():.4f}")

    # ── Grid search on Random Forest ─────────────────
    print("\n── Grid search: Random Forest ──")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth':    [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2],
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=RANDOM, n_jobs=-1)
    gs = GridSearchCV(rf_base, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.4f}")

    # ── Best RF ───────────────────────────────────────
    rf_best = gs.best_estimator_
    print(f"\n── Best RF test accuracy: {accuracy_score(y_te, rf_best.predict(X_te)):.4f}")

    # ── Weighted Voting Ensemble ──────────────────────
    # Use models that work best raw (no scaling needed)
    print("\n── Building weighted ensemble ──")
    gnb = Pipeline([('sc', StandardScaler()), ('m', GaussianNB())])
    svm = Pipeline([('sc', StandardScaler()), ('m', SVC(C=10, gamma='scale', probability=True))])
    gnb.fit(X_tr, y_tr); svm.fit(X_tr, y_tr)

    gnb_acc = accuracy_score(y_te, gnb.predict(X_te))
    svm_acc = accuracy_score(y_te, svm.predict(X_te))
    rf_acc  = accuracy_score(y_te, rf_best.predict(X_te))

    print(f"  GNB test acc : {gnb_acc:.4f}")
    print(f"  SVM test acc : {svm_acc:.4f}")
    print(f"  RF  test acc : {rf_acc:.4f}")

    ensemble = VotingClassifier(
        estimators=[
            ('gnb', gnb),
            ('svm', svm),
            ('rf',  rf_best),
        ],
        voting='soft',
        weights=[gnb_acc, svm_acc, rf_acc]
    )
    ensemble.fit(X_tr, y_tr)
    ens_acc_tr = accuracy_score(y_tr, ensemble.predict(X_tr))
    ens_acc_te = accuracy_score(y_te, ensemble.predict(X_te))
    print(f"\n  Ensemble train: {ens_acc_tr:.4f} ({ens_acc_tr*100:.2f}%)")
    print(f"  Ensemble test : {ens_acc_te:.4f} ({ens_acc_te*100:.2f}%)")

    print("\nClassification report — ENSEMBLE (test set):")
    print(classification_report(y_te, ensemble.predict(X_te), target_names=GESTURES))

    cm = confusion_matrix(y_te, ensemble.predict(X_te), labels=GESTURES)
    cm_df = pd.DataFrame(cm, index=GESTURES, columns=GESTURES)
    print("Confusion matrix:")
    print(cm_df)

    # ── Feature importance ────────────────────────────
    fi = pd.Series(rf_best.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("\nFeature Importances (Random Forest):")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {imp:.4f}  {bar}")

    # ── Calibration improvement ───────────────────────
    print("\n── Calibration simulation (8 new users) ──")
    raw_accs, cal_accs = [], []
    for i in range(8):
        u_scale = np.random.uniform(0.40, 1.80)
        X_new, y_new = [], []
        for g in GESTURES:
            for _ in range(25):
                sig  = simulate_emg(g, u_scale)
                feat = extract_features(sig)
                X_new.append(feat)
                y_new.append(g)
        X_new = np.array(X_new)

        raw = accuracy_score(y_new, ensemble.predict(X_new))

        cal_scale = calibrate_user(u_scale)
        X_cal = np.array([apply_calibration(v, cal_scale) for v in X_new])
        cal   = accuracy_score(y_new, ensemble.predict(X_cal))

        raw_accs.append(raw); cal_accs.append(cal)
        print(f"  User {i+1} (scale={u_scale:.2f}×, est={cal_scale:.2f}×):"
              f"  raw={raw:.1%}  calibrated={cal:.1%}  Δ={cal-raw:+.1%}")

    print(f"\n  Mean raw:        {np.mean(raw_accs):.2%}")
    print(f"  Mean calibrated: {np.mean(cal_accs):.2%}")
    print(f"  Improvement:     {np.mean(cal_accs)-np.mean(raw_accs):+.2%}")

    # ── Save ──────────────────────────────────────────
    print("\n── Saving models ──")
    joblib.dump({'model':ensemble,'scaler':scaler,'rf':rf_best}, 'emg_model_v2.pkl')

    meta = {
        'feature_names': FEATURE_NAMES,
        'gestures': GESTURES,
        'n_features': 15,
        'window_size': WIN,
        'sample_rate': SR,
        'n_users_train': N_USERS-3,
        'n_trials': N_TRIALS,
        'gnb_test_acc': round(gnb_acc, 4),
        'svm_test_acc': round(svm_acc, 4),
        'rf_test_acc':  round(rf_acc,  4),
        'ens_train_acc': round(ens_acc_tr, 4),
        'ens_test_acc':  round(ens_acc_te, 4),
        'cv_scores': {k: {'mean': round(v.mean(),4), 'std': round(v.std(),4)}
                      for k,v in cv_scores.items()},
        'best_rf_params': gs.best_params_,
        'feature_importances': fi.to_dict(),
        'calibration_improvement': round(float(np.mean(cal_accs)-np.mean(raw_accs)),4),
        'confusion_matrix': {g:{g2:int(cm_df.loc[g,g2]) for g2 in GESTURES} for g in GESTURES}
    }
    with open('model_meta_v2.json','w') as f:
        json.dump(meta, f, indent=2)

    # GNB parameters for Pico W (extract from sklearn GaussianNB)
    gnb_model = gnb.named_steps['m']
    gnb_sc    = gnb.named_steps['sc']
    pico_weights = {
        'classes':   GESTURES,
        'priors':    gnb_model.class_prior_.tolist(),
        'means':     {GESTURES[i]: gnb_model.theta_[i].tolist() for i in range(len(GESTURES))},
        'variances': {GESTURES[i]: gnb_model.var_[i].tolist()   for i in range(len(GESTURES))},
        'scaler_mean': gnb_sc.mean_.tolist(),
        'scaler_std':  gnb_sc.scale_.tolist(),
        'feature_names': FEATURE_NAMES,
        'accuracy': round(gnb_acc, 4),
        'note': 'Copy GNB_MEANS / GNB_VARS into pico_main.py after dividing by scaler_std'
    }
    with open('feature_weights.json','w') as f:
        json.dump(pico_weights, f, indent=2)

    print("  Saved: emg_model_v2.pkl")
    print("  Saved: model_meta_v2.json")
    print("  Saved: feature_weights.json  ← paste into pico_main.py")
    return ensemble, meta

if __name__ == '__main__':
    model, meta = train_and_evaluate()
    print(f"\n{'='*65}")
    print(f"  Final ensemble test accuracy: {meta['ens_test_acc']*100:.2f}%")
    print(f"  Calibration improvement:      {meta['calibration_improvement']*100:+.2f}%")
    print(f"{'='*65}")
