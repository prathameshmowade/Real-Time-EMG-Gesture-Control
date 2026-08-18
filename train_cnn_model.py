"""
train_cnn_model.py - Dual-Path Temporal-Spectral Fusion CNN (DTSF-CNN)
======================================================================
A custom deep learning architecture for EMG gesture classification that
processes raw signals through two parallel pathways:

  Path A: Multi-scale temporal convolutions (k=7, k=15, k=31)
  Path B: Learned spectral decomposition via Welch PSD

Fused with an adaptive sigmoid gate and conditioned on 15 handcrafted
features via Feature-wise Linear Modulation (FiLM).

Reuses signal simulation & feature extraction from train_model_v2.py.
Output: cnn_model.pth, cnn_meta.json, cnn_results.png
"""

import os, sys, json, time, warnings
# Reconfigure output streams to handle UTF-8 safely
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from scipy.signal import welch as scipy_welch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

warnings.filterwarnings("ignore")

# -- Import shared utilities from existing pipeline ------------------
from train_model_v2 import (simulate_emg, extract_features,
                            GP, EXP_RMS, GESTURES, FEATURE_NAMES,
                            SR, WIN, N_USERS, N_TRIALS, RANDOM)

# =====================================================================
# CONFIG
# =====================================================================
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
PATIENCE     = 15           # early-stopping patience
N_FOLDS      = 5
N_CLASSES    = len(GESTURES)
N_FEATURES   = len(FEATURE_NAMES)   # 15 handcrafted features for FiLM
WELCH_NPERSEG = 64          # Welch PSD segment length -> 33 freq bins

# Gesture label -> integer mapping
G2I = {g: i for i, g in enumerate(GESTURES)}
I2G = {i: g for g, i in G2I.items()}

np.random.seed(RANDOM)
torch.manual_seed(RANDOM)

# =====================================================================
# SPECTRAL TRANSFORM  (Welch Power Spectral Density)
# =====================================================================
def compute_welch_psd(signal, fs=SR, nperseg=WELCH_NPERSEG):
    """Compute normalized Welch PSD for a single EMG window.

    Returns a 1-D array of shape (n_freq_bins,) where n_freq_bins
    is determined by nperseg (default 64 -> 33 bins).
    Log10-scaled and standardized for neural network consumption.
    """
    freqs, psd = scipy_welch(signal, fs=fs, nperseg=nperseg,
                             noverlap=nperseg // 2, scaling='density')
    # Log-scale to compress dynamic range (add small epsilon)
    psd_log = np.log10(psd + 1e-12)
    return psd_log.astype(np.float32)


# =====================================================================
# DATASET
# =====================================================================
class EMGDataset(Dataset):
    """PyTorch Dataset that stores raw signals, Welch PSD, handcrafted
    features, and labels. Built once from the simulate_emg generator."""

    def __init__(self, n_users=N_USERS, n_trials=N_TRIALS):
        super().__init__()
        self.raw_signals  = []   # (N, WIN)
        self.psd_vectors  = []   # (N, n_freq_bins)
        self.hc_features  = []   # (N, 15) handcrafted
        self.labels       = []   # (N,) int
        self.user_ids     = []   # (N,) int

        for uid in range(n_users):
            u_scale = np.random.uniform(0.50, 1.60)
            for gesture in GESTURES:
                for trial in range(n_trials):
                    fatigue = (trial / n_trials) * 0.30
                    sig  = simulate_emg(gesture, u_scale, fatigue)
                    feat = extract_features(sig)
                    psd  = compute_welch_psd(sig)

                    self.raw_signals.append(sig.astype(np.float32))
                    self.psd_vectors.append(psd)
                    self.hc_features.append(feat.astype(np.float32))
                    self.labels.append(G2I[gesture])
                    self.user_ids.append(uid)

        self.raw_signals = np.array(self.raw_signals)
        self.psd_vectors = np.array(self.psd_vectors)
        self.hc_features = np.array(self.hc_features)
        self.labels      = np.array(self.labels)
        self.user_ids    = np.array(self.user_ids)

        # Normalize raw signals per-sample (zero-mean, unit-var)
        mu  = self.raw_signals.mean(axis=1, keepdims=True)
        std = self.raw_signals.std(axis=1, keepdims=True) + 1e-8
        self.raw_signals = (self.raw_signals - mu) / std

        # Normalize PSD globally
        psd_mu  = self.psd_vectors.mean()
        psd_std = self.psd_vectors.std() + 1e-8
        self.psd_vectors = (self.psd_vectors - psd_mu) / psd_std

        # Normalize handcrafted features globally
        self.hc_mean = self.hc_features.mean(axis=0)
        self.hc_std  = self.hc_features.std(axis=0) + 1e-8
        self.hc_features = (self.hc_features - self.hc_mean) / self.hc_std

        self.n_freq_bins = self.psd_vectors.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.raw_signals[idx]).unsqueeze(0),   # (1, WIN)
                torch.tensor(self.psd_vectors[idx]).unsqueeze(0),   # (1, n_freq)
                torch.tensor(self.hc_features[idx]),                # (15,)
                torch.tensor(self.labels[idx], dtype=torch.long))


# =====================================================================
# MODEL BLOCKS
# =====================================================================

class ResBlock1D(nn.Module):
    """1-D residual block with two convolutions and skip connection."""
    def __init__(self, channels, kernel_size=3, dropout=DROPOUT):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class MultiScaleTemporalBlock(nn.Module):
    """Path A: Three parallel Conv1D branches with different kernel sizes
    to capture multi-resolution temporal activation patterns.

    k=7  (~14ms @ 500Hz) -> motor unit twitch transients
    k=15 (~30ms @ 500Hz) -> voluntary contraction onset/offset
    k=31 (~62ms @ 500Hz) -> sustained grip force envelopes
    """
    def __init__(self, in_channels=1, branch_channels=32, dropout=DROPOUT):
        super().__init__()
        self.branch_7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.branch_15 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.branch_31 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=31, padding=15),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        total_ch = branch_channels * 3
        self.res1 = ResBlock1D(total_ch, kernel_size=5, dropout=dropout)
        self.res2 = ResBlock1D(total_ch, kernel_size=5, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)   # global average pooling

    def forward(self, x):
        # x: (B, 1, WIN)
        b7  = self.branch_7(x)    # (B, 32, WIN)
        b15 = self.branch_15(x)   # (B, 32, WIN)
        b31 = self.branch_31(x)   # (B, 32, WIN)
        h   = torch.cat([b7, b15, b31], dim=1)   # (B, 96, WIN)
        h   = self.res1(h)
        h   = self.res2(h)
        h   = self.pool(h).squeeze(-1)   # (B, 96)
        return h


class SpectralAttentionBlock(nn.Module):
    """Path B: CNN on Welch PSD with squeeze-excitation attention.

    Learns which frequency bands are most discriminative per gesture.
    The SE block recalibrates channel responses based on global
    spectral statistics.
    """
    def __init__(self, in_channels=1, out_channels=48, n_freq_bins=33,
                 dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.res = ResBlock1D(out_channels, kernel_size=3, dropout=dropout)

        # Squeeze-Excitation attention
        se_hidden = max(out_channels // 4, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, se_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(se_hidden, out_channels),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_psd):
        # x_psd: (B, 1, n_freq_bins)
        h = self.conv1(x_psd)     # (B, 48, n_freq)
        h = self.conv2(h)         # (B, 48, n_freq)
        h = self.res(h)           # (B, 48, n_freq)

        # Squeeze-Excitation recalibration
        se_weights = self.se(h).unsqueeze(-1)   # (B, 48, 1)
        h = h * se_weights                      # channel-wise scaling

        h = self.pool(h).squeeze(-1)   # (B, 48)
        return h


class AdaptiveFusionGate(nn.Module):
    """Sigmoid-gated fusion of temporal and spectral feature vectors.

    Learns a per-dimension gate that balances how much each pathway
    contributes. During high-SNR windows the temporal path dominates;
    during noisy windows the spectral path provides robustness.

    Formula: h = g * h_temporal + (1 - g) * h_spectral
    where g = sigmoid(W_g @ [h_temp, h_spec] + b_g)
    """
    def __init__(self, temporal_dim, spectral_dim, fused_dim):
        super().__init__()
        self.proj_temp = nn.Linear(temporal_dim, fused_dim)
        self.proj_spec = nn.Linear(spectral_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + spectral_dim, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_temp, h_spec):
        # Project both to same dimensionality
        t = self.proj_temp(h_temp)     # (B, fused_dim)
        s = self.proj_spec(h_spec)     # (B, fused_dim)

        # Compute gate from concatenated raw vectors
        g = self.gate(torch.cat([h_temp, h_spec], dim=1))   # (B, fused_dim)

        # Gated fusion
        h = g * t + (1.0 - g) * s   # (B, fused_dim)
        return h


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Injects the 15 handcrafted features as a modulation signal into the
    CNN's penultimate representation. Bridges classical domain knowledge
    with learned deep features.

    Formula: h_out = gamma(features) * h_in + beta(features)
    """
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, features):
        # h: (B, hidden_dim),  features: (B, 15)
        gamma = self.gamma_net(features)   # (B, hidden_dim)
        beta  = self.beta_net(features)    # (B, hidden_dim)
        return gamma * h + beta


# =====================================================================
# MAIN MODEL
# =====================================================================
class DTSF_CNN(nn.Module):
    """Dual-Path Temporal-Spectral Fusion CNN.

    Architecture:
      Path A (Temporal)  -> MultiScaleTemporalBlock -> 96-dim vector
      Path B (Spectral)  -> SpectralAttentionBlock  -> 48-dim vector
      Fusion             -> AdaptiveFusionGate       -> 64-dim vector
      FiLM conditioning  -> FiLMConditioner          -> 64-dim vector
      Classifier         -> Dropout -> FC -> 6 classes
    """
    def __init__(self, n_freq_bins=33, n_hc_features=N_FEATURES,
                 n_classes=N_CLASSES, temporal_channels=32,
                 spectral_channels=48, fused_dim=64, dropout=DROPOUT):
        super().__init__()

        temporal_out_dim = temporal_channels * 3   # 96

        self.temporal_path = MultiScaleTemporalBlock(
            in_channels=1, branch_channels=temporal_channels, dropout=dropout
        )
        self.spectral_path = SpectralAttentionBlock(
            in_channels=1, out_channels=spectral_channels,
            n_freq_bins=n_freq_bins, dropout=dropout
        )
        self.fusion = AdaptiveFusionGate(
            temporal_dim=temporal_out_dim,
            spectral_dim=spectral_channels,
            fused_dim=fused_dim
        )
        self.film = FiLMConditioner(
            feature_dim=n_hc_features, hidden_dim=fused_dim
        )
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, n_classes),
        )

    def forward(self, x_raw, x_psd, x_features):
        """
        Args:
            x_raw:      (B, 1, WIN) raw EMG signal
            x_psd:      (B, 1, n_freq_bins) Welch PSD
            x_features: (B, 15) handcrafted features
        Returns:
            logits: (B, n_classes)
        """
        h_temp = self.temporal_path(x_raw)       # (B, 96)
        h_spec = self.spectral_path(x_psd)       # (B, 48)
        h_fused = self.fusion(h_temp, h_spec)     # (B, 64)
        h_film  = self.film(h_fused, x_features)  # (B, 64)
        logits  = self.classifier(h_film)          # (B, 6)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# TRAINING UTILITIES
# =====================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_raw, x_psd, x_feat, labels in loader:
        x_raw  = x_raw.to(device)
        x_psd  = x_psd.to(device)
        x_feat = x_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x_raw, x_psd, x_feat)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x_raw, x_psd, x_feat, labels in loader:
        x_raw  = x_raw.to(device)
        x_psd  = x_psd.to(device)
        x_feat = x_feat.to(device)
        labels = labels.to(device)

        logits = model(x_raw, x_psd, x_feat)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS,
                lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    """Full training loop with cosine LR schedule and early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc  = 0.0
    best_state    = None
    epochs_no_imp = 0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f}  "
                  f"Val Loss={vl_loss:.4f} Acc={vl_acc:.4f}  "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")
            sys.stdout.flush()

        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val acc: {best_val_acc:.4f})")
            sys.stdout.flush()
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return history, best_val_acc


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_results(history, cm, ensemble_acc, cnn_acc, save_path):
    """Generate a 2x2 figure with training curves, confusion matrix,
    and CNN vs Ensemble comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DTSF-CNN: EMG Gesture Classification Results',
                 fontsize=16, fontweight='bold', y=0.98)

    # -- 1. Loss curves ----------------------------------
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', color='#2563eb',
            linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', color='#dc2626',
            linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- 2. Accuracy curves ------------------------------
    ax = axes[0, 1]
    ax.plot(history['train_acc'], label='Train Acc', color='#16a34a',
            linewidth=2)
    ax.plot(history['val_acc'], label='Val Acc', color='#d97706',
            linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- 3. Confusion matrix heatmap ---------------------
    ax = axes[1, 0]
    short_names = ['FIST', 'OPEN', 'W_UP', 'W_DN', 'D_FLX', 'RELX']
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(short_names, fontsize=8, rotation=45)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Test Set)')

    # Annotate cells with counts
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- 4. CNN vs Ensemble comparison -------------------
    ax = axes[1, 1]
    models = ['ML Ensemble\n(RF+SVM+GNB)', 'DTSF-CNN\n(Ours)']
    accs   = [ensemble_acc * 100, cnn_acc * 100]
    colors = ['#94a3b8', '#2563eb']
    bars   = ax.bar(models, accs, color=colors, width=0.5, edgecolor='white',
                    linewidth=2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Comparison')
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    sys.stdout.flush()


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "=" * 65)
    print("  DTSF-CNN: Dual-Path Temporal-Spectral Fusion CNN")
    print("  EMG Gesture Classification - Deep Learning Pipeline")
    print("=" * 65)
    sys.stdout.flush()

    start_time = time.time()

    # -- 1. Build dataset ---------------------------------
    print("\n[1/6] Building EMG dataset...")
    sys.stdout.flush()
    dataset = EMGDataset(n_users=N_USERS, n_trials=N_TRIALS)
    n_freq_bins = dataset.n_freq_bins
    print(f"  Total samples: {len(dataset)}")
    print(f"  Raw signal shape: (1, {WIN})")
    print(f"  Welch PSD bins: {n_freq_bins}")
    print(f"  Handcrafted features: {N_FEATURES}")
    print(f"  Classes: {N_CLASSES} -> {GESTURES}")
    sys.stdout.flush()

    # -- 2. User-independent split -------------------------
    print("\n[2/6] Splitting data (user-independent)...")
    all_uids  = np.unique(dataset.user_ids)
    test_uids = all_uids[-3:]
    test_mask  = np.isin(dataset.user_ids, test_uids)
    train_mask = ~test_mask

    train_indices = np.where(train_mask)[0]
    test_indices  = np.where(test_mask)[0]

    train_labels = dataset.labels[train_indices]
    test_labels  = dataset.labels[test_indices]
    print(f"  Train: {len(train_indices)} samples ({N_USERS-3} users)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_uids)} unseen users)")
    sys.stdout.flush()

    # -- 3. Cross-validation -------------------------------
    print(f"\n[3/6] {N_FOLDS}-fold cross-validation on training set...")
    sys.stdout.flush()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM)
    fold_accs = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_indices, train_labels), 1):
        tr_sub = train_indices[tr_idx]
        vl_sub = train_indices[vl_idx]

        tr_loader = DataLoader(Subset(dataset, tr_sub), batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=False)
        vl_loader = DataLoader(Subset(dataset, vl_sub), batch_size=BATCH_SIZE,
                               shuffle=False)

        model = DTSF_CNN(n_freq_bins=n_freq_bins).to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        # Train for CV (30 epochs with early stopping)
        _, val_acc = train_model(model, tr_loader, vl_loader, DEVICE,
                                 epochs=30, patience=8)
        fold_accs.append(val_acc)
        print(f"  Fold {fold}: Val Acc = {val_acc:.4f}")
        sys.stdout.flush()

    cv_mean = np.mean(fold_accs)
    cv_std  = np.std(fold_accs)
    print(f"\n  CV Result: {cv_mean:.4f} +/- {cv_std:.4f}")
    sys.stdout.flush()

    # -- 4. Train final model ------------------------------
    print(f"\n[4/6] Training final model on full training set...")
    sys.stdout.flush()
    # Use 10% of training data as validation for early stopping
    n_val = max(int(len(train_indices) * 0.10), BATCH_SIZE)
    perm  = np.random.permutation(len(train_indices))
    final_tr_idx = train_indices[perm[n_val:]]
    final_vl_idx = train_indices[perm[:n_val]]

    tr_loader = DataLoader(Subset(dataset, final_tr_idx), batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=False)
    vl_loader = DataLoader(Subset(dataset, final_vl_idx), batch_size=BATCH_SIZE,
                           shuffle=False)

    final_model = DTSF_CNN(n_freq_bins=n_freq_bins).to(DEVICE)
    n_params = final_model.count_parameters()
    print(f"  Model parameters: {n_params:,}")
    print(f"  Device: {DEVICE}")
    sys.stdout.flush()

    history, best_val = train_model(final_model, tr_loader, vl_loader, DEVICE,
                                    epochs=EPOCHS, patience=PATIENCE)
    print(f"  Best validation accuracy: {best_val:.4f}")
    sys.stdout.flush()

    # -- 5. Evaluate on held-out test users ----------------
    print(f"\n[5/6] Evaluating on {len(test_uids)} unseen test users...")
    sys.stdout.flush()
    te_loader = DataLoader(Subset(dataset, test_indices), batch_size=BATCH_SIZE,
                           shuffle=False)
    criterion = nn.CrossEntropyLoss()
    te_loss, te_acc, te_preds, te_labels = evaluate(
        final_model, te_loader, criterion, DEVICE)

    print(f"\n  Test Loss: {te_loss:.4f}")
    print(f"  Test Acc:  {te_acc:.4f} ({te_acc*100:.2f}%)")
    sys.stdout.flush()

    # Classification report
    target_names = [I2G[i] for i in range(N_CLASSES)]
    print("\nClassification Report (DTSF-CNN, test set):")
    report = classification_report(te_labels, te_preds,
                                   target_names=target_names)
    print(report)
    sys.stdout.flush()

    # Confusion matrix
    cm = confusion_matrix(te_labels, te_preds)
    cm_df = pd.DataFrame(cm, index=GESTURES, columns=GESTURES)
    print("Confusion Matrix:")
    print(cm_df)
    sys.stdout.flush()

    # Per-class F1
    f1_per_class = f1_score(te_labels, te_preds, average=None)
    f1_dict = {GESTURES[i]: round(float(f1_per_class[i]), 4)
               for i in range(N_CLASSES)}

    # -- 6. Compare with ensemble & save -------------------
    print(f"\n[6/6] Saving artifacts...")
    sys.stdout.flush()

    # Load ensemble accuracy from existing metadata
    ensemble_acc = 0.5065   # default
    if os.path.exists('model_meta_v2.json'):
        with open('model_meta_v2.json', 'r') as f:
            meta_v2 = json.load(f)
            ensemble_acc = meta_v2.get('ens_test_acc', 0.5065)

    improvement = (te_acc - ensemble_acc) * 100
    print(f"\n  ML Ensemble test acc:  {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    print(f"  DTSF-CNN test acc:     {te_acc:.4f} ({te_acc*100:.2f}%)")
    print(f"  Improvement:           {improvement:+.2f}%")
    sys.stdout.flush()

    elapsed = time.time() - start_time

    # Save model weights
    torch.save(final_model.state_dict(), 'cnn_model.pth')
    print(f"  Saved: cnn_model.pth")

    # Save metadata
    cnn_meta = {
        'architecture': 'DTSF-CNN (Dual-Path Temporal-Spectral Fusion CNN)',
        'components': {
            'path_a': 'MultiScaleTemporalBlock (k=7, k=15, k=31)',
            'path_b': 'SpectralAttentionBlock (Welch PSD + SE)',
            'fusion': 'AdaptiveFusionGate (sigmoid-gated)',
            'conditioning': 'FiLM (15 handcrafted EMG features)',
        },
        'total_params': n_params,
        'input_shapes': {
            'raw_signal': [1, WIN],
            'welch_psd': [1, n_freq_bins],
            'handcrafted_features': [N_FEATURES],
        },
        'hyperparameters': {
            'temporal_branch_channels': 32,
            'spectral_channels': 48,
            'fused_dim': 64,
            'dropout': DROPOUT,
            'optimizer': 'AdamW',
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'scheduler': 'CosineAnnealingLR',
            'batch_size': BATCH_SIZE,
            'max_epochs': EPOCHS,
            'early_stopping_patience': PATIENCE,
        },
        'dataset': {
            'n_users_train': N_USERS - 3,
            'n_users_test': 3,
            'n_trials_per_gesture': N_TRIALS,
            'total_samples': len(dataset),
            'train_samples': len(train_indices),
            'test_samples': len(test_indices),
        },
        'cv_accuracy': {
            'mean': round(cv_mean, 4),
            'std': round(cv_std, 4),
            'n_folds': N_FOLDS,
        },
        'test_accuracy': round(te_acc, 4),
        'test_loss': round(te_loss, 4),
        'per_class_f1': f1_dict,
        'confusion_matrix': {
            GESTURES[i]: {GESTURES[j]: int(cm[i, j]) for j in range(N_CLASSES)}
            for i in range(N_CLASSES)
        },
        'comparison_vs_ensemble': {
            'ensemble_test_acc': round(ensemble_acc, 4),
            'cnn_test_acc': round(te_acc, 4),
            'improvement_pct': round(improvement, 2),
        },
        'training_time_seconds': round(elapsed, 1),
        'device': str(DEVICE),
        'gestures': GESTURES,
        'feature_names': FEATURE_NAMES,
    }
    with open('cnn_meta.json', 'w') as f:
        json.dump(cnn_meta, f, indent=2)
    print(f"  Saved: cnn_meta.json")

    # Generate plots
    plot_results(history, cm, ensemble_acc, te_acc, 'cnn_results.png')

    # -- Summary -------------------------------------------
    print(f"\n{'='*65}")
    print(f"  DTSF-CNN Training Complete")
    print(f"  Total time:    {elapsed:.1f}s")
    print(f"  Parameters:    {n_params:,}")
    print(f"  CV Accuracy:   {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%")
    print(f"  Test Accuracy: {te_acc*100:.2f}%")
    print(f"  vs Ensemble:   {improvement:+.2f}%")
    print(f"{'='*65}")
    sys.stdout.flush()

    return final_model, cnn_meta


if __name__ == '__main__':
    model, meta = main()
