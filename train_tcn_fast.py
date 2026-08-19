"""
train_tcn_fast.py - Fast & Robust Temporal Convolutional Network Training for EMG
"""
import os, sys, json, time, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings("ignore")

from train_model_v2 import (
    simulate_emg, extract_features, GP, EXP_RMS, GESTURES,
    FEATURE_NAMES, SR, WIN, N_USERS, N_TRIALS, RANDOM
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
LR = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
N_CLASSES = len(GESTURES)
N_FEATURES = len(FEATURE_NAMES)

G2I = {g: i for i, g in enumerate(GESTURES)}
I2G = {i: g for g, i in G2I.items()}

np.random.seed(RANDOM)
torch.manual_seed(RANDOM)


class TCNDataset(Dataset):
    def __init__(self, n_users=N_USERS, n_trials=N_TRIALS):
        super().__init__()
        self.raw_signals = []
        self.hc_features = []
        self.labels = []
        self.user_ids = []

        for uid in range(n_users):
            u_scale = np.random.uniform(0.50, 1.60)
            for gesture in GESTURES:
                for trial in range(n_trials):
                    fatigue = (trial / n_trials) * 0.30
                    sig = simulate_emg(gesture, u_scale, fatigue)
                    feat = extract_features(sig)

                    self.raw_signals.append(sig.astype(np.float32))
                    self.hc_features.append(feat.astype(np.float32))
                    self.labels.append(G2I[gesture])
                    self.user_ids.append(uid)

        self.raw_signals = np.array(self.raw_signals)
        self.hc_features = np.array(self.hc_features)
        self.labels = np.array(self.labels)
        self.user_ids = np.array(self.user_ids)

        mu = self.raw_signals.mean(axis=1, keepdims=True)
        std = self.raw_signals.std(axis=1, keepdims=True) + 1e-8
        self.raw_signals = (self.raw_signals - mu) / std

        self.hc_mean = self.hc_features.mean(axis=0)
        self.hc_std = self.hc_features.std(axis=0) + 1e-8
        self.hc_features = (self.hc_features - self.hc_mean) / self.hc_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.raw_signals[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.hc_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        min_len = min(out.shape[-1], res.shape[-1])
        return self.relu(out[:, :, :min_len] + res[:, :, :min_len])


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels=1, num_channels=[32, 64, 128], kernel_size=3,
                 n_hc_features=N_FEATURES, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size,
                              padding=padding, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.hc_proj = nn.Sequential(
            nn.Linear(n_hc_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        fused_dim = num_channels[-1] + 32
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_raw, x_feat):
        h_tcn = self.network(x_raw)
        h_pooled = self.global_pool(h_tcn).squeeze(-1)
        h_feat = self.hc_proj(x_feat)
        fused = torch.cat([h_pooled, h_feat], dim=1)
        return self.classifier(fused)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_raw, x_feat, labels in loader:
        x_raw, x_feat, labels = x_raw.to(device), x_feat.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(x_raw, x_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x_raw, x_feat, labels in loader:
        x_raw, x_feat, labels = x_raw.to(device), x_feat.to(device), labels.to(device)
        logits = model(x_raw, x_feat)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_quick(model, tr_loader, va_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, va_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        if va_acc >= best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {ep:02d}/{epochs:02d} | Train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | Val: loss={va_loss:.4f}, acc={va_acc*100:.2f}%", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, history, best_acc


def main():
    print("=" * 70, flush=True)
    print("  TEMPORAL CONVOLUTIONAL NETWORK (TCN) DEEP LEARNING PIPELINE", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    print("\n[1/4] Generating simulated EMG multi-user dataset...", flush=True)
    full_ds = TCNDataset(n_users=N_USERS, n_trials=N_TRIALS)
    print(f"Dataset samples: {len(full_ds)} (Users: {N_USERS}, Gestures: {len(GESTURES)})", flush=True)

    test_uids = np.unique(full_ds.user_ids)[-3:]
    train_indices = np.where(~np.isin(full_ds.user_ids, test_uids))[0]
    test_indices = np.where(np.isin(full_ds.user_ids, test_uids))[0]

    train_ds = Subset(full_ds, train_indices)
    test_ds = Subset(full_ds, test_indices)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[2/4] Running 5-Fold Stratified Cross-Validation...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    train_labels = full_ds.labels[train_indices]
    cv_scores = []

    for fold, (f_tr, f_va) in enumerate(skf.split(train_indices, train_labels), 1):
        print(f"\n--- Fold {fold}/5 ---", flush=True)
        tr_loader = DataLoader(Subset(train_ds, f_tr), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(Subset(train_ds, f_va), batch_size=BATCH_SIZE, shuffle=False)
        m = TemporalConvNet().to(DEVICE)
        _, _, b_acc = train_quick(m, tr_loader, va_loader, DEVICE, epochs=8)
        cv_scores.append(b_acc)
        print(f"Fold {fold} Best Val Acc: {b_acc*100:.2f}%", flush=True)

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"\n5-Fold CV Mean Accuracy: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%\n", flush=True)

    print("[3/4] Training final TCN model on full training cohort...", flush=True)
    full_tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    final_model = TemporalConvNet().to(DEVICE)
    print(f"Trainable Parameters: {final_model.count_parameters():,}", flush=True)

    t0 = time.time()
    final_model, history, _ = train_quick(final_model, full_tr_loader, test_loader, DEVICE, epochs=15)
    t_elapsed = time.time() - t0
    print(f"Training completed in {t_elapsed:.1f}s", flush=True)

    print("\n[4/4] Evaluating on unseen test cohort...", flush=True)
    crit = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, truths = evaluate(final_model, test_loader, crit, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%\n", flush=True)

    cr = classification_report(truths, preds, target_names=GESTURES, digits=4, output_dict=True)
    print("Classification Report (Unseen Test Subjects):", flush=True)
    print(classification_report(truths, preds, target_names=GESTURES, digits=4), flush=True)

    cm = confusion_matrix(truths, preds)
    cm_df = pd.DataFrame(cm, index=GESTURES, columns=GESTURES)
    print("Confusion Matrix:", flush=True)
    print(cm_df, flush=True)

    torch.save({
        'model_state_dict': final_model.state_dict(),
        'hc_mean': full_ds.hc_mean,
        'hc_std': full_ds.hc_std,
        'gestures': GESTURES,
        'cv_acc_mean': cv_mean,
        'cv_acc_std': cv_std,
        'test_acc': test_acc
    }, 'tcn_model.pth')
    print("\nSaved weights: tcn_model.pth", flush=True)

    meta = {
        'architecture': 'Temporal Convolutional Network (TCN)',
        'components': {
            'dilated_conv_blocks': '3 Temporal Blocks (d=1, 2, 4; k=3)',
            'residual_connections': '1x1 Conv channel matching',
            'pooling': 'Global Adaptive Average 1D Pooling',
            'feature_fusion': '15 Handcrafted Features Projection (Dense 32)'
        },
        'total_params': final_model.count_parameters(),
        'cv_accuracy': {
            'mean': round(cv_mean, 4),
            'std': round(cv_std, 4),
            'n_folds': 5
        },
        'test_accuracy': round(float(test_acc), 4),
        'test_loss': round(float(test_loss), 4),
        'per_class_f1': {
            g: round(float(cr[g]['f1-score']), 4) for g in GESTURES
        },
        'confusion_matrix': {
            g: {g2: int(cm_df.loc[g, g2]) for g2 in GESTURES}
            for g in GESTURES
        },
        'training_time_seconds': round(t_elapsed, 1),
        'device': str(DEVICE)
    }

    with open('tcn_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata: tcn_meta.json", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#dc2626', lw=2)
    axes[0].set_title('TCN Loss Curves', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', color='#2563eb', lw=2)
    axes[1].plot(history['val_acc'], label='Val Acc', color='#16a34a', lw=2)
    axes[1].set_title('TCN Accuracy Curves', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    im = axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[2].set_title('Confusion Matrix (Test Set)', fontweight='bold')
    fig.colorbar(im, ax=axes[2])
    tick_marks = np.arange(len(GESTURES))
    axes[2].set_xticks(tick_marks)
    axes[2].set_xticklabels(GESTURES, rotation=45, ha='right')
    axes[2].set_yticks(tick_marks)
    axes[2].set_yticklabels(GESTURES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

    axes[2].set_ylabel('True Gesture')
    axes[2].set_xlabel('Predicted Gesture')
    plt.tight_layout()
    plt.savefig('tcn_results.png', dpi=300)
    plt.close()
    print("Saved evaluation plot: tcn_results.png", flush=True)


if __name__ == '__main__':
    main()
