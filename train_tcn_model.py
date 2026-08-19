"""
train_tcn_model.py - Temporal Convolutional Network (TCN) for EMG Gesture Recognition
====================================================================================
Architecture:
  - Dilated Causal 1D Convolutions with exponentially expanding receptive fields (d = 1, 2, 4, 8)
  - Residual Connections with 1x1 conv channel matching
  - Weight Normalization & Spatial Dropout
  - Feature Fusion with 15 Handcrafted EMG features
  - Global Temporal Pooling & Softmax Classification Head

Outputs:
  - tcn_model.pth (Trained PyTorch weights)
  - tcn_meta.json (Performance metrics, cross-validation scores, confusion matrix)
  - tcn_results.png (Loss/accuracy curves and multi-class confusion matrix)
"""

import os, sys, json, time, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings("ignore")

from train_model_v2 import (
    simulate_emg, extract_features, GP, EXP_RMS, GESTURES,
    FEATURE_NAMES, SR, WIN, N_USERS, N_TRIALS, RANDOM
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
BATCH_SIZE = 256
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
PATIENCE = 8
N_CLASSES = len(GESTURES)
N_FEATURES = len(FEATURE_NAMES)

G2I = {g: i for i, g in enumerate(GESTURES)}
I2G = {i: g for g, i in G2I.items()}

np.random.seed(RANDOM)
torch.manual_seed(RANDOM)


class TCNDataset(Dataset):
    """PyTorch dataset for Temporal Convolutional Network."""
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

        # Standardize raw signals
        mu = self.raw_signals.mean(axis=1, keepdims=True)
        std = self.raw_signals.std(axis=1, keepdims=True) + 1e-8
        self.raw_signals = (self.raw_signals - mu) / std

        # Standardize handcrafted features
        self.hc_mean = self.hc_features.mean(axis=0)
        self.hc_std = self.hc_features.std(axis=0) + 1e-8
        self.hc_features = (self.hc_features - self.hc_mean) / self.hc_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.raw_signals[idx]).unsqueeze(0),   # (1, WIN)
            torch.tensor(self.hc_features[idx]),                # (15,)
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class TemporalBlock(nn.Module):
    """Residual Dilated Temporal Block for TCN."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.drop1,
            self.conv2, self.bn2, self.relu2, self.drop2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) with multi-scale dilation hierarchy.
    """
    def __init__(self, in_channels=1, num_channels=[32, 64, 128], kernel_size=5,
                 n_hc_features=N_FEATURES, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation_size, padding=padding, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Handcrafted Feature Projection
        self.hc_proj = nn.Sequential(
            nn.Linear(n_hc_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

        # Classification Head
        fused_dim = num_channels[-1] + 32
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_raw, x_feat):
        # x_raw: (B, 1, WIN)
        h_tcn = self.network(x_raw)                 # (B, 128, WIN)
        h_pooled = self.global_pool(h_tcn).squeeze(-1) # (B, 128)

        h_feat = self.hc_proj(x_feat)              # (B, 32)
        fused = torch.cat([h_pooled, h_feat], dim=1) # (B, 160)

        logits = self.classifier(fused)             # (B, 6)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_raw, x_feat, labels in loader:
        x_raw = x_raw.to(device)
        x_feat = x_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x_raw, x_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x_raw, x_feat, labels in loader:
        x_raw = x_raw.to(device)
        x_feat = x_feat.to(device)
        labels = labels.to(device)

        logits = model(x_raw, x_feat)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR,
                weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:02d}/{epochs:02d} | Train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | Val: loss={va_loss:.4f}, acc={va_acc*100:.2f}%", flush=True)

        if patience_cnt >= patience:
            print(f"  Early stopping triggered at epoch {ep} (best val acc: {best_val_acc*100:.2f}%)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, history, best_val_acc


def run_pipeline():
    print("\n" + "=" * 70)
    print("  TEMPORAL CONVOLUTIONAL NETWORK (TCN) DEEP LEARNING PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Build dataset
    print("\n[1/4] Generating simulated EMG dataset...")
    full_dataset = TCNDataset(n_users=N_USERS, n_trials=N_TRIALS)
    print(f"Dataset samples: {len(full_dataset)} (Users: {N_USERS}, Gestures: {len(GESTURES)})")

    # Subject-independent split
    test_uids = np.unique(full_dataset.user_ids)[-3:]
    train_mask = ~np.isin(full_dataset.user_ids, test_uids)
    test_mask = np.isin(full_dataset.user_ids, test_uids)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5-Fold Cross-Validation
    print("\n[2/4] Running 5-Fold Stratified Cross-Validation...")
    train_labels = full_dataset.labels[train_indices]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    cv_scores = []

    for fold, (f_tr_idx, f_va_idx) in enumerate(skf.split(train_indices, train_labels), 1):
        tr_sub = torch.utils.data.Subset(train_dataset, f_tr_idx)
        va_sub = torch.utils.data.Subset(train_dataset, f_va_idx)

        tr_loader = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_sub, batch_size=BATCH_SIZE, shuffle=False)

        fold_model = TemporalConvNet().to(DEVICE)
        _, _, best_acc = train_model(fold_model, tr_loader, va_loader, DEVICE, epochs=12)
        cv_scores.append(best_acc)
        print(f"  Fold {fold} Best Val Acc: {best_acc*100:.2f}%", flush=True)

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"\n5-Fold CV Mean Accuracy: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%\n", flush=True)

    # Train final model
    print("[3/4] Training final TCN model on entire training cohort...", flush=True)
    tr_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = TemporalConvNet().to(DEVICE)
    print(f"Trainable Parameters: {final_model.count_parameters():,}", flush=True)

    start_time = time.time()
    final_model, history, _ = train_model(
        final_model, tr_loader_full, test_loader, DEVICE, epochs=EPOCHS
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s", flush=True)

    # Evaluate on test set
    print("\n[4/4] Evaluating on unseen test cohort...", flush=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, truths = evaluate(final_model, test_loader, criterion, DEVICE)

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report (Unseen Test Subjects):")
    cls_report = classification_report(truths, preds, target_names=GESTURES, digits=4)
    print(cls_report)

    cm = confusion_matrix(truths, preds)
    cm_df = pd.DataFrame(cm, index=GESTURES, columns=GESTURES)
    print("Confusion Matrix:")
    print(cm_df)

    # Save weights & metadata
    torch.save(final_model.state_dict(), 'tcn_model.pth')
    print("\nSaved weights: tcn_model.pth")

    meta = {
        'architecture': 'Temporal Convolutional Network (TCN)',
        'components': {
            'dilated_hierarchy': 'Dilated Causal 1D-Conv (d=1, 2, 4, kernel=5)',
            'channels': [32, 64, 128],
            'pooling': 'Global Adaptive Average Pooling',
            'feature_fusion': '15 Handcrafted Features Projection (Dense 32)'
        },
        'total_params': final_model.count_parameters(),
        'cv_accuracy': {'mean': round(cv_mean, 4), 'std': round(cv_std, 4), 'n_folds': 5},
        'test_accuracy': round(float(test_acc), 4),
        'test_loss': round(float(test_loss), 4),
        'per_class_f1': {
            g: round(float(f1_score(truths == i, preds == i, zero_division=0)), 4)
            for i, g in enumerate(GESTURES)
        },
        'confusion_matrix': {
            g: {g2: int(cm_df.loc[g, g2]) for g2 in GESTURES}
            for g in GESTURES
        },
        'training_time_seconds': round(training_time, 1),
        'device': str(DEVICE)
    }

    with open('tcn_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata: tcn_meta.json")

    # Plot results
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
    print("Saved evaluation plot: tcn_results.png")

    return final_model, meta


if __name__ == '__main__':
    try:
        run_pipeline()
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open('tcn_error.txt', 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(1)
