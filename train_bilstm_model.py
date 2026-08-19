"""
train_bilstm_model.py - CNN-BiLSTM Spatial-Temporal Network for EMG Gesture Recognition
========================================================================================
Architecture:
  - 1D-CNN Front-End: Multi-layer 1D convolutions to extract localized morphological patterns
  - Bidirectional LSTM: 2-layer BiLSTM to capture bidirectional temporal sequence dynamics
  - Self-Attention Temporal Pooling: Learns attention weights across time steps
  - Feature Fusion: Combines temporal embeddings with 15 handcrafted time-domain features
  - Classification Head: Multi-class softmax output across 6 gesture classes

Outputs:
  - bilstm_model.pth (Trained PyTorch weights)
  - bilstm_meta.json (Performance metrics, cross-validation scores, confusion matrix)
  - bilstm_results.png (Loss/accuracy curves and multi-class confusion matrix)
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

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 15
N_CLASSES = len(GESTURES)
N_FEATURES = len(FEATURE_NAMES)

G2I = {g: i for i, g in enumerate(GESTURES)}
I2G = {i: g for g, i in G2I.items()}

np.random.seed(RANDOM)
torch.manual_seed(RANDOM)


class BiLSTMDataset(Dataset):
    """Dataset providing raw signals, handcrafted features, and class labels."""
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

        # Per-sample zero-mean unit-variance scaling on raw signals
        mu = self.raw_signals.mean(axis=1, keepdims=True)
        std = self.raw_signals.std(axis=1, keepdims=True) + 1e-8
        self.raw_signals = (self.raw_signals - mu) / std

        # Global standardization on handcrafted features
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


class TemporalAttention(nn.Module):
    """Self-attention mechanism over sequential time steps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_out):
        # rnn_out: (B, SeqLen, HiddenDim)
        scores = self.attn(rnn_out)              # (B, SeqLen, 1)
        weights = F.softmax(scores, dim=1)        # (B, SeqLen, 1)
        context = torch.sum(rnn_out * weights, dim=1) # (B, HiddenDim)
        return context


class CNN_BiLSTM(nn.Module):
    """
    CNN-BiLSTM Architecture for Biosignal Gesture Recognition.
    Combines 1D-CNN spatial-morphological feature extraction with 
    Bidirectional LSTM temporal recurrence and self-attention pooling.
    """
    def __init__(self, in_channels=1, cnn_channels=48, lstm_hidden=64,
                 lstm_layers=2, n_hc_features=N_FEATURES, n_classes=N_CLASSES,
                 dropout=DROPOUT):
        super().__init__()

        # 1D-CNN Front-End Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels * 2, cnn_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention Temporal Pooling
        self.attention = TemporalAttention(lstm_hidden * 2)

        # Handcrafted Feature Dense Projection
        self.hc_proj = nn.Sequential(
            nn.Linear(n_hc_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification Head
        fused_dim = (lstm_hidden * 2) + 32
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_raw, x_feat):
        # x_raw: (B, 1, WIN) -> (B, 96, 64)
        c_out = self.cnn(x_raw)
        # Permute for LSTM: (B, SeqLen=64, Features=96)
        c_perm = c_out.permute(0, 2, 1)

        # BiLSTM forward: (B, 64, 128)
        lstm_out, _ = self.lstm(c_perm)

        # Attention pooling: (B, 128)
        lstm_ctx = self.attention(lstm_out)

        # Handcrafted feature branch: (B, 32)
        h_feat = self.hc_proj(x_feat)

        # Multimodal fusion: (B, 160)
        fused = torch.cat([lstm_ctx, h_feat], dim=1)

        # Logits: (B, 6)
        logits = self.classifier(fused)
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
            print(f"  Epoch {ep:02d}/{epochs:02d} | Train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | Val: loss={va_loss:.4f}, acc={va_acc*100:.2f}%")

        if patience_cnt >= patience:
            print(f"  Early stopping triggered at epoch {ep} (best val acc: {best_val_acc*100:.2f}%)")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, history, best_val_acc


def run_pipeline():
    print("\n" + "=" * 70)
    print("  CNN-BiLSTM SPATIAL-TEMPORAL DEEP LEARNING PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Build dataset
    print("\n[1/4] Generating simulated EMG multi-user dataset...")
    full_dataset = BiLSTMDataset(n_users=N_USERS, n_trials=N_TRIALS)
    print(f"Dataset samples: {len(full_dataset)} (Users: {N_USERS}, Gestures: {len(GESTURES)})")

    # Subject-independent train/test split (last 3 users for test)
    test_uids = np.unique(full_dataset.user_ids)[-3:]
    train_mask = ~np.isin(full_dataset.user_ids, test_uids)
    test_mask = np.isin(full_dataset.user_ids, test_uids)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)} ({N_USERS-3} users) | Test samples: {len(test_dataset)} (3 unseen users)")

    # 5-Fold Cross-Validation on training set
    print("\n[2/4] Running 5-Fold Stratified Cross-Validation...")
    train_labels = full_dataset.labels[train_indices]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    cv_scores = []

    for fold, (f_tr_idx, f_va_idx) in enumerate(skf.split(train_indices, train_labels), 1):
        tr_sub = torch.utils.data.Subset(train_dataset, f_tr_idx)
        va_sub = torch.utils.data.Subset(train_dataset, f_va_idx)

        tr_loader = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_sub, batch_size=BATCH_SIZE, shuffle=False)

        fold_model = CNN_BiLSTM().to(DEVICE)
        _, _, best_acc = train_model(fold_model, tr_loader, va_loader, DEVICE, epochs=30)
        cv_scores.append(best_acc)
        print(f"  Fold {fold} Best Val Acc: {best_acc*100:.2f}%")

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"\n5-Fold CV Mean Accuracy: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

    # Train final model on full training set
    print("\n[3/4] Training final CNN-BiLSTM model on entire training cohort...")
    tr_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = CNN_BiLSTM().to(DEVICE)
    print(f"Trainable Parameters: {final_model.count_parameters():,}")

    start_time = time.time()
    final_model, history, _ = train_model(
        final_model, tr_loader_full, test_loader, DEVICE, epochs=EPOCHS
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")

    # Evaluate on unseen test subjects
    print("\n[4/4] Evaluating on unseen test cohort...")
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
    torch.save(final_model.state_dict(), 'bilstm_model.pth')
    print("\nSaved weights: bilstm_model.pth")

    meta = {
        'architecture': 'CNN-BiLSTM (Spatial-Temporal Conv-Recurrent Network)',
        'components': {
            'cnn_frontend': '3-layer 1D Conv (k=7, 5, 3)',
            'recurrent_core': '2-layer Bidirectional LSTM (hidden=64)',
            'attention': 'Temporal Self-Attention Pooling',
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

    with open('bilstm_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata: bilstm_meta.json")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#dc2626', lw=2)
    axes[0].set_title('CNN-BiLSTM Loss Curves', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', color='#2563eb', lw=2)
    axes[1].plot(history['val_acc'], label='Val Acc', color='#16a34a', lw=2)
    axes[1].set_title('CNN-BiLSTM Accuracy Curves', fontweight='bold')
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
    plt.savefig('bilstm_results.png', dpi=300)
    plt.close()
    print("Saved evaluation plot: bilstm_results.png")

    return final_model, meta


if __name__ == '__main__':
    run_pipeline()
