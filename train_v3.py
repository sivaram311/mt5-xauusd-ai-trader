"""
Training Script V3: High-Accuracy Focal Loss on Liquidity Anchored Dataset
Features:
- Normalized 100-candle relative anatomy (6 channels)
- Focal Loss (gamma=2.0) with Class Weighting (Trade penalty factor = 3.5x)
- Cosine Annealing Learning Rate
- Saves best model to W:\data\trading\best_order_net_v3.pt
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time

DATASET_FILE = r"W:\data\trading\xauusd_m15_v3_high_acc_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_order_net_v3.pt"
METRICS_SAVE_PATH = r"W:\data\trading\training_metrics_v3.json"

torch.set_num_threads(4)

print("1. Loading V3 dataset...")
data = np.load(DATASET_FILE)
X_raw = data['X_m15']        # (N, 100, 4)
y_action = data['y_action']  # (N,)
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_action)
print(f"   Loaded {N} samples.")

# Preprocessing
c100 = ref_closes[:, None, None]
norm_o = X_raw[:, :, 0:1] - c100
norm_h = X_raw[:, :, 1:2] - c100
norm_l = X_raw[:, :, 2:3] - c100
norm_c = X_raw[:, :, 3:4] - c100
body = X_raw[:, :, 3:4] - X_raw[:, :, 0:1]
rng = X_raw[:, :, 1:2] - X_raw[:, :, 2:3]

X_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng], axis=-1)
X_feat = np.transpose(X_feat, (0, 2, 1)).astype(np.float32) # (N, 6, 100)
y_offsets = np.stack([y_entry, y_sl, y_tp], axis=-1).astype(np.float32)

split_idx = int(N * 0.80)
purge = 32

train_X = X_feat[:split_idx]
train_act = y_action[:split_idx]
train_off = y_offsets[:split_idx]

test_X = X_feat[split_idx + purge:]
test_act = y_action[split_idx + purge:]
test_off = y_offsets[split_idx + purge:]

print(f"   Train samples: {len(train_X)} | Test samples: {len(test_X)}")

class V3Dataset(Dataset):
    def __init__(self, X, act, off):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.act = torch.tensor(act, dtype=torch.long)
        self.off = torch.tensor(off, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.act[idx], self.off[idx]

train_loader = DataLoader(V3Dataset(train_X, train_act, train_off), batch_size=64, shuffle=True)
test_loader = DataLoader(V3Dataset(test_X, test_act, test_off), batch_size=128, shuffle=False)

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # Tensor of weights per class
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=alpha)
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Model Architecture: 1D Temporal Dilated ResNet V3
class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.GELU()
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))) + x

class HighAccuracyOrderNetV3(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.b1 = ResBlock1D(64, dilation=1)
        self.b2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.b3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.trans(h)
        h = self.b3(h)
        pooled = self.pool(h).squeeze(-1)
        feat = self.fc(pooled)
        return self.action_head(feat), self.offset_head(feat)

model = HighAccuracyOrderNetV3(in_features=6)
print(f"2. Initialized V3 Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Class weights: NO_ORDER has weight 1.0, BUY_LIMIT and SELL_LIMIT have weight 15.0 to balance rarity
class_weights = torch.tensor([1.0, 15.0, 15.0], dtype=torch.float32)
criterion_focal = FocalLoss(alpha=class_weights, gamma=2.0)
criterion_reg = nn.SmoothL1Loss(reduction='none')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

EPOCHS = 12
best_precision = 0.0

print(f"3. Starting V3 Focal Loss Training ({EPOCHS} epochs)...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    correct_train, total_train = 0, 0
    for bx, bact, boff in train_loader:
        optimizer.zero_grad()
        logits, pred_off = model(bx)
        loss_cls = criterion_focal(logits, bact)

        mask = (bact > 0).unsqueeze(-1)
        if mask.any():
            loss_reg = (criterion_reg(pred_off, boff) * mask).sum() / (mask.sum() * 3 + 1e-8)
        else:
            loss_reg = torch.tensor(0.0)

        loss = loss_cls + 0.5 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        correct_train += (preds == bact).sum().item()
        total_train += len(bact)

    scheduler.step()
    train_acc = (correct_train / total_train) * 100.0

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    tp_count, fp_count = 0, 0

    with torch.no_grad():
        for bx, bact, boff in test_loader:
            logits, pred_off = model(bx)
            preds = torch.argmax(logits, dim=-1)
            val_correct += (preds == bact).sum().item()
            val_total += len(bact)

            is_trade_true = (bact > 0)
            is_trade_pred = (preds > 0)
            tp_count += ((is_trade_true & is_trade_pred) & (preds == bact)).sum().item()
            fp_count += (is_trade_pred & ~is_trade_true).sum().item()

    val_acc = (val_correct / val_total) * 100.0
    precision = (tp_count / (tp_count + fp_count + 1e-8)) * 100.0

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Trade Precision: {precision:.2f}% (TP: {tp_count}, FP: {fp_count})")

    if precision > best_precision:
        best_precision = precision
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   --> Saved Best V3 Model: {MODEL_SAVE_PATH} (Trade Precision: {precision:.2f}%)")

elapsed = time.time() - start_time
print(f"\n4. V3 Training finished in {elapsed:.1f}s (~{elapsed/60:.1f} mins). Best Trade Precision: {best_precision:.2f}%")

metrics = {
    "version": "3.0.0",
    "architecture": "HighAccuracyOrderNetV3_FocalLoss",
    "epochs": EPOCHS,
    "elapsed_seconds": round(elapsed, 2),
    "best_trade_precision": round(best_precision, 2),
    "final_test_accuracy": round(val_acc, 2),
    "model_path": MODEL_SAVE_PATH
}
with open(METRICS_SAVE_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print("5. Metrics saved to", METRICS_SAVE_PATH)
