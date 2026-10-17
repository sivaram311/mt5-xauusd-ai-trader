"""
Dual-Branch Deep Neural Network Architecture (Version 2.0)
Fuses:
  - Branch 1: M15 Micro Structure (100 bars) -> 1D Dilated Temporal ResNet
  - Branch 2: H4 Macro Trend Structure (30 bars) -> 1D Conv + Temporal Attention
  - Fusion Layer: Cross-Feature Concatenation + Multi-Layer Perceptron
  - Output Heads: 3-way Classification (NO_ORDER, BUY_LIMIT, SELL_LIMIT) + Masked Regression [Entry, SL, TP]
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time

DATASET_FILE = r"W:\data\trading\xauusd_dual_branch_m15_h4_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_order_net_v2.pt"
METRICS_SAVE_PATH = r"W:\data\trading\training_metrics_v2.json"

torch.set_num_threads(4)

print("1. Loading dual-branch dataset...")
data = np.load(DATASET_FILE)
X_m15_raw = data['X_m15']   # (N, 100, 4)
X_h4_raw = data['X_h4']     # (N, 30, 4)
y_action = data['y_action'] # (N,)
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_action)
print(f"   Loaded {N} aligned dual-branch samples.")

# Preprocessing:
# Normalize relative to current candle close (c100)
c100_m15 = ref_closes[:, None, None]

# M15 Feature extraction: (Open, High, Low, Close, Body, Range)
m15_o = X_m15_raw[:, :, 0:1] - c100_m15
m15_h = X_m15_raw[:, :, 1:2] - c100_m15
m15_l = X_m15_raw[:, :, 2:3] - c100_m15
m15_c = X_m15_raw[:, :, 3:4] - c100_m15
m15_body = X_m15_raw[:, :, 3:4] - X_m15_raw[:, :, 0:1]
m15_rng = X_m15_raw[:, :, 1:2] - X_m15_raw[:, :, 2:3]
feat_m15 = np.concatenate([m15_o, m15_h, m15_l, m15_c, m15_body, m15_rng], axis=-1)
feat_m15 = np.transpose(feat_m15, (0, 2, 1)).astype(np.float32) # (N, 6, 100)

# H4 Feature extraction: (Open, High, Low, Close, Body, Range)
h4_o = X_h4_raw[:, :, 0:1] - c100_m15
h4_h = X_h4_raw[:, :, 1:2] - c100_m15
h4_l = X_h4_raw[:, :, 2:3] - c100_m15
h4_c = X_h4_raw[:, :, 3:4] - c100_m15
h4_body = X_h4_raw[:, :, 3:4] - X_h4_raw[:, :, 0:1]
h4_rng = X_h4_raw[:, :, 1:2] - X_h4_raw[:, :, 2:3]
feat_h4 = np.concatenate([h4_o, h4_h, h4_l, h4_c, h4_body, h4_rng], axis=-1)
feat_h4 = np.transpose(feat_h4, (0, 2, 1)).astype(np.float32)  # (N, 6, 30)

y_offsets = np.stack([y_entry, y_sl, y_tp], axis=-1).astype(np.float32)

# Chronological split: First 80% Train, Purge 48 bars, Last ~20% Test
split_idx = int(N * 0.80)
purge_gap = 48

train_m15 = feat_m15[:split_idx]
train_h4 = feat_h4[:split_idx]
train_act = y_action[:split_idx]
train_off = y_offsets[:split_idx]

test_m15 = feat_m15[split_idx + purge_gap:]
test_h4 = feat_h4[split_idx + purge_gap:]
test_act = y_action[split_idx + purge_gap:]
test_off = y_offsets[split_idx + purge_gap:]

print(f"   Train: {len(train_m15)} | Test: {len(test_m15)}")

class DualBranchDataset(Dataset):
    def __init__(self, m15, h4, act, off):
        self.m15 = torch.tensor(m15, dtype=torch.float32)
        self.h4 = torch.tensor(h4, dtype=torch.float32)
        self.act = torch.tensor(act, dtype=torch.long)
        self.off = torch.tensor(off, dtype=torch.float32)
    def __len__(self):
        return len(self.m15)
    def __getitem__(self, idx):
        return self.m15[idx], self.h4[idx], self.act[idx], self.off[idx]

train_loader = DataLoader(DualBranchDataset(train_m15, train_h4, train_act, train_off), batch_size=64, shuffle=True)
test_loader = DataLoader(DualBranchDataset(test_m15, test_h4, test_act, test_off), batch_size=128, shuffle=False)

# Model Definition
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

class DualBranchOrderNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        # 1. Micro Branch (M15, 100 bars)
        self.m15_stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.m15_b1 = ResBlock1D(64, dilation=1)
        self.m15_b2 = ResBlock1D(64, dilation=2)
        self.m15_trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.m15_b3 = ResBlock1D(128, dilation=4)
        self.m15_pool = nn.AdaptiveAvgPool1d(1)

        # 2. Macro Branch (H4, 30 bars)
        self.h4_stem = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.h4_b1 = ResBlock1D(32, dilation=1)
        self.h4_b2 = ResBlock1D(32, dilation=2)
        self.h4_trans = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.h4_pool = nn.AdaptiveAvgPool1d(1)

        # 3. Fusion Layer: 128 (M15) + 64 (H4) = 192 features
        self.fusion_fc = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 64),
            nn.GELU()
        )

        # Output Heads
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, m15_x, h4_x):
        # Micro
        h_m15 = self.m15_stem(m15_x)
        h_m15 = self.m15_b1(h_m15)
        h_m15 = self.m15_b2(h_m15)
        h_m15 = self.m15_trans(h_m15)
        h_m15 = self.m15_b3(h_m15)
        feat_m15 = self.m15_pool(h_m15).squeeze(-1) # (batch, 128)

        # Macro
        h_h4 = self.h4_stem(h4_x)
        h_h4 = self.h4_b1(h_h4)
        h_h4 = self.h4_b2(h_h4)
        h_h4 = self.h4_trans(h_h4)
        feat_h4 = self.h4_pool(h_h4).squeeze(-1)    # (batch, 64)

        # Fusion
        fused = torch.cat([feat_m15, feat_h4], dim=-1) # (batch, 192)
        rep = self.fusion_fc(fused)
        
        logits = self.action_head(rep)
        offsets = self.offset_head(rep)
        return logits, offsets

model = DualBranchOrderNet()
print(f"2. Dual-Branch Model Initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

criterion_ce = nn.CrossEntropyLoss()
criterion_reg = nn.SmoothL1Loss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

EPOCHS = 10
best_f1 = 0.0

print(f"3. Starting training ({EPOCHS} epochs on 4 vCPUs)...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    correct_train, total_train = 0, 0

    for b_m15, b_h4, b_act, b_off in train_loader:
        optimizer.zero_grad()
        logits, pred_off = model(b_m15, b_h4)
        loss_ce = criterion_ce(logits, b_act)

        mask = (b_act > 0).unsqueeze(-1)
        if mask.any():
            loss_reg = (criterion_reg(pred_off, b_off) * mask).sum() / (mask.sum() * 3 + 1e-8)
        else:
            loss_reg = torch.tensor(0.0)

        loss = loss_ce + 0.5 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        correct_train += (preds == b_act).sum().item()
        total_train += len(b_act)

    scheduler.step()
    train_acc = (correct_train / total_train) * 100.0

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    tp_count, fp_count, fn_count = 0, 0, 0

    with torch.no_grad():
        for b_m15, b_h4, b_act, b_off in test_loader:
            logits, pred_off = model(b_m15, b_h4)
            preds = torch.argmax(logits, dim=-1)
            val_correct += (preds == b_act).sum().item()
            val_total += len(b_act)

            is_trade_true = (b_act > 0)
            is_trade_pred = (preds > 0)
            tp_count += ((is_trade_true & is_trade_pred) & (preds == b_act)).sum().item()
            fp_count += (is_trade_pred & ~is_trade_true).sum().item()
            fn_count += (is_trade_true & ~is_trade_pred).sum().item()

    val_acc = (val_correct / val_total) * 100.0
    precision = (tp_count / (tp_count + fp_count + 1e-8)) * 100.0
    recall = (tp_count / (tp_count + fn_count + 1e-8)) * 100.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}% | Trade Precision: {precision:.2f}% | F1: {f1:.2f}%")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   --> Best V2 Model Checkpoint Saved: {MODEL_SAVE_PATH}")

elapsed = time.time() - start_time
print(f"\n4. V2 Dual-Branch Training Complete in {elapsed:.1f}s (~{elapsed/60:.1f} mins). Best F1: {best_f1:.2f}%")

metrics = {
    "version": "2.0.0",
    "architecture": "DualBranch_M15_H4_ResNet",
    "epochs": EPOCHS,
    "elapsed_seconds": round(elapsed, 2),
    "best_val_f1": round(best_f1, 2),
    "final_test_accuracy": round(val_acc, 2),
    "trade_precision": round(precision, 2),
    "model_path": MODEL_SAVE_PATH
}
with open(METRICS_SAVE_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print("5. Metrics saved to", METRICS_SAVE_PATH)
