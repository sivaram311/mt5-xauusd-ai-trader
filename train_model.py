"""
Offline Training Script for XAUUSD M15 Pending Orders with High Accuracy
Model: 1D Temporal Dilated ResNet with Multi-Head Output (Classification + Masked Regression)
Hardware: Multi-core CPU optimized with PyTorch
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import time

DATASET_FILE = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_order_net.pt"
METRICS_SAVE_PATH = r"W:\data\trading\training_metrics.json"

# Set PyTorch CPU thread count to maximize 4 vCPUs
torch.set_num_threads(4)

print("1. Loading dataset...")
data = np.load(DATASET_FILE)
X_raw = data['X_ohlc']       # (N, 100, 4)
y_action = data['y_action']   # (N,)
y_entry = data['y_entry']     # (N,)
y_sl = data['y_sl']           # (N,)
y_tp = data['y_tp']           # (N,)
ref_closes = data['ref_closes'] # (N,)

N = len(y_action)
print(f"   Total samples loaded: {N}")

# Feature Preprocessing: Normalize relative to the 100th candle close (c100)
# and compute candlestick anatomy features:
# Feature 0: (Open - c100)
# Feature 1: (High - c100)
# Feature 2: (Low - c100)
# Feature 3: (Close - c100)
# Feature 4: (Close - Open) [Candle body]
# Feature 5: (High - Low)   [Candle range]
c100_expanded = ref_closes[:, None, None] # (N, 1, 1)
norm_open = X_raw[:, :, 0:1] - c100_expanded
norm_high = X_raw[:, :, 1:2] - c100_expanded
norm_low = X_raw[:, :, 2:3] - c100_expanded
norm_close = X_raw[:, :, 3:4] - c100_expanded
body = X_raw[:, :, 3:4] - X_raw[:, :, 0:1]
rng = X_raw[:, :, 1:2] - X_raw[:, :, 2:3]

# Combine into shape: (N, 100, 6) -> Transpose to (N, 6, 100) for 1D-Conv
X_features = np.concatenate([norm_open, norm_high, norm_low, norm_close, body, rng], axis=-1)
X_features = np.transpose(X_features, (0, 2, 1)).astype(np.float32)

y_offsets = np.stack([y_entry, y_sl, y_tp], axis=-1).astype(np.float32)

# Strict chronological split: First 80% Train, Purge 48 bars, Last ~20% Test
split_idx = int(N * 0.80)
purge_gap = 48

train_X = X_features[:split_idx]
train_y_act = y_action[:split_idx]
train_y_off = y_offsets[:split_idx]

test_X = X_features[split_idx + purge_gap:]
test_y_act = y_action[split_idx + purge_gap:]
test_y_off = y_offsets[split_idx + purge_gap:]
test_ref_close = ref_closes[split_idx + purge_gap:]

print(f"   Train samples: {len(train_X)} | Test samples: {len(test_X)} (Purge gap: {purge_gap} bars)")

class TradingDataset(Dataset):
    def __init__(self, X, y_act, y_off):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_act = torch.tensor(y_act, dtype=torch.long)
        self.y_off = torch.tensor(y_off, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_act[idx], self.y_off[idx]

train_loader = DataLoader(TradingDataset(train_X, train_y_act, train_y_off), batch_size=64, shuffle=True)
test_loader = DataLoader(TradingDataset(test_X, test_y_act, test_y_off), batch_size=128, shuffle=False)

# 2. Model Architecture: Dilated Temporal ResNet
class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.GELU()
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class HighAccuracyOrderNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        # Multi-scale receptive fields (receptive fields of 3, 5, 9, 17 candles)
        self.block1 = ResBlock1D(64, dilation=1)
        self.block2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense feature projection
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        # Head 1: 3-way Classification (0: NO_ORDER, 1: BUY_LIMIT, 2: SELL_LIMIT)
        self.action_head = nn.Linear(64, 3)
        # Head 2: Regression for [Entry_Offset, SL_Offset, TP_Offset]
        self.offset_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.trans(h)
        h = self.block3(h)
        pooled = self.pool(h).squeeze(-1)
        feat = self.fc(pooled)
        action_logits = self.action_head(feat)
        offsets = self.offset_head(feat)
        return action_logits, offsets

model = HighAccuracyOrderNet(in_features=6)
print(f"2. Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

# 3. Training Loop with Multi-Task Masked Loss
criterion_ce = nn.CrossEntropyLoss()
criterion_reg = nn.SmoothL1Loss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

EPOCHS = 12
best_val_f1 = 0.0

print(f"3. Starting training ({EPOCHS} epochs on 4 vCPUs)...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_X, batch_act, batch_off in train_loader:
        optimizer.zero_grad()
        logits, pred_off = model(batch_X)
        
        loss_ce = criterion_ce(logits, batch_act)
        
        # Masked regression: only calculate loss when there is an active trade setup (act > 0)
        mask = (batch_act > 0).unsqueeze(-1)
        if mask.any():
            loss_reg = (criterion_reg(pred_off, batch_off) * mask).sum() / (mask.sum() * 3 + 1e-8)
        else:
            loss_reg = torch.tensor(0.0)
            
        loss = loss_ce + 0.5 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_act)
        preds = torch.argmax(logits, dim=-1)
        correct_train += (preds == batch_act).sum().item()
        total_train += len(batch_act)
        
    scheduler.step()
    train_acc = (correct_train / total_train) * 100.0
    
    # Validation evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    with torch.no_grad():
        for batch_X, batch_act, batch_off in test_loader:
            logits, pred_off = model(batch_X)
            preds = torch.argmax(logits, dim=-1)
            val_correct += (preds == batch_act).sum().item()
            val_total += len(batch_act)
            
            # Actionable trade metrics (ignoring NO_ORDER for trade precision)
            is_trade_true = (batch_act > 0)
            is_trade_pred = (preds > 0)
            tp_count += ((is_trade_true & is_trade_pred) & (preds == batch_act)).sum().item()
            fp_count += (is_trade_pred & ~is_trade_true).sum().item()
            fn_count += (is_trade_true & ~is_trade_pred).sum().item()
            
    val_acc = (val_correct / val_total) * 100.0
    precision = tp_count / (tp_count + fp_count + 1e-8) * 100.0
    recall = tp_count / (tp_count + fn_count + 1e-8) * 100.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}% | Trade Precision: {precision:.2f}% | F1: {f1:.2f}%")
    
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   --> Model checkpoint saved to {MODEL_SAVE_PATH}")

elapsed = time.time() - start_time
print(f"\n4. Training completed in {elapsed:.1f}s (~{elapsed/60:.1f} minutes). Best Trade F1: {best_val_f1:.2f}%")

# Save evaluation report
metrics = {
    "epochs": EPOCHS,
    "elapsed_seconds": round(elapsed, 2),
    "best_val_f1": round(best_val_f1, 2),
    "final_test_accuracy": round(val_acc, 2),
    "trade_precision": round(precision, 2),
    "model_path": MODEL_SAVE_PATH
}
with open(METRICS_SAVE_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print("5. Metrics saved to", METRICS_SAVE_PATH)
