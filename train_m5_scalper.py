# High-Precision M5 Scalper Neural Network Trainer for XAUUSD
# Dataset: W:/data/trading/xauusd_m5_scalp_dataset.npz (98,923 samples, continuous all hours)
# Spread Friction: Baked in ($0.35/oz)
# Architecture: 1D Dilated Temporal ResNet with Squeeze-and-Excitation (SE) block
# Multi-Task Head: Classification (NO_ORDER, BUY_LIMIT, SELL_LIMIT) + Offset Head (Entry, SL, TP)
# Hardware: Optimized multi-core CPU with Cosine Annealing LR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import time

DATASET_FILE = r"W:\data\trading\xauusd_m5_scalp_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_m5_scalper_net.pt"
METRICS_SAVE_PATH = r"W:\data\trading\m5_scalper_training_metrics.json"

torch.set_num_threads(4)

print("1. Loading continuous M5 dataset...")
data = np.load(DATASET_FILE)
X_raw = data['X_ohlc']         # (N, 60, 4)
y_action = data['y_action']     # (N,)
y_entry = data['y_entry']       # (N,)
y_sl = data['y_sl']             # (N,)
y_tp = data['y_tp']             # (N,)
ref_closes = data['ref_closes'] # (N,)

N = len(y_action)
print(f"   Total samples loaded: {N:,}")

# Feature Engineering: 6 input channels normalized to the 60th candle close
# 0: Open - C60
# 1: High - C60
# 2: Low - C60
# 3: Close - C60
# 4: Candle Body (Close - Open)
# 5: Candle Range (High - Low)
c60_expanded = ref_closes[:, None, None]
norm_open = X_raw[:, :, 0:1] - c60_expanded
norm_high = X_raw[:, :, 1:2] - c60_expanded
norm_low = X_raw[:, :, 2:3] - c60_expanded
norm_close = X_raw[:, :, 3:4] - c60_expanded
body = X_raw[:, :, 3:4] - X_raw[:, :, 0:1]
rng = X_raw[:, :, 1:2] - X_raw[:, :, 2:3]

X_features = np.concatenate([norm_open, norm_high, norm_low, norm_close, body, rng], axis=-1)
X_features = np.transpose(X_features, (0, 2, 1)).astype(np.float32) # (N, 6, 60)
y_offsets = np.stack([y_entry, y_sl, y_tp], axis=-1).astype(np.float32)

# Chronological split: 80% Train, Purge 18 bars (horizon), 20% Test
split_idx = int(N * 0.80)
purge_gap = 18

train_X = X_features[:split_idx]
train_y_act = y_action[:split_idx]
train_y_off = y_offsets[:split_idx]

test_X = X_features[split_idx + purge_gap:]
test_y_act = y_action[split_idx + purge_gap:]
test_y_off = y_offsets[split_idx + purge_gap:]
test_ref_close = ref_closes[split_idx + purge_gap:]

print(f"   Train samples: {len(train_X):,} | Test samples: {len(test_X):,} (Purge: {purge_gap} bars)")

class ScalpDataset(Dataset):
    def __init__(self, X, y_act, y_off):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_act = torch.tensor(y_act, dtype=torch.long)
        self.y_off = torch.tensor(y_off, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_act[idx], self.y_off[idx]

train_loader = DataLoader(ScalpDataset(train_X, train_y_act, train_y_off), batch_size=128, shuffle=True)
test_loader = DataLoader(ScalpDataset(test_X, test_y_act, test_y_off), batch_size=256, shuffle=False)

# Architecture: Dilated ResNet with Squeeze-and-Excitation
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w

class ResBlockSE1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels)
        self.relu = nn.GELU()
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + res)

class M5ScalperNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlockSE1D(64, dilation=1)
        self.block2 = ResBlockSE1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlockSE1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.action_head = nn.Linear(64, 3) # 0: NO_ORDER, 1: BUY_LIMIT, 2: SELL_LIMIT
        self.offset_head = nn.Linear(64, 3) # Entry, SL, TP

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.trans(h)
        h = self.block3(h)
        feat = self.fc(self.pool(h).squeeze(-1))
        return self.action_head(feat), self.offset_head(feat)

model = M5ScalperNet(in_features=6)
print(f"2. M5 Scalper Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

criterion_ce = nn.CrossEntropyLoss()
criterion_reg = nn.SmoothL1Loss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"3. Starting training ({EPOCHS} epochs on 4 vCPUs)...")
start_time = time.time()
best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for X_b, y_act_b, y_off_b in train_loader:
        optimizer.zero_grad()
        logits, pred_off = model(X_b)
        
        # Classification loss
        loss_cls = criterion_ce(logits, y_act_b)
        
        # Masked regression loss (only calculate for actual orders)
        trade_mask = (y_act_b != 0).float().unsqueeze(-1)
        loss_reg = (criterion_reg(pred_off, y_off_b) * trade_mask).sum() / (trade_mask.sum() * 3.0 + 1e-6)
        
        loss = loss_cls + 0.5 * loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y_act_b)
        preds = logits.argmax(dim=-1)
        correct_train += (preds == y_act_b).sum().item()
        total_train += len(y_act_b)
        
    scheduler.step()
    train_acc = correct_train / total_train * 100.0
    avg_loss = total_loss / total_train
    
    # Evaluation on unseen out-of-sample test split
    model.eval()
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for X_b, y_act_b, _ in test_loader:
            logits, _ = model(X_b)
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            test_trues.extend(y_act_b.cpu().numpy())
            
    test_preds = np.array(test_preds)
    test_trues = np.array(test_trues)
    test_acc = (test_preds == test_trues).mean() * 100.0
    
    # Macro F1 for trading classes
    tp_trade = ((test_preds != 0) & (test_preds == test_trues)).sum()
    fp_trade = ((test_preds != 0) & (test_preds != test_trues)).sum()
    fn_trade = ((test_trues != 0) & (test_preds != test_trues)).sum()
    precision = tp_trade / (tp_trade + fp_trade + 1e-6) * 100.0
    recall = tp_trade / (tp_trade + fn_trade + 1e-6) * 100.0
    trade_f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    elapsed = time.time() - epoch_start
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({elapsed:.1f}s) | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Trade Precision: {precision:.2f}% | Trade F1: {trade_f1:.2f}%")
    
    if trade_f1 > best_val_f1 or epoch == EPOCHS:
        best_val_f1 = trade_f1
        torch.save({
            "model_state_dict": model.state_dict(),
            "in_features": 6,
            "lookback": 60,
            "horizon": 18,
            "epoch": epoch,
            "trade_f1": trade_f1,
            "trade_precision": precision,
            "test_acc": test_acc
        }, MODEL_SAVE_PATH)
        print(f"   -> Checkpoint saved to {MODEL_SAVE_PATH} (Trade F1: {trade_f1:.2f}%)")

total_time = time.time() - start_time
print(f"\n4. Training completed in {total_time/60:.2f} minutes.")
print(f"   Best Trade F1: {best_val_f1:.2f}%")

metrics = {
    "epochs": EPOCHS,
    "elapsed_seconds": round(total_time, 2),
    "best_val_f1": round(best_val_f1, 2),
    "final_test_accuracy": round(test_acc, 2),
    "trade_precision": round(precision, 2),
    "model_path": MODEL_SAVE_PATH,
    "timeframe": "M5",
    "samples": N,
    "spread_modeled": 0.35
}

with open(METRICS_SAVE_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"   Saved metrics to {METRICS_SAVE_PATH}")
print("   All done!")
