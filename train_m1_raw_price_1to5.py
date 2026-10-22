"""
Architecture & Training Script for Raw Price 1:5 Asymmetric M1 Liquidity Sniper
Inputs:
- X_raw: (Batch, 120, 7) containing raw [Open, High, Low, Close, Minute, Hour, DayOfWeek]
Outputs:
1. confirm_logit: Binary setup confirmation (0 = Stay flat, 1 = Valid 1:5 setup)
2. dir_logit: Binary direction (0 = BUY_LIMIT, 1 = SELL_LIMIT)
3. entry_pred: Raw predicted entry price
4. sl_pred: Raw predicted tight stop loss price
5. tp_pred: Raw predicted take profit price (>= 5.0x Risk)
6. ttf_pred: Predicted time to fill in minutes (1 to 25 mins)

Loss Function:
- Focal Loss on Confirmation (handles 82% no-trade majority vs 18% setups)
- CrossEntropy on Direction
- Smooth L1 (Huber) Loss on raw price levels (Entry, SL, TP) only when confirm == 1
- Smooth L1 Loss on TimeToFill only when confirm == 1
"""
import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DATASET_PATH = r"W:\data\trading\xauusd_m1_raw_price_1to5_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_m1_raw_price_1to5_net.pt"
METRICS_SAVE_PATH = r"W:\data\trading\m1_raw_price_1to5_training_metrics.json"

BATCH_SIZE = 128
EPOCHS = 10
LR = 7e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Neural Architecture
class RawTemporalBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)

class RawM1SniperNet(nn.Module):
    def __init__(self, in_features=7):
        super().__init__()
        # Input LayerNorm ensures gradient stability with raw 4000+ dollar prices
        self.input_norm = nn.LayerNorm(in_features)
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = RawTemporalBlock(64, dilation=1)
        self.block2 = RawTemporalBlock(64, dilation=2)
        
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = RawTemporalBlock(128, dilation=4)
        self.block4 = RawTemporalBlock(128, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Multi-Head Outputs
        self.confirm_head = nn.Linear(64, 2)    # 0 = No Trade, 1 = Setup
        self.dir_head = nn.Linear(64, 2)        # 0 = BUY_LIMIT, 1 = SELL_LIMIT
        self.entry_head = nn.Linear(64, 1)      # Raw Entry Price
        self.sl_head = nn.Linear(64, 1)         # Raw SL Price
        self.tp_head = nn.Linear(64, 1)         # Raw TP Price
        self.ttf_head = nn.Linear(64, 1)        # Time to fill in minutes

    def forward(self, x_raw):
        # x_raw: (B, 120, 7)
        # Apply layer normalization across features
        x_norm = self.input_norm(x_raw) # (B, 120, 7)
        x_in = x_norm.transpose(1, 2)   # (B, 7, 120) for Conv1d

        h = self.trans(self.block2(self.block1(self.stem(x_in))))
        feat = self.fc(self.pool(self.block4(self.block3(h))).squeeze(-1))

        # Anchor raw price predictions relative to the last bar's raw close
        last_close = x_raw[:, -1, 3].unsqueeze(-1) # (B, 1)

        confirm_logits = self.confirm_head(feat)
        dir_logits = self.dir_head(feat)
        entry_pred = last_close + self.entry_head(feat)
        sl_pred = last_close + self.sl_head(feat)
        tp_pred = last_close + self.tp_head(feat)
        ttf_pred = torch.relu(self.ttf_head(feat)) + 1.0 # Min 1.0 minute

        return confirm_logits, dir_logits, entry_pred, sl_pred, tp_pred, ttf_pred

# 2. Data Loading & Splitting
print("1. Loading raw dataset...")
data = np.load(DATASET_PATH)
X_raw = data['X_raw']
y_confirm = data['y_confirm']
y_direction = data['y_direction']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
y_time_to_fill = data['y_time_to_fill']

N = len(y_confirm)
split_idx = int(N * 0.80)
purge = 60 # Gap between train and test
test_start = split_idx + purge

train_X = torch.tensor(X_raw[:split_idx], dtype=torch.float32)
train_y_conf = torch.tensor(y_confirm[:split_idx], dtype=torch.long)
train_y_dir = torch.tensor(y_direction[:split_idx], dtype=torch.long)
train_y_entry = torch.tensor(y_entry[:split_idx], dtype=torch.float32).unsqueeze(-1)
train_y_sl = torch.tensor(y_sl[:split_idx], dtype=torch.float32).unsqueeze(-1)
train_y_tp = torch.tensor(y_tp[:split_idx], dtype=torch.float32).unsqueeze(-1)
train_y_ttf = torch.tensor(y_time_to_fill[:split_idx], dtype=torch.float32).unsqueeze(-1)

test_X = torch.tensor(X_raw[test_start:], dtype=torch.float32)
test_y_conf = torch.tensor(y_confirm[test_start:], dtype=torch.long)
test_y_dir = torch.tensor(y_direction[test_start:], dtype=torch.long)
test_y_entry = torch.tensor(y_entry[test_start:], dtype=torch.float32).unsqueeze(-1)
test_y_sl = torch.tensor(y_sl[test_start:], dtype=torch.float32).unsqueeze(-1)
test_y_tp = torch.tensor(y_tp[test_start:], dtype=torch.float32).unsqueeze(-1)
test_y_ttf = torch.tensor(y_time_to_fill[test_start:], dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(train_X, train_y_conf, train_y_dir, train_y_entry, train_y_sl, train_y_tp, train_y_ttf)
test_dataset = TensorDataset(test_X, test_y_conf, test_y_dir, test_y_entry, test_y_sl, test_y_tp, test_y_ttf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"   Train samples: {len(train_dataset):,} | Test samples: {len(test_dataset):,}")

# 3. Model, Loss, Optimizer
model = RawM1SniperNet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Focal Loss for class imbalance (18% positive setups)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        return (alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss).mean()

criterion_conf = FocalLoss(alpha=0.40, gamma=2.0)
criterion_dir = nn.CrossEntropyLoss()
criterion_huber = nn.SmoothL1Loss(reduction='none')

print("2. Starting Model Training across 10 Epochs...")
start_time = time.time()
best_f1 = 0.0
metrics_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for bx, by_conf, by_dir, by_ent, by_sl, by_tp, by_ttf in train_loader:
        bx = bx.to(DEVICE)
        by_conf = by_conf.to(DEVICE)
        by_dir = by_dir.to(DEVICE)
        by_ent = by_ent.to(DEVICE)
        by_sl = by_sl.to(DEVICE)
        by_tp = by_tp.to(DEVICE)
        by_ttf = by_ttf.to(DEVICE)

        optimizer.zero_grad()
        p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf = model(bx)

        loss_conf = criterion_conf(p_conf, by_conf)

        # Mask regression losses only for positive trade setups
        mask_trade = (by_conf == 1).float().unsqueeze(-1)
        has_trades = mask_trade.sum() > 0

        if has_trades:
            trade_indices = (by_conf == 1)
            loss_dir = criterion_dir(p_dir[trade_indices], by_dir[trade_indices])
            loss_ent = (criterion_huber(p_ent, by_ent) * mask_trade).sum() / (mask_trade.sum() + 1e-4)
            loss_sl = (criterion_huber(p_sl, by_sl) * mask_trade).sum() / (mask_trade.sum() + 1e-4)
            loss_tp = (criterion_huber(p_tp, by_tp) * mask_trade).sum() / (mask_trade.sum() + 1e-4)
            loss_ttf = (criterion_huber(p_ttf, by_ttf) * mask_trade).sum() / (mask_trade.sum() + 1e-4)
        else:
            loss_dir = torch.tensor(0.0, device=DEVICE)
            loss_ent = torch.tensor(0.0, device=DEVICE)
            loss_sl = torch.tensor(0.0, device=DEVICE)
            loss_tp = torch.tensor(0.0, device=DEVICE)
            loss_ttf = torch.tensor(0.0, device=DEVICE)

        loss = loss_conf + 0.3 * loss_dir + 0.2 * (loss_ent + loss_sl + loss_tp) + 0.05 * loss_ttf
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # Evaluation on unseen test set
    model.eval()
    all_preds_conf = []
    all_targets_conf = []

    with torch.no_grad():
        for bx, by_conf, _, _, _, _, _ in test_loader:
            bx = bx.to(DEVICE)
            p_conf, _, _, _, _, _ = model(bx)
            pred_labels = torch.argmax(p_conf, dim=-1).cpu().numpy()
            all_preds_conf.extend(pred_labels)
            all_targets_conf.extend(by_conf.numpy())

    all_preds_conf = np.array(all_preds_conf)
    all_targets_conf = np.array(all_targets_conf)

    tp_cnt = int(((all_preds_conf == 1) & (all_targets_conf == 1)).sum())
    fp_cnt = int(((all_preds_conf == 1) & (all_targets_conf == 0)).sum())
    fn_cnt = int(((all_preds_conf == 0) & (all_targets_conf == 1)).sum())

    precision = (tp_cnt / (tp_cnt + fp_cnt + 1e-4)) * 100.0
    recall = (tp_cnt / (tp_cnt + fn_cnt + 1e-4)) * 100.0
    f1 = (2 * precision * recall) / (precision + recall + 1e-4)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {total_loss/len(train_loader):.4f} | Trade Prec: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")

    metrics_history.append({
        "epoch": epoch,
        "loss": round(total_loss / len(train_loader), 4),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2)
    })

    if f1 > best_f1:
        best_f1 = f1
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "f1": f1
        }, MODEL_SAVE_PATH)
        print(f"   --> Saved new best checkpoint (F1: {f1:.2f}%) to {MODEL_SAVE_PATH}")

elapsed = (time.time() - start_time) / 60.0
print(f"Training completed in {elapsed:.2f} minutes.")

with open(METRICS_SAVE_PATH, "w") as f:
    json.dump({
        "total_epochs": EPOCHS,
        "best_f1": round(best_f1, 2),
        "metrics_history": metrics_history
    }, f, indent=2)
