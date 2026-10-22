# Dual-Branch Hierarchical Neural Network Trainer for XAUUSD M1 Sniper V2
# Branch 1: 1D Dilated ResNet + SE Block on Micro 180 M1 Candles + Candle Color
# Branch 2: Dense MLP on Macro Pyramid (2-Week, 1-Week, Daily, H4, M15 levels + Time of Day)
# Fusion: Multi-task output head for Classification and exact Offset Regression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import time

DATASET_FILE = r"W:\data\trading\xauusd_m1_v2_hierarchical_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_m1_v2_hierarchical_net.pt"
METRICS_SAVE_PATH = r"W:\data\trading\m1_v2_training_metrics.json"

torch.set_num_threads(4)

print("1. Loading Hierarchical M1 dataset...")
data = np.load(DATASET_FILE)
X_micro = data['X_micro']       # (N, 180, 5) -> [O, H, L, C, Color]
X_macro = data['X_macro']       # (N, 11) -> [W2, W1, D1, H4, M15, Time]
y_action = data['y_action']     # (N,)
y_entry = data['y_entry']       # (N,)
y_sl = data['y_sl']             # (N,)
y_tp = data['y_tp']             # (N,)
ref_closes = data['ref_closes'] # (N,)

N = len(y_action)
print(f"   Loaded {N:,} samples. Normalizing micro-structure...")

# Micro Feature Preprocessing: Normalize prices relative to the 180th candle close (c180)
# Channels:
# 0: Open - c180
# 1: High - c180
# 2: Low - c180
# 3: Close - c180
# 4: Candle Body (Close - Open)
# 5: Candle Range (High - Low)
# 6: Candle Color (+1.0 Green, -1.0 Red, 0.0 Doji)
c180 = ref_closes[:, None, None]
norm_o = X_micro[:, :, 0:1] - c180
norm_h = X_micro[:, :, 1:2] - c180
norm_l = X_micro[:, :, 2:3] - c180
norm_c = X_micro[:, :, 3:4] - c180
body = X_micro[:, :, 3:4] - X_micro[:, :, 0:1]
rng = X_micro[:, :, 1:2] - X_micro[:, :, 2:3]
color = X_micro[:, :, 4:5]

X_micro_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng, color], axis=-1)
X_micro_feat = np.transpose(X_micro_feat, (0, 2, 1)).astype(np.float32) # (N, 7, 180)

X_macro_feat = X_macro.astype(np.float32) # (N, 11)
y_offsets = np.stack([y_entry, y_sl, y_tp], axis=-1).astype(np.float32)

# Strict chronological split: 80% Train, Purge 45 bars, 20% Test
split_idx = int(N * 0.80)
purge_gap = 45

train_micro = X_micro_feat[:split_idx]
train_macro = X_macro_feat[:split_idx]
train_y_act = y_action[:split_idx]
train_y_off = y_offsets[:split_idx]

test_micro = X_micro_feat[split_idx + purge_gap:]
test_macro = X_macro_feat[split_idx + purge_gap:]
test_y_act = y_action[split_idx + purge_gap:]
test_y_off = y_offsets[split_idx + purge_gap:]
test_ref_close = ref_closes[split_idx + purge_gap:]

print(f"   Train samples: {len(train_micro):,} | Test samples: {len(test_micro):,} (Purge: {purge_gap} bars)")

class HierarchicalDataset(Dataset):
    def __init__(self, micro, macro, y_act, y_off):
        self.micro = torch.tensor(micro, dtype=torch.float32)
        self.macro = torch.tensor(macro, dtype=torch.float32)
        self.y_act = torch.tensor(y_act, dtype=torch.long)
        self.y_off = torch.tensor(y_off, dtype=torch.float32)
    def __len__(self):
        return len(self.micro)
    def __getitem__(self, idx):
        return self.micro[idx], self.macro[idx], self.y_act[idx], self.y_off[idx]

train_loader = DataLoader(HierarchicalDataset(train_micro, train_macro, train_y_act, train_y_off), batch_size=128, shuffle=True)
test_loader = DataLoader(HierarchicalDataset(test_micro, test_macro, test_y_act, test_y_off), batch_size=256, shuffle=False)

# SE Block
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

# Dual-Branch Hierarchical Architecture
class M1HierarchicalSniperNet(nn.Module):
    def __init__(self, in_channels_micro=7, in_features_macro=11):
        super().__init__()
        # Branch 1: Micro 180 M1 feature extractor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels_micro, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlockSE1D(64, dilation=1)
        self.block2 = ResBlockSE1D(64, dilation=2)
        self.trans1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlockSE1D(128, dilation=4)
        self.block4 = ResBlockSE1D(128, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU()
        )

        # Branch 2: Macro Multi-Timeframe Level MLP
        self.macro_fc = nn.Sequential(
            nn.Linear(in_features_macro, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Fusion Layer: Combines 64 (Micro) + 32 (Macro) = 96 features
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Output Heads
        self.action_head = nn.Linear(64, 3) # 0: NO_ORDER, 1: BUY_LIMIT, 2: SELL_LIMIT
        self.offset_head = nn.Linear(64, 3) # Entry, SL, TP

    def forward(self, micro, macro):
        # Process Micro
        h_m = self.stem(micro)
        h_m = self.block1(h_m)
        h_m = self.block2(h_m)
        h_m = self.trans1(h_m)
        h_m = self.block3(h_m)
        h_m = self.block4(h_m)
        feat_micro = self.micro_fc(self.pool(h_m).squeeze(-1))

        # Process Macro
        feat_macro = self.macro_fc(macro)

        # Fuse
        fused = torch.cat([feat_micro, feat_macro], dim=-1)
        feat = self.fusion_fc(fused)

        return self.action_head(feat), self.offset_head(feat)

model = M1HierarchicalSniperNet(in_channels_micro=7, in_features_macro=11)
print(f"2. Dual-Branch Hierarchical Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

# Class imbalance weighting (NO_ORDER is ~70%)
class_counts = torch.tensor([(y_action == 0).sum(), (y_action == 1).sum(), (y_action == 2).sum()], dtype=torch.float32)
class_weights = (1.0 / (class_counts + 1e-4))
class_weights = (class_weights / class_weights.sum() * 3.0)

criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
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

    for micro_b, macro_b, y_act_b, y_off_b in train_loader:
        optimizer.zero_grad()
        logits, pred_off = model(micro_b, macro_b)

        loss_cls = criterion_ce(logits, y_act_b)

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

    # Validation
    model.eval()
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for micro_b, macro_b, y_act_b, _ in test_loader:
            logits, _ = model(micro_b, macro_b)
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            test_trues.extend(y_act_b.cpu().numpy())

    test_preds = np.array(test_preds)
    test_trues = np.array(test_trues)
    test_acc = (test_preds == test_trues).mean() * 100.0

    tp_trade = ((test_preds != 0) & (test_preds == test_trues)).sum()
    fp_trade = ((test_preds != 0) & (test_preds != test_trues)).sum()
    fn_trade = ((test_trues != 0) & (test_preds != test_trues)).sum()
    precision = tp_trade / (tp_trade + fp_trade + 1e-6) * 100.0
    recall = tp_trade / (tp_trade + fn_trade + 1e-6) * 100.0
    trade_f1 = 2 * precision * recall / (precision + recall + 1e-6)

    elapsed = time.time() - epoch_start
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({elapsed:.1f}s) | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Trade Precision: {precision:.2f}% | Trade F1: {trade_f1:.2f}%")

    if trade_f1 > best_val_f1 or epoch == EPOCHS:
        best_val_f1 = trade_f1
        torch.save({
            "model_state_dict": model.state_dict(),
            "in_channels_micro": 7,
            "in_features_macro": 11,
            "lookback_micro": 180,
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
    "architecture": "Dual-Branch Hierarchical M1 ResNet + Macro MLP",
    "samples": N,
    "spread_modeled": 0.35
}

with open(METRICS_SAVE_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"   Saved metrics to {METRICS_SAVE_PATH}")
print("   All done!")
