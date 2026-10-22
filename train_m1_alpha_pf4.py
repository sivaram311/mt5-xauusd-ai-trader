"""
Dual-Stream High-Alpha PF 3.0-4.0 Architecture & Trainer
Architecture:
- Micro Stream: Conv1d Dilated Temporal Network on (60, 5) raw M1 bars
- Macro Stream: Dense Embedding on (12) Multi-Timeframe Structural & Session Features
- Fusion: Deep Multi-Task Head with Asymmetric Focal Loss + Heavy False-Positive Penalty
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

DATASET_PATH = r"W:\data\trading\xauusd_alpha_pf4_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_alpha_pf4_net.pt"
METRICS_SAVE_PATH = r"W:\data\trading\m1_alpha_pf4_training_metrics.json"

BATCH_SIZE = 64
EPOCHS = 12
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

class TemporalConvBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + x)

class DualStreamAlphaNet(nn.Module):
    def __init__(self, micro_channels=5, macro_dim=12):
        super().__init__()
        # Micro Branch (60 M1 candles)
        self.micro_norm = nn.LayerNorm(micro_channels)
        self.micro_stem = nn.Sequential(
            nn.Conv1d(micro_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU()
        )
        self.block1 = TemporalConvBlock(48, dilation=1)
        self.block2 = TemporalConvBlock(48, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=1),
            nn.BatchNorm1d(96),
            nn.GELU()
        )
        self.block3 = TemporalConvBlock(96, dilation=4)
        self.micro_pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU()
        )

        # Macro Branch (12 context features)
        self.macro_norm = nn.BatchNorm1d(macro_dim)
        self.macro_fc = nn.Sequential(
            nn.Linear(macro_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(48 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.20)
        )

        # Multi-Head Outputs
        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.entry_head = nn.Linear(64, 1)
        self.sl_head = nn.Linear(64, 1)
        self.tp_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x_micro, x_macro):
        # Micro branch
        x_m_norm = self.micro_norm(x_micro)
        x_m_in = x_m_norm.transpose(1, 2) # (B, 5, 60)
        h_m = self.trans(self.block2(self.block1(self.micro_stem(x_m_in))))
        feat_micro = self.micro_fc(self.micro_pool(self.block3(h_m)).squeeze(-1))

        # Macro branch
        feat_macro = self.macro_fc(self.macro_norm(x_macro))

        # Fusion
        feat = self.fusion(torch.cat([feat_micro, feat_macro], dim=-1))

        # Price anchoring
        last_close = x_micro[:, -1, 3].unsqueeze(-1)

        p_conf = self.confirm_head(feat)
        p_dir = self.dir_head(feat)
        p_ent = last_close + self.entry_head(feat)
        p_sl = last_close + self.sl_head(feat)
        p_tp = last_close + self.tp_head(feat)
        p_ttf = torch.relu(self.ttf_head(feat)) + 1.0

        return p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf

print("1. Loading dataset...")
data = np.load(DATASET_PATH)
X_micro = data['X_micro']
X_macro = data['X_macro']
y_confirm = data['y_confirm']
y_direction = data['y_direction']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
y_ttf = data['y_ttf']

N = len(y_confirm)
split_idx = int(N * 0.80)
purge = 60
test_start = split_idx + purge

train_dataset = TensorDataset(
    torch.tensor(X_micro[:split_idx], dtype=torch.float32),
    torch.tensor(X_macro[:split_idx], dtype=torch.float32),
    torch.tensor(y_confirm[:split_idx], dtype=torch.long),
    torch.tensor(y_direction[:split_idx], dtype=torch.long),
    torch.tensor(y_entry[:split_idx], dtype=torch.float32).unsqueeze(-1),
    torch.tensor(y_sl[:split_idx], dtype=torch.float32).unsqueeze(-1),
    torch.tensor(y_tp[:split_idx], dtype=torch.float32).unsqueeze(-1),
    torch.tensor(y_ttf[:split_idx], dtype=torch.float32).unsqueeze(-1)
)

test_dataset = TensorDataset(
    torch.tensor(X_micro[test_start:], dtype=torch.float32),
    torch.tensor(X_macro[test_start:], dtype=torch.float32),
    torch.tensor(y_confirm[test_start:], dtype=torch.long),
    torch.tensor(y_direction[test_start:], dtype=torch.long),
    torch.tensor(y_entry[test_start:], dtype=torch.float32).unsqueeze(-1),
    torch.tensor(y_sl[test_start:], dtype=torch.float32).unsqueeze(-1),
    torch.tensor(y_tp[test_start:], dtype=torch.float32).unsqueeze(-1),
    torch.tensor(y_ttf[test_start:], dtype=torch.float32).unsqueeze(-1)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"   Train samples: {len(train_dataset):,} | Test samples: {len(test_dataset):,}")

# Weighted Cross Entropy for high selectivity
weights = torch.tensor([1.0, 50.0]).to(DEVICE)
criterion_conf = nn.CrossEntropyLoss(weight=weights)
criterion_dir = nn.CrossEntropyLoss()
criterion_huber = nn.SmoothL1Loss(reduction='none')

model = DualStreamAlphaNet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print("2. Training Dual-Stream Network...")
best_loss = 9999.0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for bx_micro, bx_macro, by_conf, by_dir, by_ent, by_sl, by_tp, by_ttf in train_loader:
        bx_micro = bx_micro.to(DEVICE)
        bx_macro = bx_macro.to(DEVICE)
        by_conf = by_conf.to(DEVICE)
        by_dir = by_dir.to(DEVICE)
        by_ent = by_ent.to(DEVICE)
        by_sl = by_sl.to(DEVICE)
        by_tp = by_tp.to(DEVICE)
        by_ttf = by_ttf.to(DEVICE)

        optimizer.zero_grad()
        p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf = model(bx_micro, bx_macro)

        loss_conf = criterion_conf(p_conf, by_conf)
        mask = (by_conf == 1).float().unsqueeze(-1)

        if mask.sum() > 0:
            idx = (by_conf == 1)
            loss_dir = criterion_dir(p_dir[idx], by_dir[idx])
            loss_reg = (
                criterion_huber(p_ent, by_ent) +
                criterion_huber(p_sl, by_sl) +
                criterion_huber(p_tp, by_tp)
            ) * mask
            loss_reg = loss_reg.sum() / (mask.sum() + 1e-4)
            loss_ttf = (criterion_huber(p_ttf, by_ttf) * mask).sum() / (mask.sum() + 1e-4)
        else:
            loss_dir = torch.tensor(0.0, device=DEVICE)
            loss_reg = torch.tensor(0.0, device=DEVICE)
            loss_ttf = torch.tensor(0.0, device=DEVICE)

        loss = loss_conf + 0.5 * loss_dir + 0.3 * loss_reg + 0.1 * loss_ttf
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Eval
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx_m, bx_mac, by_c, by_d, by_e, by_s, by_t, by_tt in test_loader:
            bx_m, bx_mac = bx_m.to(DEVICE), bx_mac.to(DEVICE)
            by_c, by_d = by_c.to(DEVICE), by_d.to(DEVICE)
            by_e, by_s, by_t, by_tt = by_e.to(DEVICE), by_s.to(DEVICE), by_t.to(DEVICE), by_tt.to(DEVICE)

            pc, pd, pe, ps, pt, ptt = model(bx_m, bx_mac)
            l_c = criterion_conf(pc, by_c)
            val_loss += l_c.item()

    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    history.append({
        "epoch": epoch,
        "train_loss": round(avg_train_loss, 4),
        "val_loss": round(avg_val_loss, 4)
    })

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val_loss
        }, MODEL_SAVE_PATH)
        print(f"   --> Saved new best model checkpoint to {MODEL_SAVE_PATH}")

with open(METRICS_SAVE_PATH, "w") as f:
    json.dump({"history": history}, f, indent=2)

print("Training successfully finished!")
