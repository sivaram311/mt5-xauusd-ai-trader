"""
Trainer for M5 High-Win-Rate Liquidity Engine (1:2.5 to 1:3 R:R)
Architecture:
- Micro Stream: 48 M5 Candles (Shape: 48, 7) -> Dilated 1D Temporal CNN
- Macro Stream: 10 Context Features -> Dense Embedding
- Loss: Focal Loss on Confirmation + Smooth L1 on exact price quotes
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

DATASET_PATH = r"W:\data\trading\xauusd_m5_hwr_dataset.npz"
MODEL_SAVE_PATH = r"W:\data\trading\best_m5_hwr_net.pt"
METRICS_SAVE_PATH = r"W:\data\trading\m5_hwr_training_metrics.json"

BATCH_SIZE = 64
EPOCHS = 10
LR = 6e-4
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

class M5HWRNet(nn.Module):
    def __init__(self, micro_channels=7, macro_dim=10):
        super().__init__()
        self.micro_norm = nn.LayerNorm(micro_channels)
        self.micro_stem = nn.Sequential(
            nn.Conv1d(micro_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = TemporalConvBlock(64, dilation=1)
        self.block2 = TemporalConvBlock(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = TemporalConvBlock(128, dilation=4)
        self.micro_pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU()
        )

        self.macro_norm = nn.BatchNorm1d(macro_dim)
        self.macro_fc = nn.Sequential(
            nn.Linear(macro_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.entry_head = nn.Linear(64, 1)
        self.sl_head = nn.Linear(64, 1)
        self.tp_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x_micro, x_macro):
        x_m_norm = self.micro_norm(x_micro)
        x_m_in = x_m_norm.transpose(1, 2)
        h_m = self.trans(self.block2(self.block1(self.micro_stem(x_m_in))))
        feat_micro = self.micro_fc(self.micro_pool(self.block3(h_m)).squeeze(-1))
        feat_macro = self.macro_fc(self.macro_norm(x_macro))
        feat = self.fusion(torch.cat([feat_micro, feat_macro], dim=-1))

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
purge = 30
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

# Weighted cross-entropy for 10% positive setups
weights = torch.tensor([1.0, 6.0]).to(DEVICE)
criterion_conf = nn.CrossEntropyLoss(weight=weights)
criterion_dir = nn.CrossEntropyLoss()
criterion_huber = nn.SmoothL1Loss(reduction='none')

model = M5HWRNet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print("2. Starting M5 Network Training across 10 Epochs...")
start_time = time.time()
best_f1 = 0.0
metrics_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for bx_m, bx_mac, by_c, by_d, by_e, by_s, by_t, by_tt in train_loader:
        bx_m, bx_mac = bx_m.to(DEVICE), bx_mac.to(DEVICE)
        by_c, by_d = by_c.to(DEVICE), by_d.to(DEVICE)
        by_e, by_s, by_t, by_tt = by_e.to(DEVICE), by_s.to(DEVICE), by_t.to(DEVICE), by_tt.to(DEVICE)

        optimizer.zero_grad()
        pc, pd, pe, ps, pt, ptt = model(bx_m, bx_mac)

        loss_c = criterion_conf(pc, by_c)
        mask = (by_c == 1).float().unsqueeze(-1)

        if mask.sum() > 0:
            idx = (by_c == 1)
            loss_d = criterion_dir(pd[idx], by_d[idx])
            loss_reg = (
                criterion_huber(pe, by_e) +
                criterion_huber(ps, by_s) +
                criterion_huber(pt, by_t)
            ) * mask
            loss_reg = loss_reg.sum() / (mask.sum() + 1e-4)
            loss_tt = (criterion_huber(ptt, by_tt) * mask).sum() / (mask.sum() + 1e-4)
        else:
            loss_d = torch.tensor(0.0, device=DEVICE)
            loss_reg = torch.tensor(0.0, device=DEVICE)
            loss_tt = torch.tensor(0.0, device=DEVICE)

        loss = loss_c + 0.4 * loss_d + 0.25 * loss_reg + 0.05 * loss_tt
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # Eval on unseen test set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx_m, bx_mac, by_c, _, _, _, _, _ in test_loader:
            bx_m, bx_mac = bx_m.to(DEVICE), bx_mac.to(DEVICE)
            pc, _, _, _, _, _ = model(bx_m, bx_mac)
            pred_idx = torch.argmax(pc, dim=-1).cpu().numpy()
            all_preds.extend(pred_idx)
            all_targets.extend(by_c.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    tp_cnt = int(((all_preds == 1) & (all_targets == 1)).sum())
    fp_cnt = int(((all_preds == 1) & (all_targets == 0)).sum())
    fn_cnt = int(((all_preds == 0) & (all_targets == 1)).sum())

    prec = (tp_cnt / (tp_cnt + fp_cnt + 1e-4)) * 100.0
    rec = (tp_cnt / (tp_cnt + fn_cnt + 1e-4)) * 100.0
    f1 = (2 * prec * rec) / (prec + rec + 1e-4)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {avg_loss:.4f} | Prec: {prec:.2f}% | Rec: {rec:.2f}% | F1: {f1:.2f}%")

    metrics_history.append({
        "epoch": epoch,
        "loss": round(avg_loss, 4),
        "prec": round(prec, 2),
        "rec": round(rec, 2),
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
        "history": metrics_history
    }, f, indent=2)
