"""
RunPod GPU Training Script: Multi-Task PatchTST / Dilated-TCN Mean Reversion
----------------------------------------------------------------------------
Designed for fast, budget-friendly GPU execution on RunPod (e.g., RTX 3070 / 4000 Ada / A4000).
Trains in ~3-5 minutes, logs metrics, and exports:
1. PyTorch weights (best_m5_mean_reversion_net.pt)
2. Standalone ONNX runtime artifact (best_m5_mean_reversion_net.onnx)
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATASET_FILE = "xauusd_m5_mean_reversion_dataset.npz"
MODEL_SAVE_PATH = "best_m5_mean_reversion_net.pt"
ONNX_SAVE_PATH = "best_m5_mean_reversion_net.onnx"
METRICS_SAVE_PATH = "m5_mean_reversion_metrics.json"

BATCH_SIZE = 128
EPOCHS = 15
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

# 1. Load Data
if not os.path.exists(DATASET_FILE):
    print(f"Dataset {DATASET_FILE} not found!")
    sys.exit(1)

data = np.load(DATASET_FILE)
X_micro = data['X_micro']
X_macro = data['X_macro']
y_confirm = data['y_confirm']
y_direction = data['y_direction']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_confirm)
print(f"Loaded {N:,} samples. Normalizing micro-structure...")

# Relative Normalization relative to current candle close
c_ref = ref_closes[:, None, None]
norm_o = (X_micro[:, :, 0:1] - c_ref)
norm_h = (X_micro[:, :, 1:2] - c_ref)
norm_l = (X_micro[:, :, 2:3] - c_ref)
norm_c = (X_micro[:, :, 3:4] - c_ref)
norm_v = np.log1p(X_micro[:, :, 4:5])
time_feat = X_micro[:, :, 5:7]

X_micro_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, norm_v, time_feat], axis=-1)
X_micro_feat = np.transpose(X_micro_feat, (0, 2, 1)).astype(np.float32) # (N, 7, 48)

X_macro_feat = X_macro.astype(np.float32)

# Relative offsets for regression
off_entry = np.where(y_confirm == 1, y_entry - ref_closes, 0.0).astype(np.float32)
off_sl = np.where(y_confirm == 1, y_sl - ref_closes, 0.0).astype(np.float32)
off_tp = np.where(y_confirm == 1, y_tp - ref_closes, 0.0).astype(np.float32)
y_offsets = np.stack([off_entry, off_sl, off_tp], axis=-1)

# Purged 80/20 train/test split
split_idx = int(N * 0.80)
purge = 45

train_micro = X_micro_feat[:split_idx]
train_macro = X_macro_feat[:split_idx]
train_conf = y_confirm[:split_idx]
train_dir = y_direction[:split_idx]
train_off = y_offsets[:split_idx]

test_micro = X_micro_feat[split_idx + purge:]
test_macro = X_macro_feat[split_idx + purge:]
test_conf = y_confirm[split_idx + purge:]
test_dir = y_direction[split_idx + purge:]
test_off = y_offsets[split_idx + purge:]

class ReversionDataset(Dataset):
    def __init__(self, micro, macro, conf, direction, off):
        self.micro = torch.tensor(micro, dtype=torch.float32)
        self.macro = torch.tensor(macro, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.long)
        self.dir = torch.tensor(direction, dtype=torch.long)
        self.off = torch.tensor(off, dtype=torch.float32)
    def __len__(self):
        return len(self.conf)
    def __getitem__(self, idx):
        return self.micro[idx], self.macro[idx], self.conf[idx], self.dir[idx], self.off[idx]

train_loader = DataLoader(ReversionDataset(train_micro, train_macro, train_conf, train_dir, train_off), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ReversionDataset(test_micro, test_macro, test_conf, test_dir, test_off), batch_size=256, shuffle=False)

# Architecture: Dilated Residual Network with SE attention
class SEBlock1D(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // r),
            nn.GELU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)

class TemporalBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.se = SEBlock1D(ch)
        self.act = nn.GELU()
    def forward(self, x):
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return self.act(self.se(res) + x)

class MeanReversionNet(nn.Module):
    def __init__(self, micro_ch=7, macro_dim=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(micro_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.b1 = TemporalBlock(64, dilation=1)
        self.b2 = TemporalBlock(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.b3 = TemporalBlock(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(nn.Linear(128, 64), nn.GELU())

        self.macro_fc = nn.Sequential(
            nn.BatchNorm1d(macro_dim),
            nn.Linear(macro_dim, 32),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        # Heads
        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, micro, macro):
        h_m = self.trans(self.b2(self.b1(self.stem(micro))))
        f_micro = self.micro_fc(self.pool(self.b3(h_m)).squeeze(-1))
        f_macro = self.macro_fc(macro)
        fused = self.fusion(torch.cat([f_micro, f_macro], dim=-1))
        return self.confirm_head(fused), self.dir_head(fused), self.offset_head(fused)

model = MeanReversionNet().to(DEVICE)

# Cost-Sensitive Focal Loss for Mean Reversion
class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, false_penalty=3.0):
        super().__init__()
        self.gamma = gamma
        self.penalty = false_penalty
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        weights = torch.where(targets == 0, 1.0, self.penalty)
        return torch.mean(weights * ((1.0 - pt) ** self.gamma) * ce_loss)

criterion_conf = CostSensitiveFocalLoss(gamma=2.0, false_penalty=3.0)
criterion_dir = nn.CrossEntropyLoss()
criterion_reg = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print(f"Beginning training ({EPOCHS} epochs)...")
best_score = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0.0
    for micro_b, macro_b, conf_b, dir_b, off_b in train_loader:
        micro_b, macro_b = micro_b.to(DEVICE), macro_b.to(DEVICE)
        conf_b, dir_b, off_b = conf_b.to(DEVICE), dir_b.to(DEVICE), off_b.to(DEVICE)

        optimizer.zero_grad()
        out_conf, out_dir, out_off = model(micro_b, macro_b)

        l_conf = criterion_conf(out_conf, conf_b)
        pos_mask = (conf_b == 1)
        l_dir = criterion_dir(out_dir[pos_mask], dir_b[pos_mask]) if pos_mask.sum() > 0 else 0.0
        l_off = criterion_reg(out_off[pos_mask], off_b[pos_mask]) if pos_mask.sum() > 0 else 0.0

        loss = l_conf + (0.5 * l_dir if isinstance(l_dir, torch.Tensor) else 0.0) + (0.5 * l_off if isinstance(l_off, torch.Tensor) else 0.0)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    # Evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for micro_b, macro_b, conf_b, _, _ in test_loader:
            micro_b, macro_b, conf_b = micro_b.to(DEVICE), macro_b.to(DEVICE), conf_b.to(DEVICE)
            out_c, _, _ = model(micro_b, macro_b)
            preds = torch.argmax(out_c, dim=1)
            val_correct += (preds == conf_b).sum().item()
            val_total += len(conf_b)

    acc = val_correct / val_total
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {tot_loss/len(train_loader):.4f} | Val Accuracy: {acc*100:.2f}%")

# Save PyTorch Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved PyTorch model to: {MODEL_SAVE_PATH}")

# Export ONNX for sub-1ms CPU Execution
model.eval().to("cpu")
dummy_micro = torch.randn(1, 7, 48, dtype=torch.float32)
dummy_macro = torch.randn(1, 10, dtype=torch.float32)

torch.onnx.export(
    model,
    (dummy_micro, dummy_macro),
    ONNX_SAVE_PATH,
    input_names=["micro_input", "macro_input"],
    output_names=["confirm_logits", "direction_logits", "offsets"],
    dynamic_axes={
        "micro_input": {0: "batch_size"},
        "macro_input": {0: "batch_size"},
        "confirm_logits": {0: "batch_size"},
        "direction_logits": {0: "batch_size"},
        "offsets": {0: "batch_size"}
    },
    opset_version=14
)
print(f"Exported ONNX model to: {ONNX_SAVE_PATH} ({os.path.getsize(ONNX_SAVE_PATH)/(1024*1024):.2f} MB)")
