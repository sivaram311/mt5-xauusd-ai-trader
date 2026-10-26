"""
RunPod GPU Training Script for M15 v1.2 Series (Exact v1.0 Replication on GPU)
------------------------------------------------------------------------------
- Architecture: Exact 1D Temporal Dilated ResNet (in_features=6, 100 bars)
- Input: Pure OHLC relative normalization (norm_o, norm_h, norm_l, norm_c, body, range)
- Dataset: xauusd_m15_100bars_dataset.npz (98,853 pure OHLC samples, 4.2+ years)
- Hardware: Full GPU mixed precision (AMP FP16) on NVIDIA TensorCores
- Epochs: 40 Epochs with Cosine Annealing Learning Rate
- Exports:
    1. best_order_net_v1_2.pt
    2. best_order_net_v1_2.onnx
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

DATASET_FILE = "xauusd_m15_100bars_dataset.npz"
MODEL_SAVE_PATH = "best_order_net_v1_2.pt"
ONNX_SAVE_PATH = "best_order_net_v1_2.onnx"

BATCH_SIZE = 512
EPOCHS = 40
LR = 1.5e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"=== SATURATING FULL GPU COMPUTE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ===")

if not os.path.exists(DATASET_FILE):
    print(f"Error: {DATASET_FILE} not found!")
    sys.exit(1)

data = np.load(DATASET_FILE)
X_raw = data['X_ohlc']         # (98853, 100, 4) pure OHLC
y_action = data['y_action']     # (98853,)
y_entry = data['y_entry']       # (98853,)
y_sl = data['y_sl']             # (98853,)
y_tp = data['y_tp']             # (98853,)
ref_closes = data['ref_closes'] # (98853,)

N = len(y_action)
print(f"Loaded {N:,} samples from original v1.0 dataset. Computing exact (6, 100) features...")

# Exact v1.0 feature preprocessing
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

# Strict chronological split: 80% train, 48-bar purge, 20% test (matches v1.0)
split = int(N * 0.80)
purge = 48

train_x = torch.tensor(X_feat[:split], dtype=torch.float32)
train_act = torch.tensor(y_action[:split], dtype=torch.long)
train_off = torch.tensor(y_offsets[:split], dtype=torch.float32)

test_x = torch.tensor(X_feat[split + purge:], dtype=torch.float32)
test_act = torch.tensor(y_action[split + purge:], dtype=torch.long)
test_off = torch.tensor(y_offsets[split + purge:], dtype=torch.float32)

print(f"Train samples: {len(train_act):,} | Unseen Test samples: {len(test_act):,}")

class GpuDataset(Dataset):
    def __init__(self, x, act, off):
        self.x = x; self.act = act; self.off = off
    def __len__(self): return len(self.act)
    def __getitem__(self, i): return self.x[i], self.act[i], self.off[i]

train_loader = DataLoader(GpuDataset(train_x, train_act, train_off), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(GpuDataset(test_x, test_act, test_off), batch_size=1024, shuffle=False, pin_memory=True)

# Exact HighAccuracyOrderNet from v1.0
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

class HighAccuracyOrderNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlock1D(64, dilation=1)
        self.block2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.trans(self.block2(self.block1(self.stem(x))))
        pooled = self.pool(self.block3(h)).squeeze(-1)
        feat = self.fc(pooled)
        return self.action_head(feat), self.offset_head(feat)

model = HighAccuracyOrderNet(in_features=6).to(DEVICE)

criterion_act = nn.CrossEntropyLoss()
criterion_reg = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler()

print(f"Beginning {EPOCHS}-Epoch Deep Training on GPU...")
t_start = time.time()
best_loss = 999.0

for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for bx, bact, boff in train_loader:
        bx, bact, boff = bx.to(DEVICE), bact.to(DEVICE), boff.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            out_act, out_off = model(bx)
            l_act = criterion_act(out_act, bact)
            pos = (bact > 0)
            l_reg = criterion_reg(out_off[pos], boff[pos]) if pos.sum() > 0 else 0.0
            loss = l_act + (0.5 * l_reg if isinstance(l_reg, torch.Tensor) else 0.0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    if ep % 5 == 0 or ep == EPOCHS:
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, bact, _ in test_loader:
                bx, bact = bx.to(DEVICE), bact.to(DEVICE)
                out_act, _ = model(bx)
                preds = torch.argmax(out_act, dim=-1)
                val_correct += (preds == bact).sum().item()
                val_total += len(bact)
        val_acc = val_correct / val_total
        print(f"Epoch {ep:02d}/{EPOCHS:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Time: {int(time.time() - t_start)}s")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\nTraining completed in {int(time.time() - t_start)}s. Best Loss: {best_loss:.4f}")

# Export ONNX
model.eval().to("cpu")
dummy_x = torch.randn(1, 6, 100, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_x,
    ONNX_SAVE_PATH,
    input_names=["candle_input"],
    output_names=["action_logits", "offsets"],
    dynamic_axes={
        "candle_input": {0: "batch_size"},
        "action_logits": {0: "batch_size"},
        "offsets": {0: "batch_size"}
    },
    opset_version=14
)
print(f"Exported ONNX model to: {ONNX_SAVE_PATH} ({os.path.getsize(ONNX_SAVE_PATH)/(1024*1024):.2f} MB)")
