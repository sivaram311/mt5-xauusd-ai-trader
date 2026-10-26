"""
RunPod GPU Training Script for 4-to-10 Strategy (v1.5 Series)
------------------------------------------------------------
- Architecture: 1D Temporal Dilated ResNet with Multi-Head outputs.
- Inputs: Exact 6 pure-OHLC relative channels across 100 bars:
    * norm_o = Open - c100
    * norm_h = High - c100
    * norm_l = Low  - c100
    * norm_c = Close - c100
    * body   = Close - Open
    * range  = High  - Low
- Parameters:
    * SL: Static $4.00 (Anchored logical stop)
    * TP: Extended $10.00 Target (1:2.50 payoff)
- Training:
    * Mixed Precision AMP (FP16) on CUDA
    * Batch Size: 512
    * Epochs: 60 with Cosine Annealing Learning Rate
    * Cost-sensitive Focal/Cross-Entropy weighting for high precision on 4-to-10 targets.
- Output artifacts:
    * best_order_net_v1_5_4to10.pt
    * best_order_net_v1_5_4to10.onnx
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

DATASET_FILE = "/workspace/xauusd_m15_v1_5_4to10_dataset.npz"
MODEL_SAVE_PATH = "/workspace/best_order_net_v1_5_4to10.pt"
ONNX_SAVE_PATH = "/workspace/best_order_net_v1_5_4to10.onnx"

BATCH_SIZE = 512
EPOCHS = 60
LR = 1.5e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"=== FULL GPU UTILIZATION TRAINING: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ===")

if not os.path.exists(DATASET_FILE):
    print(f"Error: {DATASET_FILE} not found!")
    sys.exit(1)

data = np.load(DATASET_FILE)
X_ohlc = data['X_ohlc']       # (98853, 100, 4) pure OHLC
y_action = data['y_action']   # (98853,)
y_entry = data['y_entry']     # (98853,)
y_sl = data['y_sl']           # (98853,)
y_tp = data['y_tp']           # (98853,)
ref_closes = data['ref_closes']

N = len(y_action)
print(f"Loaded {N:,} pure M15 OHLC samples. Normalizing (6, 100) input tensor...")

c100 = ref_closes[:, None, None]
norm_o = X_ohlc[:, :, 0:1] - c100
norm_h = X_ohlc[:, :, 1:2] - c100
norm_l = X_ohlc[:, :, 2:3] - c100
norm_c = X_ohlc[:, :, 3:4] - c100
body = X_ohlc[:, :, 3:4] - X_ohlc[:, :, 0:1]
rng = X_ohlc[:, :, 1:2] - X_ohlc[:, :, 2:3]

X_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng], axis=-1)
X_feat = np.transpose(X_feat, (0, 2, 1)).astype(np.float32) # (N, 6, 100)

off_entry = np.where(y_action > 0, y_entry - ref_closes, 0.0).astype(np.float32)
off_sl = np.where(y_action > 0, y_sl - ref_closes, 0.0).astype(np.float32)
off_tp = np.where(y_action > 0, y_tp - ref_closes, 0.0).astype(np.float32)
y_offsets = np.stack([off_entry, off_sl, off_tp], axis=-1)

# Chronological split: 80% train, 48-bar purge, 20% test (19,700 candles ~10 months)
split = int(N * 0.80)
purge = 48

X_train, y_act_train, y_off_train = X_feat[:split], y_action[:split], y_offsets[:split]
X_test, y_act_test, y_off_test = X_feat[split+purge:], y_action[split+purge:], y_offsets[split+purge:]

print(f"Train samples: {len(y_act_train):,}, Test samples: {len(y_act_test):,}")
print(f"Train Action Counts: {np.bincount(y_act_train)}")
print(f"Test Action Counts:  {np.bincount(y_act_test)}")

class XAUUSDDataset(Dataset):
    def __init__(self, X, y_act, y_off):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_act = torch.tensor(y_act, dtype=torch.long)
        self.y_off = torch.tensor(y_off, dtype=torch.float32)

    def __len__(self):
        return len(self.y_act)

    def __getitem__(self, idx):
        return self.X[idx], self.y_act[idx], self.y_off[idx]

train_dataset = XAUUSDDataset(X_train, y_act_train, y_off_train)
test_dataset = XAUUSDDataset(X_test, y_act_test, y_off_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.Mish()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class BestOrderNetV1_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.Mish()
        )
        self.layer1 = ResidualBlock1D(64, dilation=1)
        self.layer2 = ResidualBlock1D(64, dilation=2)
        self.layer3 = ResidualBlock1D(64, dilation=4)
        self.layer4 = ResidualBlock1D(64, dilation=8)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Action Classifier Head (0: HOLD, 1: BUY_LIMIT, 2: SELL_LIMIT)
        self.action_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Dropout(0.25),
            nn.Linear(64, 3)
        )
        
        # Offset Regressor Head (Entry_off, SL_off, TP_off)
        self.offset_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Dropout(0.20),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        feat = self.pool(h).squeeze(-1)
        
        action_logits = self.action_head(feat)
        offsets = self.offset_head(feat)
        return action_logits, offsets

model = BestOrderNetV1_5().to(DEVICE)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Weighted CrossEntropy to balance rare 4-to-10 opportunities
class_weights = torch.tensor([1.0, 15.0, 15.0], dtype=torch.float32).to(DEVICE)
criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
criterion_reg = nn.SmoothL1Loss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_val_score = -1e9

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for x_b, act_b, off_b in train_loader:
        x_b = x_b.to(DEVICE, non_blocking=True)
        act_b = act_b.to(DEVICE, non_blocking=True)
        off_b = off_b.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred_act, pred_off = model(x_b)
            loss_cls = criterion_cls(pred_act, act_b)
            
            mask = (act_b > 0)
            if mask.sum() > 0:
                loss_reg = criterion_reg(pred_off[mask], off_b[mask])
            else:
                loss_reg = torch.tensor(0.0, device=DEVICE)
            
            loss = loss_cls + 2.0 * loss_reg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * len(act_b)
        pred_cls = pred_act.argmax(dim=-1)
        correct += (pred_cls == act_b).sum().item()
        total += len(act_b)

    scheduler.step()
    train_loss /= total
    train_acc = (correct / total) * 100

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_buys_pred = 0
    val_sells_pred = 0
    val_active_true = 0
    val_active_correct = 0

    with torch.no_grad():
        for x_b, act_b, off_b in test_loader:
            x_b = x_b.to(DEVICE, non_blocking=True)
            act_b = act_b.to(DEVICE, non_blocking=True)
            off_b = off_b.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_act, pred_off = model(x_b)
                loss_cls = criterion_cls(pred_act, act_b)
                mask = (act_b > 0)
                loss_reg = criterion_reg(pred_off[mask], off_b[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=DEVICE)
                loss = loss_cls + 2.0 * loss_reg

            val_loss += loss.item() * len(act_b)
            pred_cls = pred_act.argmax(dim=-1)
            val_correct += (pred_cls == act_b).sum().item()
            val_total += len(act_b)
            
            val_buys_pred += (pred_cls == 1).sum().item()
            val_sells_pred += (pred_cls == 2).sum().item()
            
            active_mask = (act_b > 0)
            val_active_true += active_mask.sum().item()
            val_active_correct += ((pred_cls == act_b) & active_mask).sum().item()

    val_loss /= val_total
    val_acc = (val_correct / val_total) * 100
    val_recall = (val_active_correct / max(1, val_active_true)) * 100
    val_score = -val_loss + 0.1 * val_recall

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:2d}/{EPOCHS} | TrLoss: {train_loss:.4f} Acc: {train_acc:.2f}% | ValLoss: {val_loss:.4f} Acc: {val_acc:.2f}% | Pred: B={val_buys_pred} S={val_sells_pred} | Active Recall: {val_recall:.2f}%")

    if val_score > best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"Optimal model saved to {MODEL_SAVE_PATH}")

# Export ONNX
print(f"Exporting to ONNX: {ONNX_SAVE_PATH}...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

dummy_input = torch.randn(1, 6, 100, device=DEVICE)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_SAVE_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['action_logits', 'offsets'],
    dynamic_axes={'input': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}, 'offsets': {0: 'batch_size'}}
)

print(f"Exported successfully! Size: {os.path.getsize(ONNX_SAVE_PATH)/(1024*1024):.2f} MB")
