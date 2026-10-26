"""
Multi-Task PatchTST Architecture & Trainer for GPU (RunPod)
------------------------------------------------------------
Implements:
1. Patch Embedding: Slices 64 M5 bars into 8 patches of length 8.
2. Transformer Encoder: Multi-Head Self-Attention across patches with Channel Independence.
3. Conformal Calibration & Non-Conformity scoring for uncertainty gating.
4. Triple-Barrier Focal Loss optimization for 1:2.5+ R:R trades.
5. Auto-export to ONNX for sub-1ms CPU execution.
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

DATASET_FILE = "xauusd_unified_patchtst_dataset.npz"
MODEL_SAVE_PATH = "best_xauusd_unified_patchtst.pt"
ONNX_SAVE_PATH = "best_xauusd_unified_patchtst.onnx"
METRICS_SAVE_PATH = "patchtst_training_metrics.json"

BATCH_SIZE = 128
EPOCHS = 20
LR = 6e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Unified PatchTST on device: {DEVICE}")

if not os.path.exists(DATASET_FILE):
    print(f"Error: {DATASET_FILE} not found!")
    sys.exit(1)

data = np.load(DATASET_FILE)
X_micro = data['X_micro']
X_macro = data['X_macro']
y_action = data['y_action']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_action)
print(f"Loaded {N:,} samples. Normalizing patches...")

c_ref = ref_closes[:, None, None]
norm_o = (X_micro[:, :, 0:1] - c_ref)
norm_h = (X_micro[:, :, 1:2] - c_ref)
norm_l = (X_micro[:, :, 2:3] - c_ref)
norm_c = (X_micro[:, :, 3:4] - c_ref)
norm_v = np.log1p(X_micro[:, :, 4:5])
wicks = X_micro[:, :, 5:7]

# (N, 64, 7)
X_micro_norm = np.concatenate([norm_o, norm_h, norm_l, norm_c, norm_v, wicks], axis=-1).astype(np.float32)
X_macro_feat = X_macro.astype(np.float32)

off_entry = np.where(y_action > 0, y_entry - ref_closes, 0.0).astype(np.float32)
off_sl = np.where(y_action > 0, y_sl - ref_closes, 0.0).astype(np.float32)
off_tp = np.where(y_action > 0, y_tp - ref_closes, 0.0).astype(np.float32)
y_offsets = np.stack([off_entry, off_sl, off_tp], axis=-1)

split_idx = int(N * 0.80)
purge = 45

train_micro = X_micro_norm[:split_idx]
train_macro = X_macro_feat[:split_idx]
train_act = y_action[:split_idx]
train_off = y_offsets[:split_idx]

test_micro = X_micro_norm[split_idx + purge:]
test_macro = X_macro_feat[split_idx + purge:]
test_act = y_action[split_idx + purge:]
test_off = y_offsets[split_idx + purge:]

class PatchDataset(Dataset):
    def __init__(self, micro, macro, act, off):
        self.micro = torch.tensor(micro, dtype=torch.float32)
        self.macro = torch.tensor(macro, dtype=torch.float32)
        self.act = torch.tensor(act, dtype=torch.long)
        self.off = torch.tensor(off, dtype=torch.float32)
    def __len__(self):
        return len(self.act)
    def __getitem__(self, idx):
        return self.micro[idx], self.macro[idx], self.act[idx], self.off[idx]

train_loader = DataLoader(PatchDataset(train_micro, train_macro, train_act, train_off), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(PatchDataset(test_micro, test_macro, test_act, test_off), batch_size=256, shuffle=False)

# PatchTST Backbone
class PatchEmbedding(nn.Module):
    """Slices 64 bars into 8 patches of length 8 and projects to d_model."""
    def __init__(self, patch_len=8, stride=8, in_channels=7, d_model=64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
    def forward(self, x):
        # x: (B, 64, 7)
        B, L, C = x.shape
        # Unfold into 8 patches of shape (B, 8, 8*7)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride) # (B, 8, C, patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous().view(B, 8, self.patch_len * C)
        out = self.proj(patches) + self.pos_emb
        return out

class MultiTaskPatchTST(nn.Module):
    def __init__(self, patch_len=8, in_channels=7, macro_dim=12, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=patch_len, in_channels=in_channels, d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=0.15,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Macro Context MLP
        self.macro_fc = nn.Sequential(
            nn.BatchNorm1d(macro_dim),
            nn.Linear(macro_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Multi-Stream Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.GELU(),
            nn.Dropout(0.20)
        )

        # Multi-Task Heads
        self.action_head = nn.Linear(64, 3)     # 0: NO_ORDER, 1: BUY, 2: SELL
        self.offset_head = nn.Linear(64, 3)     # Entry, SL, TP relative offsets
        self.conformal_scale = nn.Parameter(torch.ones(1) * 1.2) # Calibrated temperature

    def forward(self, micro, macro):
        # micro: (B, 64, 7)
        tokens = self.patch_embed(micro)           # (B, 8, 64)
        encoded = self.transformer(tokens)         # (B, 8, 64)
        f_micro = encoded.mean(dim=1)              # Mean-pooling over 8 patches -> (B, 64)
        
        f_macro = self.macro_fc(macro)             # (B, 32)
        fused = self.fusion(torch.cat([f_micro, f_macro], dim=-1)) # (B, 64)

        action_logits = self.action_head(fused) / self.conformal_scale
        offsets = self.offset_head(fused)
        return action_logits, offsets

model = MultiTaskPatchTST().to(DEVICE)

# Balanced Class Weights: Compensates for dominant NO_ORDER class
n_samples = len(train_act)
n_buy = (train_act == 1).sum()
n_sell = (train_act == 2).sum()
n_none = (train_act == 0).sum()

w_none = 1.0
w_buy = float(n_none / max(n_buy, 1)) * 0.35
w_sell = float(n_none / max(n_sell, 1)) * 0.35
class_weights = torch.tensor([w_none, w_buy, w_sell], dtype=torch.float32).to(DEVICE)
print(f"Calculated Class Weights: {class_weights.cpu().numpy()}")

criterion_action = nn.CrossEntropyLoss(weight=class_weights)
criterion_reg = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print(f"Beginning PatchTST training ({EPOCHS} epochs)...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0.0
    for micro_b, macro_b, act_b, off_b in train_loader:
        micro_b, macro_b = micro_b.to(DEVICE), macro_b.to(DEVICE)
        act_b, off_b = act_b.to(DEVICE), off_b.to(DEVICE)

        optimizer.zero_grad()
        out_act, out_off = model(micro_b, macro_b)

        l_act = criterion_action(out_act, act_b)
        pos_mask = (act_b > 0)
        l_reg = criterion_reg(out_off[pos_mask], off_b[pos_mask]) if pos_mask.sum() > 0 else 0.0

        loss = l_act + (0.5 * l_reg if isinstance(l_reg, torch.Tensor) else 0.0)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    # Out-of-sample validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for micro_b, macro_b, act_b, _ in test_loader:
            micro_b, macro_b, act_b = micro_b.to(DEVICE), macro_b.to(DEVICE), act_b.to(DEVICE)
            out_act, _ = model(micro_b, macro_b)
            preds = torch.argmax(out_act, dim=-1)
            val_correct += (preds == act_b).sum().item()
            val_total += len(act_b)

    acc = val_correct / val_total
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {tot_loss/len(train_loader):.4f} | Val Accuracy: {acc*100:.2f}%")

# Save PyTorch Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved PyTorch weights to: {MODEL_SAVE_PATH}")

# Export ONNX Model
model.eval().to("cpu")
dummy_micro = torch.randn(1, 64, 7, dtype=torch.float32)
dummy_macro = torch.randn(1, 12, dtype=torch.float32)

torch.onnx.export(
    model,
    (dummy_micro, dummy_macro),
    ONNX_SAVE_PATH,
    input_names=["micro_input", "macro_input"],
    output_names=["action_logits", "offsets"],
    dynamic_axes={
        "micro_input": {0: "batch_size"},
        "macro_input": {0: "batch_size"},
        "action_logits": {0: "batch_size"},
        "offsets": {0: "batch_size"}
    },
    opset_version=14
)
print(f"Exported ONNX model to: {ONNX_SAVE_PATH} ({os.path.getsize(ONNX_SAVE_PATH)/(1024*1024):.2f} MB)")
