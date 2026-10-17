"""
Realistic Out-of-Sample Backtester for V2 Dual-Branch Model (M15 + H4)
Evaluates over the 19,723 test candles (~10 months) comparing V2 with V1.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time

MODEL_PATH = r"W:\data\trading\best_order_net_v2.pt"
DATASET_PATH = r"W:\data\trading\xauusd_dual_branch_m15_h4_dataset.npz"

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

class DualBranchOrderNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.m15_stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.m15_b1 = ResBlock1D(64, dilation=1)
        self.m15_b2 = ResBlock1D(64, dilation=2)
        self.m15_trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.m15_b3 = ResBlock1D(128, dilation=4)
        self.m15_pool = nn.AdaptiveAvgPool1d(1)

        self.h4_stem = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.h4_b1 = ResBlock1D(32, dilation=1)
        self.h4_b2 = ResBlock1D(32, dilation=2)
        self.h4_trans = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.h4_pool = nn.AdaptiveAvgPool1d(1)

        self.fusion_fc = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 64),
            nn.GELU()
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, m15_x, h4_x):
        h_m15 = self.m15_stem(m15_x)
        h_m15 = self.m15_b1(h_m15)
        h_m15 = self.m15_b2(h_m15)
        h_m15 = self.m15_trans(h_m15)
        h_m15 = self.m15_b3(h_m15)
        feat_m15 = self.m15_pool(h_m15).squeeze(-1)

        h_h4 = self.h4_stem(h4_x)
        h_h4 = self.h4_b1(h_h4)
        h_h4 = self.h4_b2(h_h4)
        h_h4 = self.h4_trans(h_h4)
        feat_h4 = self.h4_pool(h_h4).squeeze(-1)

        fused = torch.cat([feat_m15, feat_h4], dim=-1)
        rep = self.fusion_fc(fused)
        return self.action_head(rep), self.offset_head(rep)

print("1. Loading V2 dataset and model...")
data = np.load(DATASET_PATH)
X_m15_raw = data['X_m15']
X_h4_raw = data['X_h4']
ref_closes = data['ref_closes']
end_times = data['end_times']

N = len(ref_closes)
split_idx = int(N * 0.80)
purge_gap = 48
test_start = split_idx + purge_gap

test_m15_raw = X_m15_raw[test_start:]
test_h4_raw = X_h4_raw[test_start:]
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

c100_m15 = test_ref_close[:, None, None]

m15_o = test_m15_raw[:, :, 0:1] - c100_m15
m15_h = test_m15_raw[:, :, 1:2] - c100_m15
m15_l = test_m15_raw[:, :, 2:3] - c100_m15
m15_c = test_m15_raw[:, :, 3:4] - c100_m15
m15_body = test_m15_raw[:, :, 3:4] - test_m15_raw[:, :, 0:1]
m15_rng = test_m15_raw[:, :, 1:2] - test_m15_raw[:, :, 2:3]
feat_m15 = np.concatenate([m15_o, m15_h, m15_l, m15_c, m15_body, m15_rng], axis=-1)
feat_m15 = np.transpose(feat_m15, (0, 2, 1)).astype(np.float32)

h4_o = test_h4_raw[:, :, 0:1] - c100_m15
h4_h = test_h4_raw[:, :, 1:2] - c100_m15
h4_l = test_h4_raw[:, :, 2:3] - c100_m15
h4_c = test_h4_raw[:, :, 3:4] - c100_m15
h4_body = test_h4_raw[:, :, 3:4] - test_h4_raw[:, :, 0:1]
h4_rng = test_h4_raw[:, :, 1:2] - test_h4_raw[:, :, 2:3]
feat_h4 = np.concatenate([h4_o, h4_h, h4_l, h4_c, h4_body, h4_rng], axis=-1)
feat_h4 = np.transpose(feat_h4, (0, 2, 1)).astype(np.float32)

model = DualBranchOrderNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

print("2. Running batch model inference over all test sequences...")
with torch.no_grad():
    tensor_m15 = torch.tensor(feat_m15, dtype=torch.float32)
    tensor_h4 = torch.tensor(feat_h4, dtype=torch.float32)
    batch_size = 512
    all_logits = []
    all_offsets = []
    for i in range(0, len(tensor_m15), batch_size):
        bm = tensor_m15[i:i+batch_size]
        bh = tensor_h4[i:i+batch_size]
        logits, offsets = model(bm, bh)
        all_logits.append(logits.numpy())
        all_offsets.append(offsets.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_offsets = np.concatenate(all_offsets, axis=0)

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=-1, keepdims=True)
pred_actions = np.argmax(probs, axis=-1)
pred_confs = np.max(probs, axis=-1)

SPREAD = 0.25
EXPIRY = 48
TRADING_DAYS = 218

def evaluate_v2(min_conf=0.40):
    balance = 10000.0
    equity = [balance]
    trades = []

    for i in range(len(test_ref_close) - EXPIRY):
        act = pred_actions[i]
        conf = pred_confs[i]
        c100 = test_ref_close[i]
        t_int = test_end_times[i]

        if act == 0 or conf < min_conf:
            continue

        entry_off, sl_off, tp_off = all_offsets[i]
        entry = round(c100 + entry_off, 2)
        sl = round(c100 + sl_off, 2)
        tp = round(c100 + tp_off, 2)

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk < 1.0 or risk > 15.0 or (reward / (risk + 1e-4)) < 2.0:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c100: continue
        if order_type == "SELL_LIMIT" and entry <= c100: continue

        global_idx = test_start + i
        fwd_highs = X_m15_raw[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 1]
        fwd_lows = X_m15_raw[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 2]

        triggered = False
        outcome = "EXPIRED"

        if order_type == "BUY_LIMIT":
            for k in range(len(fwd_lows)):
                if not triggered:
                    if fwd_lows[k] <= entry:
                        triggered = True
                        if fwd_lows[k] <= sl: outcome = "LOSS"; break
                        if fwd_highs[k] >= tp: outcome = "WIN"; break
                else:
                    if fwd_lows[k] <= sl: outcome = "LOSS"; break
                    if fwd_highs[k] >= tp: outcome = "WIN"; break

        elif order_type == "SELL_LIMIT":
            for k in range(len(fwd_highs)):
                if not triggered:
                    if fwd_highs[k] >= entry:
                        triggered = True
                        if fwd_highs[k] + SPREAD >= sl: outcome = "LOSS"; break
                        if fwd_lows[k] <= tp: outcome = "WIN"; break
                else:
                    if fwd_highs[k] + SPREAD >= sl: outcome = "LOSS"; break
                    if fwd_lows[k] <= tp: outcome = "WIN"; break

        if triggered and outcome in ["WIN", "LOSS"]:
            risk_amt = balance * 0.01
            lot_size = risk_amt / (risk * 100.0)
            pnl = (reward * 100.0 * lot_size) if outcome == "WIN" else (-risk * 100.0 * lot_size)
            balance += pnl
            equity.append(balance)
            trades.append({"outcome": outcome, "rr": reward / risk, "pnl": pnl})

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    avg_rr = np.mean([t["rr"] for t in trades]) if trades else 0.0
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = (gross_win / (gross_loss + 1e-4)) if trades else 0.0

    peak = 10000.0
    max_dd = 0.0
    for b in equity:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100.0
        if dd > max_dd: max_dd = dd

    entries_per_day = len(trades) / TRADING_DAYS

    return {
        "Trades": len(trades),
        "Trades / Day": round(entries_per_day, 1),
        "Win Rate": f"{win_rate:.1f}%",
        "Average R:R": f"1:{avg_rr:.2f}",
        "Profit Factor": round(float(pf), 2),
        "Max Drawdown": f"{max_dd:.1f}%",
        "Final Balance": f"${balance:,.2f}"
    }

print("\n--- V2 Model Performance Across Confidence Gates ---")
res_list = []
for c in [0.40, 0.45, 0.50, 0.55]:
    r = evaluate_v2(min_conf=c)
    res_list.append({"Confidence": f">={c:.2f}", **r})

import pandas as pd
df_res = pd.DataFrame(res_list)
print(df_res.to_string(index=False))

with open(r"W:\data\trading\v2_backtest_report.json", "w") as f:
    json.dump(res_list, f, indent=2)
