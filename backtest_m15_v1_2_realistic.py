"""
Realistic Forward Simulation Backtest for M15 v1.2 Pure OHLC Model
------------------------------------------------------------------
Runs the EXACT simulation logic from original v1.0 backtest.py:
- Evaluates across 19,723 unseen test candles (Dec 2025 to Oct 2026).
- Places pending limit orders (BUY_LIMIT below market / SELL_LIMIT above market).
- Simulates forward 48 candles to track order fill, Take Profit, and Stop Loss.
- Unfilled orders expire cleanly without cost (realistic pending order simulation).
- Minimum R:R >= 2.0 with $0.25 broker spread fee.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

DATASET_PATH = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"
MODEL_PATH = r"W:\data\trading\best_order_net_v1_2.pt"
REPORT_PATH = r"W:\data\trading\backtest_m15_v1_2_realistic_report.json"

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

print("1. Loading dataset and newly trained v1.2 model...")
data = np.load(DATASET_PATH)
X_raw = data['X_ohlc']         # (98853, 100, 4)
ref_closes = data['ref_closes']
end_times = data['end_times']

N = len(ref_closes)
split_idx = int(N * 0.80)
purge_gap = 48
test_start = split_idx + purge_gap

test_X_raw = X_raw[test_start:]
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

print(f"   Total test samples: {len(test_ref_close):,} from {test_end_times[0]} to {test_end_times[-1]}")

c100_expanded = test_ref_close[:, None, None]
norm_open = test_X_raw[:, :, 0:1] - c100_expanded
norm_high = test_X_raw[:, :, 1:2] - c100_expanded
norm_low = test_X_raw[:, :, 2:3] - c100_expanded
norm_close = test_X_raw[:, :, 3:4] - c100_expanded
body = test_X_raw[:, :, 3:4] - test_X_raw[:, :, 0:1]
rng = test_X_raw[:, :, 1:2] - test_X_raw[:, :, 2:3]

X_feat = np.concatenate([norm_open, norm_high, norm_low, norm_close, body, rng], axis=-1)
X_feat = np.transpose(X_feat, (0, 2, 1)).astype(np.float32)

model = HighAccuracyOrderNet(in_features=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

print("2. Running batch model inference...")
t0 = time.time()
with torch.no_grad():
    tensor_X = torch.tensor(X_feat, dtype=torch.float32)
    batch_size = 512
    all_logits = []
    all_offsets = []
    for i in range(0, len(tensor_X), batch_size):
        bx = tensor_X[i:i+batch_size]
        logits, offsets = model(bx)
        all_logits.append(logits.numpy())
        all_offsets.append(offsets.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_offsets = np.concatenate(all_offsets, axis=0)

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=-1, keepdims=True)
pred_actions = np.argmax(probs, axis=-1)
pred_confs = np.max(probs, axis=-1)

# Exact original simulation parameters
EXPIRY_BARS = 48        # 12 hours forward expiry
MIN_RR_RATIO = 2.0      # Minimum 1:2 R:R
SPREAD_POINTS = 0.25    # $0.25 spread
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE_PCT = 1.0

print("\n3. Testing across confidence tiers using real forward bar simulation...")

results = {}
for CONF_THRESHOLD in [0.40, 0.45, 0.50, 0.55]:
    balance = INITIAL_BALANCE
    wins = 0
    losses = 0
    timed_out = 0
    triggered_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    peak_equity = INITIAL_BALANCE
    max_dd = 0.0

    for i in range(len(test_ref_close) - EXPIRY_BARS):
        act = pred_actions[i]
        conf = pred_confs[i]
        c100 = test_ref_close[i]
        entry_off, sl_off, tp_off = all_offsets[i]

        if act == 0 or conf < CONF_THRESHOLD:
            continue

        entry = round(c100 + entry_off, 2)
        sl = round(c100 + sl_off, 2)
        tp = round(c100 + tp_off, 2)

        risk_pts = abs(entry - sl)
        reward_pts = abs(tp - entry)

        if risk_pts < 1.0 or risk_pts > 15.0 or (reward_pts / (risk_pts + 1e-4)) < MIN_RR_RATIO:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c100: continue
        if order_type == "SELL_LIMIT" and entry <= c100: continue

        global_idx = test_start + i
        fwd_highs = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 1]
        fwd_lows = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 2]

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
                        if fwd_highs[k] + SPREAD_POINTS >= sl: outcome = "LOSS"; break
                        if fwd_lows[k] <= tp: outcome = "WIN"; break
                else:
                    if fwd_highs[k] + SPREAD_POINTS >= sl: outcome = "LOSS"; break
                    if fwd_lows[k] <= tp: outcome = "WIN"; break

        if triggered:
            triggered_count += 1
            risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
            lot_size = risk_amount / (risk_pts * 100.0)

            if outcome == "WIN":
                profit = reward_pts * 100.0 * lot_size
                balance += profit
                gross_profit += profit
                wins += 1
            elif outcome == "LOSS":
                loss_val = (risk_pts + SPREAD_POINTS) * 100.0 * lot_size
                balance -= loss_val
                gross_loss += loss_val
                losses += 1

            if balance > peak_equity: peak_equity = balance
            dd = (peak_equity - balance) / peak_equity * 100.0
            if dd > max_dd: max_dd = dd

    wr = (wins / triggered_count * 100) if triggered_count > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    net_ret = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100.0

    print(f"Gate >= {CONF_THRESHOLD:.2f} | Filled: {triggered_count:4d} | Wins: {wins:3d} | WR: {wr:5.2f}% | PF: {pf:4.2f} | Net Return: {net_ret:+6.2f}% | Max DD: {max_dd:5.2f}%")
    results[f"gate_{CONF_THRESHOLD}"] = {
        "filled": int(triggered_count), "wins": int(wins), "losses": int(losses),
        "win_rate": float(round(wr, 2)), "profit_factor": float(round(pf, 2)),
        "net_return_pct": float(round(net_ret, 2)), "max_dd_pct": float(round(max_dd, 2))
    }

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nReport written to: {REPORT_PATH}")
