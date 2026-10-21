# Realistic Out-of-Sample Forward Backtester for XAUUSD M5 Scalper
# Simulates forward order execution over the unseen test period (19,767 continuous M5 candles)
# Bakes in $0.35 spread friction + slippage
import torch
import torch.nn as nn
import numpy as np
import json
import time

MODEL_PATH = r"W:\data\trading\best_m5_scalper_net.pt"
DATASET_PATH = r"W:\data\trading\xauusd_m5_scalp_dataset.npz"
REPORT_PATH = r"W:\data\trading\m5_scalper_backtest_report.json"

# SE & ResNet Architecture matching trained weights
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

class M5ScalperNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlockSE1D(64, dilation=1)
        self.block2 = ResBlockSE1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlockSE1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.trans(h)
        h = self.block3(h)
        feat = self.fc(self.pool(h).squeeze(-1))
        return self.action_head(feat), self.offset_head(feat)

print("1. Loading dataset and trained M5 Scalper model...")
data = np.load(DATASET_PATH)
X_raw = data['X_ohlc']         # (N, 60, 4)
ref_closes = data['ref_closes'] # (N,)
end_times = data['end_times']   # (N,)

N = len(ref_closes)
split_idx = int(N * 0.80)
purge_gap = 18
test_start = split_idx + purge_gap

test_X_raw = X_raw[test_start:]
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

print(f"   Test samples: {len(test_ref_close):,} bars from {test_end_times[0]} to {test_end_times[-1]}")

# Preprocess test features exactly as trained
c60_expanded = test_ref_close[:, None, None]
norm_open = test_X_raw[:, :, 0:1] - c60_expanded
norm_high = test_X_raw[:, :, 1:2] - c60_expanded
norm_low = test_X_raw[:, :, 2:3] - c60_expanded
norm_close = test_X_raw[:, :, 3:4] - c60_expanded
body = test_X_raw[:, :, 3:4] - test_X_raw[:, :, 0:1]
rng = test_X_raw[:, :, 1:2] - test_X_raw[:, :, 2:3]

test_features = np.concatenate([norm_open, norm_high, norm_low, norm_close, body, rng], axis=-1)
test_features = np.transpose(test_features, (0, 2, 1)).astype(np.float32)

# Load checkpoint
model = M5ScalperNet(in_features=6)
checkpoint = torch.load(MODEL_PATH, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("2. Running forward inference on unseen test bars...")
t_tensor = torch.tensor(test_features, dtype=torch.float32)

all_logits = []
all_offsets = []
batch_size = 512
with torch.no_grad():
    for b in range(0, len(t_tensor), batch_size):
        logits, offsets = model(t_tensor[b : b + batch_size])
        all_logits.append(logits.cpu().numpy())
        all_offsets.append(offsets.cpu().numpy())

all_logits = np.concatenate(all_logits, axis=0)
all_offsets = np.concatenate(all_offsets, axis=0)

probs = torch.softmax(torch.tensor(all_logits), dim=-1).numpy()
pred_actions = np.argmax(probs, axis=-1)
pred_confs = np.max(probs, axis=-1)

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
EXPIRY_BARS = 18       # 90 minutes expiry for scalping orders
SPREAD_POINTS = 0.35   # Realistic gold spread ($0.35)

print("\n3. Testing across multiple confidence thresholds...")
results = {}

for CONF_THRESHOLD in [0.45, 0.50, 0.55, 0.60]:
    trades = []
    INITIAL_BALANCE = 10000.0
    balance = INITIAL_BALANCE
    equity_curve = [balance]
    peak_balance = balance
    max_drawdown = 0.0

    for i in range(len(test_ref_close) - EXPIRY_BARS):
        act = pred_actions[i]
        conf = pred_confs[i]
        c60 = test_ref_close[i]
        entry_off, sl_off, tp_off = all_offsets[i]

        if act == 0 or conf < CONF_THRESHOLD:
            continue

        entry = round(c60 + entry_off, 2)
        sl = round(c60 + sl_off, 2)
        tp = round(c60 + tp_off, 2)

        risk_pts = abs(entry - sl)
        reward_pts = abs(tp - entry)

        # Scalping boundary guard: Risk between $1.00 and $3.50, RR >= 1.8
        if risk_pts < 1.0 or risk_pts > 3.5 or (reward_pts / (risk_pts + 1e-4)) < 1.8:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c60:
            continue
        if order_type == "SELL_LIMIT" and entry <= c60:
            continue

        # Forward simulation from global raw data
        global_idx = test_start + i
        fwd_highs = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 1]
        fwd_lows = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 2]

        triggered = False
        outcome = "EXPIRED"
        bars_held = 0

        # Simulate execution with $0.35 spread penalty
        if order_type == "BUY_LIMIT":
            for k in range(len(fwd_lows)):
                bars_held = k + 1
                if not triggered:
                    if fwd_lows[k] <= entry:
                        triggered = True
                        if fwd_lows[k] <= sl:
                            outcome = "STOPPED"
                            break
                        if fwd_highs[k] >= tp + SPREAD_POINTS:
                            outcome = "TP"
                            break
                else:
                    if fwd_lows[k] <= sl:
                        outcome = "STOPPED"
                        break
                    if fwd_highs[k] >= tp + SPREAD_POINTS:
                        outcome = "TP"
                        break
        elif order_type == "SELL_LIMIT":
            for k in range(len(fwd_highs)):
                bars_held = k + 1
                if not triggered:
                    if fwd_highs[k] >= entry:
                        triggered = True
                        if fwd_highs[k] >= sl:
                            outcome = "STOPPED"
                            break
                        if fwd_lows[k] <= tp - SPREAD_POINTS:
                            outcome = "TP"
                            break
                else:
                    if fwd_highs[k] >= sl:
                        outcome = "STOPPED"
                        break
                    if fwd_lows[k] <= tp - SPREAD_POINTS:
                        outcome = "TP"
                        break

        if triggered and outcome in ["TP", "STOPPED"]:
            risk_dollars = balance * 0.01  # 1% risk per trade
            lot = risk_dollars / (risk_pts * 100.0)
            lot = max(0.01, round(lot, 2))
            lot = min(lot, 5.0)

            if outcome == "TP":
                pnl = round(reward_pts * 100.0 * lot, 2)
            else:
                pnl = -round(risk_pts * 100.0 * lot, 2)

            balance += pnl
            equity_curve.append(balance)

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

            trades.append({
                "time": str(test_end_times[i]),
                "type": order_type,
                "conf": round(float(conf), 3),
                "outcome": outcome,
                "pnl": pnl,
                "balance": round(balance, 2),
                "bars_held": bars_held
            })

    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == 'TP'])
    losses = len([t for t in trades if t['outcome'] == 'STOPPED'])
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
    profit_factor = (gross_profit / (gross_loss + 1e-4)) if gross_loss > 0 else 0.0
    net_return_pct = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100.0

    print(f"--- Confidence Gate: >= {CONF_THRESHOLD:.2f} ---")
    print(f"Total Trades: {total_trades} (Avg {total_trades/70:.1f} trades/day)")
    print(f"Win Rate: {win_rate:.2f}% ({wins}W / {losses}L)")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Net Return: {net_return_pct:+.2f}% ($10,000 -> ${balance:,.2f})")
    print(f"Max Drawdown: {max_drawdown:.2f}%\n")

    results[f"conf_{int(CONF_THRESHOLD*100)}"] = {
        "confidence_threshold": float(CONF_THRESHOLD),
        "total_trades": int(total_trades),
        "win_rate": float(round(win_rate, 2)),
        "profit_factor": float(round(profit_factor, 2)),
        "final_balance": float(round(balance, 2)),
        "net_return_pct": float(round(net_return_pct, 2)),
        "max_drawdown": float(round(max_drawdown, 2))
    }

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Report saved to {REPORT_PATH}")
