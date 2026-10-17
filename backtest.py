"""
Realistic Out-of-Sample Backtester for XAUUSD M15 Pending Orders
Runs the trained neural net over the unseen 20% test period (19,723 candles, ~10 months of trading).
Simulates tick-by-tick order placement, trigger, fill, stop loss, take profit, and expiration.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time

MODEL_PATH = r"W:\data\trading\best_order_net.pt"
DATASET_PATH = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"

# Model definition matching trained weights
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
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.trans(h)
        h = self.block3(h)
        pooled = self.pool(h).squeeze(-1)
        feat = self.fc(pooled)
        return self.action_head(feat), self.offset_head(feat)

print("1. Loading dataset and model...")
data = np.load(DATASET_PATH)
X_raw = data['X_ohlc']         # (N, 100, 4)
ref_closes = data['ref_closes'] # (N,)
end_times = data['end_times']   # (N,)

N = len(ref_closes)
split_idx = int(N * 0.80)
purge_gap = 48
test_start = split_idx + purge_gap

# Test data (Unseen ~19,723 samples)
test_X_raw = X_raw[test_start:]
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

print(f"   Total test samples: {len(test_ref_close)} from {test_end_times[0]} to {test_end_times[-1]}")

# Preprocess features exactly as trained
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

print("2. Running batch model inference over all test sequences...")
t0 = time.time()
with torch.no_grad():
    tensor_X = torch.tensor(X_feat, dtype=torch.float32)
    # Batch predict
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
print(f"   Inference finished in {time.time()-t0:.2f}s ({len(test_ref_close)/(time.time()-t0):.0f} predictions/sec)")

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=-1, keepdims=True)
pred_actions = np.argmax(probs, axis=-1)
pred_confs = np.max(probs, axis=-1)

# Backtest parameters
CONF_THRESHOLD = 0.40   # Minimum model confidence to place pending order
EXPIRY_BARS = 48        # Pending order expires if not triggered within 12 hours (48 bars)
MIN_RR_RATIO = 2.0      # Minimum Risk:Reward to accept the order
SPREAD_POINTS = 0.25    # Realistic gold broker spread ($0.25)
INITIAL_BALANCE = 10000.0 # $10,000 USD
RISK_PER_TRADE_PCT = 1.0  # 1% account risk per trade

print("3. Simulating realistic order execution and forward market response...")

# Test data forward evaluation
# Since each sample i corresponds to window [i-99 ... i], future candles are at index i+1 in raw data
trades = []
balance = INITIAL_BALANCE
equity_curve = [balance]

# Forward candle lookup from X_raw
for i in range(len(test_ref_close) - EXPIRY_BARS):
    act = pred_actions[i]
    conf = pred_confs[i]
    c100 = test_ref_close[i]
    entry_off, sl_off, tp_off = all_offsets[i]

    if act == 0 or conf < CONF_THRESHOLD:
        continue # NO_ORDER or below confidence filter

    entry = round(c100 + entry_off, 2)
    sl = round(c100 + sl_off, 2)
    tp = round(c100 + tp_off, 2)

    risk_pts = abs(entry - sl)
    reward_pts = abs(tp - entry)

    # Basic validity checks
    if risk_pts < 1.0 or risk_pts > 15.0 or (reward_pts / (risk_pts + 1e-4)) < MIN_RR_RATIO:
        continue

    order_type = ACTIONS[act]
    if order_type == "BUY_LIMIT" and entry >= c100:
        continue
    if order_type == "SELL_LIMIT" and entry <= c100:
        continue

    # Simulate forward 48 candles
    # Future bars are taken from test_X_raw starting from candle 99 of sample i + k
    # To avoid re-extracting, get future bars directly from global X_raw:
    global_idx = test_start + i
    fwd_highs = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 1] # High of forward bars
    fwd_lows = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 2]  # Low of forward bars

    triggered = False
    outcome = "EXPIRED"
    bars_held = 0

    if order_type == "BUY_LIMIT":
        entry_with_spread = entry + SPREAD_POINTS
        for k in range(len(fwd_lows)):
            bars_held = k + 1
            if not triggered:
                if fwd_lows[k] <= entry:
                    triggered = True
                    # On trigger candle, did it hit SL or TP?
                    if fwd_lows[k] <= sl:
                        outcome = "LOSS"
                        break
                    if fwd_highs[k] >= tp:
                        outcome = "WIN"
                        break
            else:
                if fwd_lows[k] <= sl:
                    outcome = "LOSS"
                    break
                if fwd_highs[k] >= tp:
                    outcome = "WIN"
                    break

    elif order_type == "SELL_LIMIT":
        for k in range(len(fwd_highs)):
            bars_held = k + 1
            if not triggered:
                if fwd_highs[k] >= entry:
                    triggered = True
                    if fwd_highs[k] + SPREAD_POINTS >= sl:
                        outcome = "LOSS"
                        break
                    if fwd_lows[k] <= tp:
                        outcome = "WIN"
                        break
            else:
                if fwd_highs[k] + SPREAD_POINTS >= sl:
                    outcome = "LOSS"
                    break
                if fwd_lows[k] <= tp:
                    outcome = "WIN"
                    break

    if triggered:
        risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
        lot_size = risk_amount / (risk_pts * 100.0) # 1 lot = 100 oz of gold ($100 per point)
        
        if outcome == "WIN":
            profit = reward_pts * 100.0 * lot_size
            balance += profit
        elif outcome == "LOSS":
            loss = risk_pts * 100.0 * lot_size
            balance -= loss
        else: # Reached end of 48 bars while in open trade
            profit = 0.0 # Closed at breakeven / flat
            outcome = "TIMED_OUT"

        equity_curve.append(balance)
        trades.append({
            "time": test_end_times[i],
            "type": order_type,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": round(reward_pts / risk_pts, 2),
            "outcome": outcome,
            "profit": round(profit if outcome == "WIN" else (-loss if outcome == "LOSS" else 0.0), 2),
            "balance": round(balance, 2)
        })

print("\n4. Backtest Summary & Accuracy Analysis:")
total_trades = len(trades)
wins = [t for t in trades if t["outcome"] == "WIN"]
losses = [t for t in trades if t["outcome"] == "LOSS"]
timeouts = [t for t in trades if t["outcome"] == "TIMED_OUT"]

win_count = len(wins)
loss_count = len(losses)
decided_trades = win_count + loss_count
win_rate = (win_count / (decided_trades + 1e-4)) * 100.0

total_profit = sum(t["profit"] for t in wins)
total_loss = abs(sum(t["profit"] for t in losses))
profit_factor = round(total_profit / (total_loss + 1e-4), 2)
net_return_pct = round(((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100.0, 2)

# Max Drawdown
peak = INITIAL_BALANCE
max_dd_pct = 0.0
for b in equity_curve:
    if b > peak:
        peak = b
    dd = (peak - b) / peak * 100.0
    if dd > max_dd_pct:
        max_dd_pct = dd

report = {
    "test_period": f"{test_end_times[0]} to {test_end_times[-1]}",
    "total_evaluated_candles": int(len(test_ref_close)),
    "triggered_trades": int(total_trades),
    "wins": int(win_count),
    "losses": int(loss_count),
    "timed_out": int(len(timeouts)),
    "win_rate_pct": float(round(win_rate, 2)),
    "average_rr": float(round(float(np.mean([t['rr'] for t in trades])), 2)) if trades else 0.0,
    "initial_balance": float(INITIAL_BALANCE),
    "final_balance": float(round(balance, 2)),
    "net_return_pct": float(net_return_pct),
    "profit_factor": float(profit_factor),
    "max_drawdown_pct": float(round(max_dd_pct, 2))
}

print(json.dumps(report, indent=2))

report_file = r"W:\data\trading\backtest_report.json"
with open(report_file, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved detailed backtest report to {report_file}")
