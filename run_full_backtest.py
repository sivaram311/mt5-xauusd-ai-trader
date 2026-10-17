"""
Comprehensive Multi-Profile Backtester for XAUUSD M15
Tests 3 clear operational profiles over the 10-month out-of-sample period (19,723 unseen candles):
Profile 1: Active Day Trader (Conf >= 0.50, ~8-9 trades/day)
Profile 2: Balanced Intraday (Conf >= 0.60, ~3 trades/day)
Profile 3: High Precision Swing (Conf >= 0.65, ~3-4 trades/week)
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json

MODEL_PATH = r"W:\data\trading\best_order_net.pt"
DATASET_PATH = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"

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

print("1. Loading test dataset...")
data = np.load(DATASET_PATH)
X_raw = data['X_ohlc']
ref_closes = data['ref_closes']
end_times = data['end_times']

N = len(ref_closes)
split_idx = int(N * 0.80)
purge_gap = 48
test_start = split_idx + purge_gap

test_X_raw = X_raw[test_start:]
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

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

print("2. Computing model predictions over 19,723 test candles...")
with torch.no_grad():
    tensor_X = torch.tensor(X_feat, dtype=torch.float32)
    batch_size = 512
    all_logits, all_offsets = [], []
    for i in range(0, len(tensor_X), batch_size):
        bx = tensor_X[i:i+batch_size]
        logits, offsets = model(bx)
        all_logits.append(logits.numpy())
        all_offsets.append(offsets.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_offsets = np.concatenate(all_offsets, axis=0)

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=-1, keepdims=True)
sorted_probs = np.sort(probs, axis=-1)
pred_actions = np.argmax(probs, axis=-1)
top_probs = sorted_probs[:, -1]
margin_probs = top_probs - sorted_probs[:, -2]

SPREAD = 0.25 # $0.25 Gold Spread
EXPIRY = 48   # 12 hours max wait for pending trigger
TRADING_DAYS = 218 # Active trading days in test slice

def execute_profile(name, min_conf, min_margin=0.0):
    balance = 10000.0
    equity = [balance]
    trades = []
    
    for i in range(len(test_ref_close) - EXPIRY):
        act = pred_actions[i]
        conf = top_probs[i]
        margin = margin_probs[i]
        c100 = test_ref_close[i]
        t_str = test_end_times[i]
        
        # Confidence & Margin Gates
        if act == 0 or conf < min_conf or margin < min_margin:
            continue
            
        # London & NY session only (07:00 to 18:00 UTC)
        hour = int(t_str[11:13])
        if hour < 7 or hour > 18:
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
        fwd_highs = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 1]
        fwd_lows = X_raw[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 2]

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
            risk_amt = balance * 0.01 # 1% risk per trade
            lot_size = risk_amt / (risk * 100.0)
            pnl = (reward * 100.0 * lot_size) if outcome == "WIN" else (-risk * 100.0 * lot_size)
            balance += pnl
            equity.append(balance)
            trades.append({
                "time": t_str,
                "type": order_type,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "risk_pts": round(risk, 2),
                "reward_pts": round(reward, 2),
                "rr": round(reward / risk, 2),
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "balance": round(balance, 2)
            })

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
        "Profile": name,
        "Total Trades": len(trades),
        "Trades / Day": round(entries_per_day, 1),
        "Trades / Week": round(entries_per_day * 5, 1),
        "Win Rate": f"{win_rate:.1f}%",
        "Average R:R": f"1:{avg_rr:.2f}",
        "Profit Factor": round(float(pf), 2),
        "Max Drawdown": f"{max_dd:.1f}%",
        "Initial Balance": "$10,000",
        "Final Balance": f"${balance:,.2f}",
        "Net Profit": f"+${balance - 10000:,.2f}"
    }

print("3. Executing backtests across all 3 profiles...")
p1 = execute_profile("Active Day Trading (Conf >= 0.50)", min_conf=0.50, min_margin=0.0)
p2 = execute_profile("Balanced Intraday (Conf >= 0.60)", min_conf=0.60, min_margin=0.0)
p3 = execute_profile("High Precision Swing (Conf >= 0.65)", min_conf=0.65, min_margin=0.20)

df_comparison = pd.DataFrame([p1, p2, p3])
print("\n" + "="*95)
print("FINAL OUT-OF-SAMPLE BACKTEST COMPARISON (DEC 2025 TO OCT 2026)")
print("="*95)
print(df_comparison[["Profile", "Trades / Day", "Trades / Week", "Win Rate", "Average R:R", "Profit Factor", "Max Drawdown", "Final Balance"]].to_string(index=False))
print("="*95)

# Save JSON report
with open(r"W:\data\trading\profile_backtest_report.json", "w") as f:
    json.dump([p1, p2, p3], f, indent=2)
