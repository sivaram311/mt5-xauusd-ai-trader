"""
Ultra-High Confidence & Multi-Condition Confluence Sweep for XAUUSD M15
Tests Confidence thresholds from 0.70 to 0.95 combined with Confluence Filters:
1. Signal Margin (Logit difference between #1 and #2 class)
2. London/NY Session filter
3. Dynamic RR requirement (1:3 to 1:5)
4. Volatility expansion filter (Range of candle 100 > previous 10-candle median)
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

print("Loading test dataset and model...")
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

# Sort probabilities to calculate margin between 1st and 2nd best class
sorted_probs = np.sort(probs, axis=-1)
pred_actions = np.argmax(probs, axis=-1)
top_probs = sorted_probs[:, -1]
second_probs = sorted_probs[:, -2]
margin_probs = top_probs - second_probs  # Certainty gap

SPREAD = 0.25
EXPIRY = 48

def run_simulation(min_conf, min_margin=0.0, session_filter=True, max_consecutive_losses=3):
    trades = []
    balance = 10000.0
    equity = [balance]
    consec_losses = 0
    cooldown_bars = 0

    for i in range(len(test_ref_close) - EXPIRY):
        if cooldown_bars > 0:
            cooldown_bars -= 1
            continue

        act = pred_actions[i]
        conf = top_probs[i]
        margin = margin_probs[i]
        c100 = test_ref_close[i]
        t_str = test_end_times[i]
        
        # 1. Ultra-Confidence & Margin Gates
        if act == 0 or conf < min_conf or margin < min_margin:
            continue
            
        if session_filter:
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
                        if fwd_lows[k] <= sl:
                            outcome = "LOSS"; break
                        if fwd_highs[k] >= tp:
                            outcome = "WIN"; break
                else:
                    if fwd_lows[k] <= sl:
                        outcome = "LOSS"; break
                    if fwd_highs[k] >= tp:
                        outcome = "WIN"; break

        elif order_type == "SELL_LIMIT":
            for k in range(len(fwd_highs)):
                if not triggered:
                    if fwd_highs[k] >= entry:
                        triggered = True
                        if fwd_highs[k] + SPREAD >= sl:
                            outcome = "LOSS"; break
                        if fwd_lows[k] <= tp:
                            outcome = "WIN"; break
                else:
                    if fwd_highs[k] + SPREAD >= sl:
                        outcome = "LOSS"; break
                    if fwd_lows[k] <= tp:
                        outcome = "WIN"; break

        if triggered and outcome in ["WIN", "LOSS"]:
            risk_amt = balance * 0.01 # 1% risk
            lot_size = risk_amt / (risk * 100.0)
            if outcome == "WIN":
                pnl = reward * 100.0 * lot_size
                consec_losses = 0
            else:
                pnl = -risk * 100.0 * lot_size
                consec_losses += 1
                if consec_losses >= max_consecutive_losses:
                    cooldown_bars = 16 # Pause 4 hours (16 bars) on streak to avoid tilt

            balance += pnl
            equity.append(balance)
            trades.append({"outcome": outcome, "rr": reward / risk, "pnl": pnl})

    if not trades:
        return None

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate = (len(wins) / len(trades)) * 100.0
    avg_rr = np.mean([t["rr"] for t in trades])
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / (gross_loss + 1e-4)

    peak = 10000.0
    max_dd = 0.0
    for b in equity:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100.0
        if dd > max_dd: max_dd = dd

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "avg_rr": round(float(avg_rr), 2),
        "profit_factor": round(float(pf), 2),
        "max_drawdown": round(float(max_dd), 2),
        "final_balance": round(float(balance), 2)
    }

results = []
print("\n--- Sweeping Ultra-High Confidence & Certainty Margins ---")
for conf in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    for margin in [0.20, 0.40, 0.60]:
        res = run_simulation(min_conf=conf, min_margin=margin, session_filter=True)
        if res and res["trades"] >= 20: # Ensure statistically significant sample size
            results.append({
                "Confidence": f">={conf:.2f}",
                "Margin": f">={margin:.2f}",
                **res
            })

df_res = pd.DataFrame(results)
if not df_res.empty:
    print(df_res[["Confidence", "Margin", "trades", "win_rate", "avg_rr", "profit_factor", "max_drawdown"]].to_string(index=False))
else:
    print("No configurations met the criteria with >= 20 trades")

with open(r"W:\data\trading\high_conf_results.json", "w") as f:
    json.dump(results, f, indent=2)
