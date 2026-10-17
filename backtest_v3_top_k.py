"""
Top-K Daily Ranker Backtester for V3 Architecture
Instead of static cutoffs, tests dynamic daily ranking to guarantee 1 to 5 entries per day
during London & NY active trading sessions.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json

MODEL_PATH = r"W:\data\trading\best_order_net_v3.pt"
DATASET_FILE = r"W:\data\trading\xauusd_m15_v3_high_acc_dataset.npz"

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

class HighAccuracyOrderNetV3(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.b1 = ResBlock1D(64, dilation=1)
        self.b2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.b3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.trans(h)
        h = self.b3(h)
        pooled = self.pool(h).squeeze(-1)
        feat = self.fc(pooled)
        return self.action_head(feat), self.offset_head(feat)

print("1. Loading V3 test data and model...")
data = np.load(DATASET_FILE)
X_raw = data['X_m15']
ref_closes = data['ref_closes']
end_times = data['end_times']

N = len(ref_closes)
split_idx = int(N * 0.80)
purge = 32
test_start = split_idx + purge

test_X_raw = X_raw[test_start:]
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

c100 = test_ref_close[:, None, None]
norm_o = test_X_raw[:, :, 0:1] - c100
norm_h = test_X_raw[:, :, 1:2] - c100
norm_l = test_X_raw[:, :, 2:3] - c100
norm_c = test_X_raw[:, :, 3:4] - c100
body = test_X_raw[:, :, 3:4] - test_X_raw[:, :, 0:1]
rng = test_X_raw[:, :, 1:2] - test_X_raw[:, :, 2:3]

X_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng], axis=-1)
X_feat = np.transpose(X_feat, (0, 2, 1)).astype(np.float32)

model = HighAccuracyOrderNetV3(in_features=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

print("2. Batch inference over test sequences...")
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

# Trade probabilities (P(BUY) + P(SELL))
trade_probs = probs[:, 1] + probs[:, 2]
pred_actions = np.argmax(probs, axis=-1)

# Group test sequences by Date
df_test = pd.DataFrame({
    'idx': np.arange(len(test_ref_close)),
    'time_dt': pd.to_datetime(test_end_times, unit='s'),
    'trade_prob': trade_probs,
    'pred_action': pred_actions
})
df_test['date'] = df_test['time_dt'].dt.date
df_test['hour'] = df_test['time_dt'].dt.hour

# Filter active London/NY session: 07:00 to 18:00 UTC
df_session = df_test[(df_test['hour'] >= 7) & (df_test['hour'] <= 18)].copy()

SPREAD = 0.25
EXPIRY = 32
TARGET_RR = 3.0
RISK_PTS = 2.0

def evaluate_top_k(k_per_day=3):
    # Select top K candles per day with highest trade probability
    selected_indices = []
    for date, group in df_session.groupby('date'):
        top_k = group[group['pred_action'] > 0].nlargest(k_per_day, 'trade_prob')
        selected_indices.extend(top_k['idx'].tolist())
    selected_indices = set(selected_indices)

    trades = []
    balance = 10000.0
    equity = [balance]

    for i in range(len(test_ref_close) - EXPIRY):
        if i not in selected_indices:
            continue

        act = pred_actions[i]
        c100_val = test_ref_close[i]
        entry_off, sl_off, tp_off = all_offsets[i]

        entry = round(c100_val + entry_off, 2)
        sl = round(c100_val + sl_off, 2)
        tp = round(c100_val + tp_off, 2)

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk < 1.0 or risk > 10.0 or (reward / (risk + 1e-4)) < 2.0:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c100_val: continue
        if order_type == "SELL_LIMIT" and entry <= c100_val: continue

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
            risk_amt = balance * 0.01
            lot_size = risk_amt / (risk * 100.0)
            pnl = (reward * 100.0 * lot_size) if outcome == "WIN" else (-risk * 100.0 * lot_size)
            balance += pnl
            equity.append(balance)
            trades.append({"outcome": outcome, "rr": reward / risk, "pnl": pnl})

    if not trades:
        return None

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

    num_days = len(df_session['date'].unique())
    entries_per_day = len(trades) / num_days

    return {
        "Target Top-K": f"Top {k_per_day} / Day",
        "Total Trades": len(trades),
        "Entries / Day": round(entries_per_day, 1),
        "Win Rate": f"{win_rate:.1f}%",
        "Average R:R": f"1:{avg_rr:.2f}",
        "Profit Factor": round(float(pf), 2),
        "Max Drawdown": f"{max_dd:.1f}%",
        "Final Balance": f"${balance:,.2f}"
    }

print("\n--- Top-K Daily Ranker Performance Sweep (1 to 5 Entries/Day) ---")
k_results = []
for k in [1, 2, 3, 4, 5]:
    res = evaluate_top_k(k_per_day=k)
    if res:
        k_results.append(res)

df_rank = pd.DataFrame(k_results)
print(df_rank.to_string(index=False))

with open(r"W:\data\trading\v3_top_k_backtest_report.json", "w") as f:
    json.dump(k_results, f, indent=2)
