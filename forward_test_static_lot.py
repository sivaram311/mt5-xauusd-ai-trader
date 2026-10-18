"""
Static 0.01 Lot Forward Testing Engine for XAUUSD M15
Tests all tracked profiles with static 0.01 lot size ($1.00 per point on Gold):
- Profiles tested:
  1. P-V1-LIVE (Conf >= 0.43 + London/NY)
  2. P-V1-BALANCED (Conf >= 0.60 + London/NY)
  3. P-V1-SWING (Conf >= 0.65 + Margin >= 0.20)
  4. P-V3-TOP1 (1 trade/day Top Ranker)
  5. P-V3-TOP2 (2 trades/day Top Ranker)
  6. P-V3-TOP3 (3 trades/day Top Ranker)
- Account parameters: Initial Balance $718.16 (your actual MT5 balance), static 0.01 lot size.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json

INITIAL_BALANCE = 718.16  # Your exact live account balance
STATIC_LOT = 0.01         # Static 0.01 lot (1 oz gold = $1.00 PnL per $1.00 move)
SPREAD = 0.25             # $0.25 gold spread

# 1. Load V1 Dataset & Model
DATASET_V1 = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"
MODEL_V1 = r"W:\data\trading\best_order_net.pt"

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

class OrderNetV1(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_features, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.GELU())
        self.block1 = ResBlock1D(64, dilation=1)
        self.block2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1), nn.BatchNorm1d(128), nn.GELU())
        self.block3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15))
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)
    def forward(self, x):
        h = self.trans(self.block2(self.block1(self.stem(x))))
        feat = self.fc(self.pool(self.block3(h)).squeeze(-1))
        return self.action_head(feat), self.offset_head(feat)

print("1. Running forward test on V1 Profiles with static 0.01 lot...")
d1 = np.load(DATASET_V1)
X_raw1 = d1['X_ohlc']
ref_c1 = d1['ref_closes']
times1 = d1['end_times']

N1 = len(ref_c1)
test_start1 = int(N1 * 0.80) + 48
test_X1 = X_raw1[test_start1:]
test_ref1 = ref_c1[test_start1:]
test_times1 = times1[test_start1:]

c100_1 = test_ref1[:, None, None]
feat1 = np.concatenate([
    test_X1[:, :, 0:1] - c100_1,
    test_X1[:, :, 1:2] - c100_1,
    test_X1[:, :, 2:3] - c100_1,
    test_X1[:, :, 3:4] - c100_1,
    test_X1[:, :, 3:4] - test_X1[:, :, 0:1],
    test_X1[:, :, 1:2] - test_X1[:, :, 2:3]
], axis=-1)
feat1 = np.transpose(feat1, (0, 2, 1)).astype(np.float32)

m1 = OrderNetV1()
m1.load_state_dict(torch.load(MODEL_V1, map_location=torch.device('cpu'), weights_only=True))
m1.eval()

with torch.no_grad():
    tx = torch.tensor(feat1, dtype=torch.float32)
    all_logits1, all_off1 = [], []
    for i in range(0, len(tx), 512):
        l, o = m1(tx[i:i+512])
        all_logits1.append(l.numpy())
        all_off1.append(o.numpy())
    logits1 = np.concatenate(all_logits1, axis=0)
    offsets1 = np.concatenate(all_off1, axis=0)

probs1 = np.exp(logits1) / np.sum(np.exp(logits1), axis=-1, keepdims=True)
sorted_probs1 = np.sort(probs1, axis=-1)
pred_act1 = np.argmax(probs1, axis=-1)
top_p1 = sorted_probs1[:, -1]
margin1 = top_p1 - sorted_probs1[:, -2]

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]

def run_forward_test_v1(name, min_conf, min_margin=0.0):
    balance = INITIAL_BALANCE
    equity = [balance]
    trades = []
    EXPIRY = 48

    for i in range(len(test_ref1) - EXPIRY):
        act = pred_act1[i]
        conf = top_p1[i]
        m = margin1[i]
        c100 = test_ref1[i]
        t_str = test_times1[i]

        if act == 0 or conf < min_conf or m < min_margin:
            continue
        hour = int(t_str[11:13])
        if hour < 7 or hour > 18:
            continue

        entry = round(c100 + offsets1[i, 0], 2)
        sl = round(c100 + offsets1[i, 1], 2)
        tp = round(c100 + offsets1[i, 2], 2)

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk < 1.0 or risk > 15.0 or (reward / (risk + 1e-4)) < 2.0:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c100: continue
        if order_type == "SELL_LIMIT" and entry <= c100: continue

        global_idx = test_start1 + i
        fwd_h = X_raw1[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 1]
        fwd_l = X_raw1[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 2]

        triggered, outcome = False, "EXPIRED"
        if order_type == "BUY_LIMIT":
            for k in range(len(fwd_l)):
                if not triggered:
                    if fwd_l[k] <= entry:
                        triggered = True
                        if fwd_l[k] <= sl: outcome = "LOSS"; break
                        if fwd_h[k] >= tp: outcome = "WIN"; break
                else:
                    if fwd_l[k] <= sl: outcome = "LOSS"; break
                    if fwd_h[k] >= tp: outcome = "WIN"; break
        elif order_type == "SELL_LIMIT":
            for k in range(len(fwd_h)):
                if not triggered:
                    if fwd_h[k] >= entry:
                        triggered = True
                        if fwd_h[k] + SPREAD >= sl: outcome = "LOSS"; break
                        if fwd_l[k] <= tp: outcome = "WIN"; break
                else:
                    if fwd_h[k] + SPREAD >= sl: outcome = "LOSS"; break
                    if fwd_l[k] <= tp: outcome = "WIN"; break

        if triggered and outcome in ["WIN", "LOSS"]:
            # STATIC 0.01 LOT: 1 point ($1.00 move in gold) = $1.00 profit/loss
            pnl = (reward * 1.0) if outcome == "WIN" else (-risk * 1.0)
            balance += pnl
            equity.append(balance)
            trades.append({"outcome": outcome, "pnl": pnl, "risk": risk, "reward": reward})

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    pf = (sum(t["pnl"] for t in wins) / (abs(sum(t["pnl"] for t in losses)) + 1e-4)) if losses else 0.0

    peak = INITIAL_BALANCE
    max_dd = 0.0
    for b in equity:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100.0
        if dd > max_dd: max_dd = dd

    net_profit = balance - INITIAL_BALANCE
    return {
        "Profile": name,
        "Total Trades": len(trades),
        "Trades / Day": round(len(trades) / 218, 1),
        "Win Rate": f"{win_rate:.1f}%",
        "Avg Win ($)": f"+${avg_win:.2f}",
        "Avg Loss ($)": f"-${avg_loss:.2f}",
        "Profit Factor": round(float(pf), 2),
        "Max Drawdown": f"{max_dd:.1f}%",
        "Net Profit ($)": f"+${net_profit:.2f}" if net_profit >= 0 else f"-${abs(net_profit):.2f}",
        "Final Balance": f"${balance:.2f}"
    }

# Run tests
res_live = run_forward_test_v1("P-V1-LIVE (Conf >= 0.43)", min_conf=0.43)
res_balanced = run_forward_test_v1("P-V1-BALANCED (Conf >= 0.60)", min_conf=0.60)
res_swing = run_forward_test_v1("P-V1-SWING (Conf >= 0.65, Margin >= 0.20)", min_conf=0.65, min_margin=0.20)

# 2. V3 Top-K Tests with Static 0.01 Lot
print("2. Running forward test on V3 Top-K Profiles with static 0.01 lot...")
d3 = np.load(r"W:\data\trading\xauusd_m15_v3_high_acc_dataset.npz")
X_raw3 = d3['X_m15']
ref_c3 = d3['ref_closes']
times3 = d3['end_times']

N3 = len(ref_c3)
test_start3 = int(N3 * 0.80) + 32
test_X3 = X_raw3[test_start3:]
test_ref3 = ref_c3[test_start3:]
test_times3 = times3[test_start3:]

c100_3 = test_ref3[:, None, None]
feat3 = np.concatenate([
    test_X3[:, :, 0:1] - c100_3,
    test_X3[:, :, 1:2] - c100_3,
    test_X3[:, :, 2:3] - c100_3,
    test_X3[:, :, 3:4] - c100_3,
    test_X3[:, :, 3:4] - test_X3[:, :, 0:1],
    test_X3[:, :, 1:2] - test_X3[:, :, 2:3]
], axis=-1)
feat3 = np.transpose(feat3, (0, 2, 1)).astype(np.float32)

class OrderNetV3(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_features, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.GELU())
        self.b1 = ResBlock1D(64, dilation=1)
        self.b2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1), nn.BatchNorm1d(128), nn.GELU())
        self.b3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.20))
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)
    def forward(self, x):
        h = self.trans(self.b2(self.b1(self.stem(x))))
        feat = self.fc(self.pool(self.b3(h)).squeeze(-1))
        return self.action_head(feat), self.offset_head(feat)

m3 = OrderNetV3()
m3.load_state_dict(torch.load(r"W:\data\trading\best_order_net_v3.pt", map_location=torch.device('cpu'), weights_only=True))
m3.eval()

with torch.no_grad():
    tx3 = torch.tensor(feat3, dtype=torch.float32)
    all_logits3, all_off3 = [], []
    for i in range(0, len(tx3), 512):
        l, o = m3(tx3[i:i+512])
        all_logits3.append(l.numpy())
        all_off3.append(o.numpy())
    logits3 = np.concatenate(all_logits3, axis=0)
    offsets3 = np.concatenate(all_off3, axis=0)

probs3 = np.exp(logits3) / np.sum(np.exp(logits3), axis=-1, keepdims=True)
pred_act3 = np.argmax(probs3, axis=-1)
trade_p3 = probs3[:, 1] + probs3[:, 2]

df_t3 = pd.DataFrame({
    'idx': np.arange(len(test_ref3)),
    'time_dt': pd.to_datetime(test_times3, unit='s'),
    'trade_prob': trade_p3,
    'pred_action': pred_act3
})
df_t3['date'] = df_t3['time_dt'].dt.date
df_t3['hour'] = df_t3['time_dt'].dt.hour
df_sess3 = df_t3[(df_t3['hour'] >= 7) & (df_t3['hour'] <= 18)].copy()

def run_forward_test_v3(name, k_per_day):
    selected_indices = []
    for date, group in df_sess3.groupby('date'):
        top_k = group[group['pred_action'] > 0].nlargest(k_per_day, 'trade_prob')
        selected_indices.extend(top_k['idx'].tolist())
    selected_indices = set(selected_indices)

    balance = INITIAL_BALANCE
    equity = [balance]
    trades = []
    EXPIRY = 32

    for i in range(len(test_ref3) - EXPIRY):
        if i not in selected_indices:
            continue
        act = pred_act3[i]
        c100 = test_ref3[i]
        entry = round(c100 + offsets3[i, 0], 2)
        sl = round(c100 + offsets3[i, 1], 2)
        tp = round(c100 + offsets3[i, 2], 2)

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk < 1.0 or risk > 10.0 or (reward / (risk + 1e-4)) < 2.0:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c100: continue
        if order_type == "SELL_LIMIT" and entry <= c100: continue

        global_idx = test_start3 + i
        fwd_h = X_raw3[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 1]
        fwd_l = X_raw3[global_idx + 1 : global_idx + 1 + EXPIRY, -1, 2]

        triggered, outcome = False, "EXPIRED"
        if order_type == "BUY_LIMIT":
            for k in range(len(fwd_l)):
                if not triggered:
                    if fwd_l[k] <= entry:
                        triggered = True
                        if fwd_l[k] <= sl: outcome = "LOSS"; break
                        if fwd_h[k] >= tp: outcome = "WIN"; break
                else:
                    if fwd_l[k] <= sl: outcome = "LOSS"; break
                    if fwd_h[k] >= tp: outcome = "WIN"; break
        elif order_type == "SELL_LIMIT":
            for k in range(len(fwd_h)):
                if not triggered:
                    if fwd_h[k] >= entry:
                        triggered = True
                        if fwd_h[k] + SPREAD >= sl: outcome = "LOSS"; break
                        if fwd_l[k] <= tp: outcome = "WIN"; break
                else:
                    if fwd_h[k] + SPREAD >= sl: outcome = "LOSS"; break
                    if fwd_l[k] <= tp: outcome = "WIN"; break

        if triggered and outcome in ["WIN", "LOSS"]:
            pnl = (reward * 1.0) if outcome == "WIN" else (-risk * 1.0)
            balance += pnl
            equity.append(balance)
            trades.append({"outcome": outcome, "pnl": pnl})

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    pf = (sum(t["pnl"] for t in wins) / (abs(sum(t["pnl"] for t in losses)) + 1e-4)) if losses else 0.0

    peak = INITIAL_BALANCE
    max_dd = 0.0
    for b in equity:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100.0
        if dd > max_dd: max_dd = dd

    net_profit = balance - INITIAL_BALANCE
    num_days = len(df_sess3['date'].unique())

    return {
        "Profile": name,
        "Total Trades": len(trades),
        "Trades / Day": round(len(trades) / num_days, 1),
        "Win Rate": f"{win_rate:.1f}%",
        "Avg Win ($)": f"+${avg_win:.2f}",
        "Avg Loss ($)": f"-${avg_loss:.2f}",
        "Profit Factor": round(float(pf), 2),
        "Max Drawdown": f"{max_dd:.1f}%",
        "Net Profit ($)": f"+${net_profit:.2f}" if net_profit >= 0 else f"-${abs(net_profit):.2f}",
        "Final Balance": f"${balance:.2f}"
    }

res_top1 = run_forward_test_v3("P-V3-TOP1 (Top 1 / Day)", k_per_day=1)
res_top2 = run_forward_test_v3("P-V3-TOP2 (Top 2 / Day)", k_per_day=2)
res_top3 = run_forward_test_v3("P-V3-TOP3 (Top 3 / Day)", k_per_day=3)

all_results = [res_live, res_balanced, res_swing, res_top1, res_top2, res_top3]
df_final = pd.DataFrame(all_results)

print("\n" + "="*115)
print(f"FORWARD TESTING REPORT — STATIC 0.01 LOT (STARTING BALANCE: ${INITIAL_BALANCE:.2f})")
print("="*115)
print(df_final.to_string(index=False))
print("="*115)

with open(r"W:\data\trading\forward_test_static_001_report.json", "w") as f:
    json.dump(all_results, f, indent=2)
