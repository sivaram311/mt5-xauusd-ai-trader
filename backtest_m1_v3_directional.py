# Out-of-Sample Forward Backtester for Direction-Aware M1 Sniper (V3)
# Tests over the unseen 16,866 M1 test bars with $0.35 spread friction,
# widened $1.50 - $2.50 stops, and H4 macro slope directionality.
import torch
import torch.nn as nn
import numpy as np
import json
import time

MODEL_PATH = r"W:\data\trading\best_m1_v3_rl_directional_net.pt"
DATASET_PATH = r"W:\data\trading\xauusd_m1_v3_rl_directional_dataset.npz"
REPORT_PATH = r"W:\data\trading\m1_v3_directional_backtest_report.json"

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
        return x * self.fc(x).unsqueeze(-1)

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
        return self.relu(self.se(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))) + x)

class M1RLDirectionalNet(nn.Module):
    def __init__(self, in_channels_micro=7, in_features_macro=12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels_micro, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlockSE1D(64, dilation=1)
        self.block2 = ResBlockSE1D(64, dilation=2)
        self.trans1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlockSE1D(128, dilation=4)
        self.block4 = ResBlockSE1D(128, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(nn.Linear(128, 64), nn.GELU())

        self.macro_fc = nn.Sequential(
            nn.Linear(in_features_macro, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, micro, macro):
        h_m = self.trans1(self.block2(self.block1(self.stem(micro))))
        feat_micro = self.micro_fc(self.pool(self.block4(self.block3(h_m))).squeeze(-1))
        feat_macro = self.macro_fc(macro)
        feat = self.fusion_fc(torch.cat([feat_micro, feat_macro], dim=-1))
        return self.action_head(feat), self.offset_head(feat)

print("1. Loading dataset and model...")
data = np.load(DATASET_PATH)
X_micro = data['X_micro']
X_macro = data['X_macro']
ref_closes = data['ref_closes']
end_times = data['end_times']

N = len(ref_closes)
split_idx = int(N * 0.80)
purge_gap = 45
test_start = split_idx + purge_gap

test_micro_raw = X_micro[test_start:]
test_macro = X_macro[test_start:].astype(np.float32)
test_ref_close = ref_closes[test_start:]
test_end_times = end_times[test_start:]

print(f"   Unseen test samples: {len(test_ref_close):,} bars")

c180 = test_ref_close[:, None, None]
norm_o = test_micro_raw[:, :, 0:1] - c180
norm_h = test_micro_raw[:, :, 1:2] - c180
norm_l = test_micro_raw[:, :, 2:3] - c180
norm_c = test_micro_raw[:, :, 3:4] - c180
body = test_micro_raw[:, :, 3:4] - test_micro_raw[:, :, 0:1]
rng = test_micro_raw[:, :, 1:2] - test_micro_raw[:, :, 2:3]
color = test_micro_raw[:, :, 4:5]

test_micro_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng, color], axis=-1)
test_micro_feat = np.transpose(test_micro_feat, (0, 2, 1)).astype(np.float32)

model = M1RLDirectionalNet(in_channels_micro=7, in_features_macro=12)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("2. Running forward inference on unseen test bars...")
t_micro = torch.tensor(test_micro_feat, dtype=torch.float32)
t_macro = torch.tensor(test_macro, dtype=torch.float32)

all_logits = []
all_offsets = []
batch_size = 512
with torch.no_grad():
    for b in range(0, len(t_micro), batch_size):
        logits, offsets = model(t_micro[b : b + batch_size], t_macro[b : b + batch_size])
        all_logits.append(logits.cpu().numpy())
        all_offsets.append(offsets.cpu().numpy())

all_logits = np.concatenate(all_logits, axis=0)
all_offsets = np.concatenate(all_offsets, axis=0)

probs = torch.softmax(torch.tensor(all_logits), dim=-1).numpy()
pred_actions = np.argmax(probs, axis=-1)
pred_confs = np.max(probs, axis=-1)

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
EXPIRY_BARS = 25
SPREAD_FEE = 0.35

print("\n3. Testing performance across confidence tiers...")
results = {}

for CONF_GATE in [0.45, 0.50, 0.55, 0.60]:
    trades = []
    balance = 10000.0
    initial_balance = balance
    peak_balance = balance
    max_drawdown = 0.0

    for i in range(len(test_ref_close) - EXPIRY_BARS):
        act = pred_actions[i]
        conf = pred_confs[i]
        c = test_ref_close[i]
        entry_off, sl_off, tp_off = all_offsets[i]

        if act == 0 or conf < CONF_GATE:
            continue

        h4_slope = test_macro[i, 0]
        # Hard trend gate: Forbid Buy if H4 slope < -0.5, Forbid Sell if H4 slope > +0.5
        if act == 1 and h4_slope < -0.5:
            continue
        if act == 2 and h4_slope > 0.5:
            continue

        entry = round(c + entry_off, 2)
        sl = round(c + sl_off, 2)
        tp = round(c + tp_off, 2)

        risk_pts = abs(entry - sl)
        reward_pts = abs(tp - entry)

        if risk_pts < 1.20 or risk_pts > 3.50 or (reward_pts / (risk_pts + 1e-4)) < 2.0:
            continue

        order_type = ACTIONS[act]
        if order_type == "BUY_LIMIT" and entry >= c:
            continue
        if order_type == "SELL_LIMIT" and entry <= c:
            continue

        global_idx = test_start + i
        fwd_highs = X_micro[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 1]
        fwd_lows = X_micro[global_idx + 1 : global_idx + 1 + EXPIRY_BARS, -1, 2]

        triggered = False
        outcome = "EXPIRED"

        if order_type == "BUY_LIMIT":
            for k in range(len(fwd_lows)):
                if not triggered:
                    if fwd_lows[k] <= entry:
                        triggered = True
                        if fwd_lows[k] <= sl:
                            outcome = "STOPPED"
                            break
                        if fwd_highs[k] >= tp + SPREAD_FEE:
                            outcome = "TP"
                            break
                else:
                    if fwd_lows[k] <= sl:
                        outcome = "STOPPED"
                        break
                    if fwd_highs[k] >= tp + SPREAD_FEE:
                        outcome = "TP"
                        break

        elif order_type == "SELL_LIMIT":
            for k in range(len(fwd_highs)):
                if not triggered:
                    if fwd_highs[k] >= entry:
                        triggered = True
                        if fwd_highs[k] >= sl:
                            outcome = "STOPPED"
                            break
                        if fwd_lows[k] <= tp - SPREAD_FEE:
                            outcome = "TP"
                            break
                else:
                    if fwd_highs[k] >= sl:
                        outcome = "STOPPED"
                        break
                    if fwd_lows[k] <= tp - SPREAD_FEE:
                        outcome = "TP"
                        break

        if triggered and outcome in ["TP", "STOPPED"]:
            lot = max(0.01, round((balance * 0.005) / (risk_pts * 100.0), 2))
            lot = min(lot, 5.0)

            if outcome == "TP":
                pnl = round(reward_pts * 100.0 * lot, 2)
            else:
                pnl = -round(risk_pts * 100.0 * lot, 2)

            balance += pnl
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

            trades.append({
                "time": str(test_end_times[i]),
                "type": order_type,
                "outcome": outcome,
                "pnl": pnl
            })

    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == 'TP'])
    losses = len([t for t in trades if t['outcome'] == 'STOPPED'])
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
    pf = (gross_profit / (gross_loss + 1e-4)) if gross_loss > 0 else 0.0
    net_return = ((balance - initial_balance) / initial_balance) * 100.0
    trades_per_day = total_trades / 11.7
    mins_per_trade = (11.7 * 24 * 60) / (total_trades + 1e-4)

    print(f"--- Confidence Gate: >= {CONF_GATE:.2f} ---")
    print(f"Total Trades: {total_trades} (Avg 1 trade every {mins_per_trade:.1f} mins | {trades_per_day:.1f}/day)")
    print(f"Win Rate: {win_rate:.2f}% ({wins}W / {losses}L)")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Net Return: {net_return:+.2f}% ($10,000 -> ${balance:,.2f})")
    print(f"Max Drawdown: {max_drawdown:.2f}%\n")

    results[f"conf_{int(CONF_GATE*100)}"] = {
        "confidence_gate": float(CONF_GATE),
        "total_trades": int(total_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(round(win_rate, 2)),
        "profit_factor": float(round(pf, 2)),
        "net_return_pct": float(round(net_return, 2)),
        "max_drawdown": float(round(max_drawdown, 2))
    }

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Report saved to {REPORT_PATH}")
