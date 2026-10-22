"""
Out-of-Sample Forward Backtester for Raw Price 1:5 Asymmetric M1 Liquidity Sniper
Evaluates across 16,902 unseen test bars.
Includes:
- Broker spread friction ($0.35/oz)
- Simulated pending order placement, fill detection, and auto-timeout based on predicted TimeToFill
- Strict Stop Loss and Take Profit execution
"""
import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = r"W:\data\trading\best_m1_raw_price_1to5_net.pt"
DATASET_PATH = r"W:\data\trading\xauusd_m1_raw_price_1to5_dataset.npz"
REPORT_PATH = r"W:\data\trading\m1_raw_price_1to5_backtest_report.json"

SPREAD_FEE = 0.35
STARTING_BALANCE = 10000.0

class RawTemporalBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)

class RawM1SniperNet(nn.Module):
    def __init__(self, in_features=7):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = RawTemporalBlock(64, dilation=1)
        self.block2 = RawTemporalBlock(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = RawTemporalBlock(128, dilation=4)
        self.block4 = RawTemporalBlock(128, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.entry_head = nn.Linear(64, 1)
        self.sl_head = nn.Linear(64, 1)
        self.tp_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x_raw):
        x_norm = self.input_norm(x_raw)
        x_in = x_norm.transpose(1, 2)
        h = self.trans(self.block2(self.block1(self.stem(x_in))))
        feat = self.fc(self.pool(self.block4(self.block3(h))).squeeze(-1))
        last_close = x_raw[:, -1, 3].unsqueeze(-1)
        confirm_logits = self.confirm_head(feat)
        dir_logits = self.dir_head(feat)
        entry_pred = last_close + self.entry_head(feat)
        sl_pred = last_close + self.sl_head(feat)
        tp_pred = last_close + self.tp_head(feat)
        ttf_pred = torch.relu(self.ttf_head(feat)) + 1.0
        return confirm_logits, dir_logits, entry_pred, sl_pred, tp_pred, ttf_pred

print("1. Loading dataset and trained checkpoint...")
data = np.load(DATASET_PATH)
X_raw = data['X_raw']
N = len(X_raw)
split_idx = int(N * 0.80)
purge = 60
test_start = split_idx + purge

test_X = X_raw[test_start:]
n_test = len(test_X)
print(f"   Unseen test samples: {n_test:,} bars")

device = torch.device("cpu")
model = RawM1SniperNet()
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Pre-extract forward price series for fast vectorized simulation
# test_X shape: (n_test, 120, 7)
# candle features: 0=open, 1=high, 2=low, 3=close
test_closes = test_X[:, -1, 3]

print("2. Running model inference across unseen test set...")
batch_size = 512
all_conf_probs = []
all_dirs = []
all_entries = []
all_sls = []
all_tps = []
all_ttfs = []

with torch.no_grad():
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        bx = torch.tensor(test_X[start:end], dtype=torch.float32)
        p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf = model(bx)
        conf_probs = torch.softmax(p_conf, dim=-1)[:, 1].numpy()
        dirs = torch.argmax(p_dir, dim=-1).numpy()
        entries = p_ent.squeeze(-1).numpy()
        sls = p_sl.squeeze(-1).numpy()
        tps = p_tp.squeeze(-1).numpy()
        ttfs = p_ttf.squeeze(-1).numpy()

        all_conf_probs.extend(conf_probs)
        all_dirs.extend(dirs)
        all_entries.extend(entries)
        all_sls.extend(sls)
        all_tps.extend(tps)
        all_ttfs.extend(ttfs)

all_conf_probs = np.array(all_conf_probs)
all_dirs = np.array(all_dirs)
all_entries = np.array(all_entries)
all_sls = np.array(all_sls)
all_tps = np.array(all_tps)
all_ttfs = np.array(all_ttfs)

print("3. Evaluating performance across confidence gates...")
results = {}

for conf_gate in [0.49, 0.50, 0.51, 0.52]:
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0

    trades = []
    wins = 0
    losses = 0
    expired = 0
    gross_profit = 0.0
    gross_loss = 0.0

    # Simulate bar-by-bar
    i = 0
    while i < n_test - 60:
        conf = all_conf_probs[i]
        if conf >= conf_gate:
            direction = all_dirs[i] # 0 = BUY_LIMIT, 1 = SELL_LIMIT
            entry = round(float(all_entries[i]), 2)
            sl = round(float(all_sls[i]), 2)
            tp = round(float(all_tps[i]), 2)
            ttf_max = min(int(round(float(all_ttfs[i]))), 25)
            ttf_max = max(ttf_max, 2)
            c_now = float(test_closes[i])

            risk_pts = abs(entry - sl)
            reward_pts = abs(tp - entry)

            # Sanity geometry check
            is_valid = False
            if direction == 0: # BUY_LIMIT
                if entry < c_now and sl < entry and tp > entry and risk_pts >= 0.40 and risk_pts <= 5.0:
                    is_valid = True
            else: # SELL_LIMIT
                if entry > c_now and sl > entry and tp < entry and risk_pts >= 0.40 and risk_pts <= 5.0:
                    is_valid = True

            if is_valid:
                # Forward simulation window
                fwd_highs = test_X[i+1 : i+1+60, -1, 1]
                fwd_lows = test_X[i+1 : i+1+60, -1, 2]

                filled = False
                hit_tp = False
                stopped = False
                outcome = "EXPIRED"

                for k in range(len(fwd_highs)):
                    if not filled:
                        if k < ttf_max:
                            if direction == 0 and fwd_lows[k] <= entry:
                                filled = True
                                if fwd_lows[k] <= sl:
                                    stopped = True
                                    outcome = "STOPPED"
                                    break
                                if fwd_highs[k] >= tp + SPREAD_FEE:
                                    hit_tp = True
                                    outcome = "TP"
                                    break
                            elif direction == 1 and fwd_highs[k] >= entry:
                                filled = True
                                if fwd_highs[k] >= sl:
                                    stopped = True
                                    outcome = "STOPPED"
                                    break
                                if fwd_lows[k] <= tp - SPREAD_FEE:
                                    hit_tp = True
                                    outcome = "TP"
                                    break
                        else:
                            outcome = "EXPIRED"
                            break
                    else:
                        if direction == 0:
                            if fwd_lows[k] <= sl:
                                stopped = True
                                outcome = "STOPPED"
                                break
                            if fwd_highs[k] >= tp + SPREAD_FEE:
                                hit_tp = True
                                outcome = "TP"
                                break
                        else:
                            if fwd_highs[k] >= sl:
                                stopped = True
                                outcome = "STOPPED"
                                break
                            if fwd_lows[k] <= tp - SPREAD_FEE:
                                hit_tp = True
                                outcome = "TP"
                                break

                if filled:
                    # Risk 0.5% of current balance per trade
                    lot = max(0.01, round((balance * 0.005) / (risk_pts * 100.0), 2))
                    lot = min(lot, 5.0)

                    if outcome == "TP":
                        pnl = round(reward_pts * 100.0 * lot, 2)
                        wins += 1
                        gross_profit += pnl
                    else:
                        pnl = -round(risk_pts * 100.0 * lot, 2)
                        losses += 1
                        gross_loss += abs(pnl)

                    balance += pnl
                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance * 100.0
                    if dd > max_drawdown:
                        max_drawdown = dd

                    trades.append(pnl)
                else:
                    expired += 1

        i += 1

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    pf = (gross_profit / (gross_loss + 1e-4)) if gross_loss > 0 else 0.0
    net_return = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100.0

    gate_name = f"conf_{int(conf_gate*100)}"
    results[gate_name] = {
        "confidence_gate": conf_gate,
        "total_filled_trades": total_trades,
        "expired_unfilled": expired,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(pf, 2),
        "net_return_pct": round(net_return, 2),
        "final_balance": round(balance, 2),
        "max_drawdown_pct": round(max_drawdown, 2)
    }

    print(f"\n--- Gate >= {conf_gate:.2f} ---")
    print(f"Total Filled Trades: {total_trades:,} | Expired Orders: {expired:,}")
    print(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {pf:.2f} | Net Return: {net_return:+.2f}% | Max DD: {max_drawdown:.2f}%")

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nReport saved to {REPORT_PATH}")
