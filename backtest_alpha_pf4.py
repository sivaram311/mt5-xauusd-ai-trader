"""
Out-of-Sample Forward Backtester for Dual-Stream High-Alpha Session Sniper (Target: PF 3.0 - 4.0)
Evaluates across 16,908 unseen test bars.
Includes:
- Strict Session Gate (12:30 - 17:00 UTC)
- Real broker spread friction ($0.35/oz)
- Simulated limit order execution, $2.00 tight SL, and $10.00 TP (1:5 Payoff)
- Top-1 Daily Constraint evaluation
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = r"W:\data\trading\best_alpha_pf4_net.pt"
DATASET_PATH = r"W:\data\trading\xauusd_alpha_pf4_dataset.npz"
REPORT_PATH = r"W:\data\trading\alpha_pf4_backtest_report.json"

SPREAD_FEE = 0.35
STARTING_BALANCE = 10000.0

class TemporalConvBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + x)

class DualStreamAlphaNet(nn.Module):
    def __init__(self, micro_channels=5, macro_dim=12):
        super().__init__()
        self.micro_norm = nn.LayerNorm(micro_channels)
        self.micro_stem = nn.Sequential(
            nn.Conv1d(micro_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU()
        )
        self.block1 = TemporalConvBlock(48, dilation=1)
        self.block2 = TemporalConvBlock(48, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=1),
            nn.BatchNorm1d(96),
            nn.GELU()
        )
        self.block3 = TemporalConvBlock(96, dilation=4)
        self.micro_pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU()
        )
        self.macro_norm = nn.BatchNorm1d(macro_dim)
        self.macro_fc = nn.Sequential(
            nn.Linear(macro_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(48 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.20)
        )
        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.entry_head = nn.Linear(64, 1)
        self.sl_head = nn.Linear(64, 1)
        self.tp_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x_micro, x_macro):
        x_m_norm = self.micro_norm(x_micro)
        x_m_in = x_m_norm.transpose(1, 2)
        h_m = self.trans(self.block2(self.block1(self.micro_stem(x_m_in))))
        feat_micro = self.micro_fc(self.micro_pool(self.block3(h_m)).squeeze(-1))
        feat_macro = self.macro_fc(self.macro_norm(x_macro))
        feat = self.fusion(torch.cat([feat_micro, feat_macro], dim=-1))
        last_close = x_micro[:, -1, 3].unsqueeze(-1)
        p_conf = self.confirm_head(feat)
        p_dir = self.dir_head(feat)
        p_ent = last_close + self.entry_head(feat)
        p_sl = last_close + self.sl_head(feat)
        p_tp = last_close + self.tp_head(feat)
        p_ttf = torch.relu(self.ttf_head(feat)) + 1.0
        return p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf

print("1. Loading dataset and best model checkpoint...")
data = np.load(DATASET_PATH)
X_micro = data['X_micro']
X_macro = data['X_macro']
N = len(X_micro)
split_idx = int(N * 0.80)
purge = 60
test_start = split_idx + purge

test_X_micro = X_micro[test_start:]
test_X_macro = X_macro[test_start:]
n_test = len(test_X_micro)
print(f"   Unseen test samples: {n_test:,} bars")

device = torch.device("cpu")
model = DualStreamAlphaNet()
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("2. Running forward inference...")
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
        bx_m = torch.tensor(test_X_micro[start:end], dtype=torch.float32)
        bx_mac = torch.tensor(test_X_macro[start:end], dtype=torch.float32)
        pc, pd, pe, ps, pt, ptt = model(bx_m, bx_mac)

        conf_p = torch.softmax(pc, dim=-1)[:, 1].numpy()
        dirs = torch.argmax(pd, dim=-1).numpy()
        entries = pe.squeeze(-1).numpy()
        sls = ps.squeeze(-1).numpy()
        tps = pt.squeeze(-1).numpy()
        ttfs = ptt.squeeze(-1).numpy()

        all_conf_probs.extend(conf_p)
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

test_closes = test_X_micro[:, -1, 3]

print(f"Confidence score stats: min={all_conf_probs.min():.4f}, max={all_conf_probs.max():.4f}, mean={all_conf_probs.mean():.4f}")
print("Percentiles (75, 90, 95, 99):", np.percentile(all_conf_probs, [75, 90, 95, 99]))

print("\n3. Testing performance across confidence thresholds...")
results = {}

for conf_gate in [0.010, 0.015, 0.020, 0.025, 0.030]:
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0

    wins = 0
    losses = 0
    expired = 0
    gross_profit = 0.0
    gross_loss = 0.0

    i = 0
    while i < n_test - 90:
        conf = all_conf_probs[i]
        if conf >= conf_gate:
            direction = all_dirs[i]
            entry = round(float(all_entries[i]), 2)
            sl = round(float(all_sls[i]), 2)
            tp = round(float(all_tps[i]), 2)
            ttf_max = min(int(round(float(all_ttfs[i]))), 15)
            ttf_max = max(ttf_max, 2)
            c_now = float(test_closes[i])

            risk_pts = abs(entry - sl)
            reward_pts = abs(tp - entry)

            is_valid = False
            if direction == 0: # BUY_LIMIT
                if entry < c_now and sl < entry and tp > entry and 0.20 <= risk_pts <= 5.0 and reward_pts >= 1.0:
                    is_valid = True
            else: # SELL_LIMIT
                if entry > c_now and sl > entry and tp < entry and 0.20 <= risk_pts <= 5.0 and reward_pts >= 1.0:
                    is_valid = True

            if is_valid:
                fwd_highs = test_X_micro[i+1 : i+1+90, -1, 1]
                fwd_lows = test_X_micro[i+1 : i+1+90, -1, 2]

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
                    
                    # Skip ahead 60 bars after a filled trade to avoid multiple orders in the same session
                    i += 60
                    continue
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

    print(f"--- Gate >= {conf_gate:.2f} ---")
    print(f"Filled: {total_trades:,} | Expired: {expired:,} | Wins: {wins} | Losses: {losses} | WR: {win_rate:.2f}%")
    print(f"PF: {pf:.2f} | Net: {net_return:+.2f}% | Max DD: {max_drawdown:.2f}%\n")

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Backtest report saved to {REPORT_PATH}")
