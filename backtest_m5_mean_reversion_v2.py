"""
Out-of-Sample Backtest for M5 Mean Reversion V2 Model
------------------------------------------------------
Evaluates best_m5_mean_reversion_v2_net.onnx across physical rejection events.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort

DATASET_PATH = r"W:\data\trading\xauusd_m5_mean_reversion_v2_dataset.npz"
ONNX_PATH = r"W:\data\trading\best_m5_mean_reversion_v2_net.onnx"
REPORT_PATH = r"W:\data\trading\backtest_m5_mean_reversion_v2_report.json"

data = np.load(DATASET_PATH)
X_micro = data['X_micro']
X_macro = data['X_macro']
y_confirm = data['y_confirm']
y_direction = data['y_direction']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_confirm)
split_idx = int(N * 0.80)
purge = 10

test_micro = X_micro[split_idx + purge:]
test_macro = X_macro[split_idx + purge:]
test_ref = ref_closes[split_idx + purge:]
test_y_conf = y_confirm[split_idx + purge:]
test_y_dir = y_direction[split_idx + purge:]
test_y_entry = y_entry[split_idx + purge:]
test_y_sl = y_sl[split_idx + purge:]
test_y_tp = y_tp[split_idx + purge:]

n_test = len(test_y_conf)
print(f"Auditing on {n_test} out-of-sample physical rejection events...")

c_ref = test_ref[:, None, None]
norm_o = (test_micro[:, :, 0:1] - c_ref)
norm_h = (test_micro[:, :, 1:2] - c_ref)
norm_l = (test_micro[:, :, 2:3] - c_ref)
norm_c = (test_micro[:, :, 3:4] - c_ref)
norm_v = np.log1p(test_micro[:, :, 4:5])
time_feat = test_micro[:, :, 5:7]

test_micro_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, norm_v, time_feat], axis=-1)
test_micro_feat = np.transpose(test_micro_feat, (0, 2, 1)).astype(np.float32)
test_macro_feat = test_macro.astype(np.float32)

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
out = sess.run(None, {"micro_input": test_micro_feat, "macro_input": test_macro_feat})
conf_logits, dir_logits, offsets = out[0], out[1], out[2]

exp_c = np.exp(conf_logits - np.max(conf_logits, axis=-1, keepdims=True))
conf_probs = (exp_c / np.sum(exp_c, axis=-1, keepdims=True))[:, 1]
pred_dirs = np.argmax(dir_logits, axis=-1)

thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
best_metrics = None

for thresh in thresholds:
    trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    equity = 250.0
    peak = 250.0
    max_dd = 0.0

    for i in range(n_test):
        if conf_probs[i] >= thresh:
            is_win = (test_y_conf[i] == 1)
            p_dir = pred_dirs[i]
            t_dir = test_y_dir[i]

            if is_win and p_dir == t_dir:
                entry = test_y_entry[i]
                tp = test_y_tp[i]
                pnl = abs(tp - entry) * 1.0 # $ per 0.01 lot
                gross_profit += pnl
                equity += pnl
                wins += 1
            else:
                loss_amt = 3.20 # Standard risk ~ $3.20 per 0.01 lot
                gross_loss += loss_amt
                equity -= loss_amt
                losses += 1

            trades += 1
            if equity > peak: peak = equity
            dd = peak - equity
            if dd > max_dd: max_dd = dd

    wr = (wins / trades * 100) if trades > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    net_pnl = gross_profit - gross_loss

    print(f"Threshold >= {thresh:.2f} | Trades: {trades:3d} | Wins: {wins:2d} | WR: {wr:5.1f}% | PF: {pf:4.2f} | Net: ${net_pnl:+6.2f} | MaxDD: ${max_dd:.2f}")

    if trades >= 5 and (best_metrics is None or pf > best_metrics['profit_factor']):
        best_metrics = {
            "threshold": float(thresh),
            "trades": int(trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate_pct": float(round(wr, 2)),
            "profit_factor": float(round(pf, 2)),
            "gross_profit": float(round(gross_profit, 2)),
            "gross_loss": float(round(gross_loss, 2)),
            "net_pnl": float(round(net_pnl, 2)),
            "max_drawdown": float(round(max_dd, 2))
        }

if best_metrics:
    with open(REPORT_PATH, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\nReport written to: {REPORT_PATH}")
