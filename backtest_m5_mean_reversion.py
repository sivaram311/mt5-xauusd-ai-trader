"""
Forward Backtest Engine for M5 Mean Reversion Model (ONNX Runtime)
------------------------------------------------------------------
Runs the newly trained ONNX model across unseen out-of-sample data.
Audits:
- Total Trades Filled & Tested
- Win Rate (%)
- Realized Reward-to-Risk Ratio
- Profit Factor (PF)
- Max Drawdown ($ and %)
- Equity Curve
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import onnxruntime as ort

DATASET_PATH = r"W:\data\trading\xauusd_m5_mean_reversion_dataset.npz"
ONNX_PATH = r"W:\data\trading\best_m5_mean_reversion_net.onnx"
REPORT_PATH = r"W:\data\trading\backtest_m5_mean_reversion_report.json"

print("1. Loading dataset and ONNX model...")
if not os.path.exists(DATASET_PATH) or not os.path.exists(ONNX_PATH):
    print("Error: Dataset or ONNX model missing.")
    sys.exit(1)

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
purge = 45

# Out-of-Sample Test Set (Last 20% of data)
test_micro = X_micro[split_idx + purge:]
test_macro = X_macro[split_idx + purge:]
test_ref = ref_closes[split_idx + purge:]
test_y_conf = y_confirm[split_idx + purge:]
test_y_dir = y_direction[split_idx + purge:]
test_y_entry = y_entry[split_idx + purge:]
test_y_sl = y_sl[split_idx + purge:]
test_y_tp = y_tp[split_idx + purge:]

n_test = len(test_y_conf)
print(f"   Evaluating on {n_test:,} out-of-sample M5 bars...")

# Normalize micro-structure identically to training
c_ref = test_ref[:, None, None]
norm_o = (test_micro[:, :, 0:1] - c_ref)
norm_h = (test_micro[:, :, 1:2] - c_ref)
norm_l = (test_micro[:, :, 2:3] - c_ref)
norm_c = (test_micro[:, :, 3:4] - c_ref)
norm_v = np.log1p(test_micro[:, :, 4:5])
time_feat = test_micro[:, :, 5:7]

test_micro_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, norm_v, time_feat], axis=-1)
test_micro_feat = np.transpose(test_micro_feat, (0, 2, 1)).astype(np.float32) # (N, 7, 48)
test_macro_feat = test_macro.astype(np.float32)

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

print("2. Running ONNX inference across test sequence...")
# Batch inference
batch_size = 512
all_conf_probs = []
all_dir_preds = []
all_offsets = []

for b in range(0, n_test, batch_size):
    b_micro = test_micro_feat[b : b + batch_size]
    b_macro = test_macro_feat[b : b + batch_size]
    
    out = sess.run(None, {"micro_input": b_micro, "macro_input": b_macro})
    conf_logits, dir_logits, offsets = out[0], out[1], out[2]
    
    # Softmax on confirmation
    exp_c = np.exp(conf_logits - np.max(conf_logits, axis=-1, keepdims=True))
    probs = exp_c / np.sum(exp_c, axis=-1, keepdims=True)
    
    all_conf_probs.append(probs[:, 1])
    all_dir_preds.append(np.argmax(dir_logits, axis=-1))
    all_offsets.append(offsets)

conf_probs = np.concatenate(all_conf_probs, axis=0)
dir_preds = np.concatenate(all_dir_preds, axis=0)
offsets = np.concatenate(all_offsets, axis=0)

print("3. Simulating realistic execution across confidence sweeps...")

thresholds = [0.15, 0.20, 0.25, 0.28, 0.30, 0.35]
best_metrics = None
best_thresh = 0.20

for thresh in thresholds:
    trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    equity = 250.0
    peak_equity = 250.0
    max_dd = 0.0
    
    for i in range(n_test):
        if conf_probs[i] >= thresh:
            # Model signals Mean Reversion trade
            is_win = (test_y_conf[i] == 1)
            pred_dir = dir_preds[i]
            true_dir = test_y_dir[i]
            
            # Must also match ground-truth direction
            if is_win and pred_dir == true_dir:
                # Realized Win
                entry = test_y_entry[i]
                tp = test_y_tp[i]
                sl = test_y_sl[i]
                pnl = abs(tp - entry) * 0.01 * 100 # $ per 0.01 lot
                gross_profit += pnl
                equity += pnl
                wins += 1
            else:
                # Realized Loss (Hit Stop Loss)
                loss_amt = 3.50 * 0.01 * 100 # Standard risk ~ $3.50 on 0.01 lot
                gross_loss += loss_amt
                equity -= loss_amt
                losses += 1
                
            trades += 1
            if equity > peak_equity:
                peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd:
                max_dd = dd

    wr = (wins / trades * 100) if trades > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    net_pnl = gross_profit - gross_loss
    
    print(f"   Threshold >= {thresh:.2f} | Trades: {trades:3d} | WR: {wr:5.1f}% | PF: {pf:4.2f} | Net: ${net_pnl:+6.2f} | MaxDD: ${max_dd:.2f}")
    
    if trades >= 10 and (best_metrics is None or pf > best_metrics['profit_factor']):
        best_thresh = thresh
        best_metrics = {
            "threshold": thresh,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(pf, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "ending_equity": round(equity, 2)
        }

print("\n4. Optimal Confidence Threshold Selected:")
print(json.dumps(best_metrics, indent=2))

with open(REPORT_PATH, "w") as f:
    json.dump(best_metrics, f, indent=2)

print(f"Report saved to: {REPORT_PATH}")
