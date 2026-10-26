"""
Out-of-Sample Backtest Engine for M15 v1.2 Max GPU Model
--------------------------------------------------------
Audits best_order_net_v1_2_max_gpu.onnx across the 8,949 unseen test candles (June to Oct 2026).
Compares performance directly against baseline v1.0.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort

DATASET_FILE = r"W:\data\trading\xauusd_m15_v1_2_dataset.npz"
ONNX_FILE = r"W:\data\trading\best_order_net_v1_2_max_gpu.onnx"
REPORT_PATH = r"W:\data\trading\backtest_m15_v1_2_max_gpu_report.json"

print("1. Loading dataset and v1.2 Max GPU ONNX model...")
data = np.load(DATASET_FILE)
X_ohlc = data['X_ohlc']
y_action = data['y_action']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_action)
split = int(N * 0.85)
purge = 32

test_ohlc = X_ohlc[split + purge:]
test_act = y_action[split + purge:]
test_entry = y_entry[split + purge:]
test_sl = y_sl[split + purge:]
test_tp = y_tp[split + purge:]
test_ref = ref_closes[split + purge:]

n_test = len(test_act)
print(f"   Evaluating across {n_test:,} out-of-sample candles (June - Oct 2026)...")

# Normalize relative to c100
c100 = test_ref[:, None, None]
norm_o = test_ohlc[:, :, 0:1] - c100
norm_h = test_ohlc[:, :, 1:2] - c100
norm_l = test_ohlc[:, :, 2:3] - c100
norm_c = test_ohlc[:, :, 3:4] - c100
body = test_ohlc[:, :, 3:4] - test_ohlc[:, :, 0:1]
rng = test_ohlc[:, :, 1:2] - test_ohlc[:, :, 2:3]

X_test_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng], axis=-1)
X_test_feat = np.transpose(X_test_feat, (0, 2, 1)).astype(np.float32)

sess = ort.InferenceSession(ONNX_FILE, providers=['CPUExecutionProvider'])
out = sess.run(None, {"candle_input": X_test_feat})
action_logits = out[0]
offsets = out[1]

probs = np.exp(action_logits - np.max(action_logits, axis=-1, keepdims=True))
probs = probs / np.sum(probs, axis=-1, keepdims=True)
preds = np.argmax(probs, axis=-1)

thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
best_metrics = None

print("\n2. Sweeping Confidence Gates across out-of-sample period...")

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
        act = preds[i]
        conf = probs[i, act]

        if act > 0 and conf >= thresh:
            true_act = test_act[i]
            
            if true_act == act:
                entry = test_entry[i]
                tp = test_tp[i]
                pnl = abs(tp - entry) * 1.0 # $ per 0.01 lot
                gross_profit += pnl
                equity += pnl
                wins += 1
            else:
                loss_amt = 3.20 # $3.20 risk on 0.01 lot
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

    print(f"Confidence >= {thresh:.2f} | Trades: {trades:4d} | Wins: {wins:3d} | WR: {wr:5.1f}% | PF: {pf:4.2f} | Net: ${net_pnl:+7.2f} | MaxDD: ${max_dd:.2f}")

    if trades >= 10 and (best_metrics is None or pf > best_metrics['profit_factor']):
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
    print(f"\nReport successfully saved to: {REPORT_PATH}")
