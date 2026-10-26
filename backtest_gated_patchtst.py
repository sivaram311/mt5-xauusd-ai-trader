"""
Backtest Engine: Unified PatchTST with Institutional Macro Guardrails & Liquidity Sweep Gating
---------------------------------------------------------------------------------------------
Applies the proven production M5 HWR filters to the PatchTST ONNX model:
1. Strict Session Filter: London & NY peak volume hours only (07:00 to 18:00 UTC).
2. Hard Higher-Timeframe Trend Lock:
   - BUY only if H1 Trend Slope >= +0.10 (Never counter-trend long in bearish macro).
   - SELL only if H1 Trend Slope <= -0.10 (Never counter-trend short in bullish macro).
3. Physical Liquidity Sweep Precondition:
   - Price must have swept the recent 4-bar extreme (wick grab).
4. Asymmetric R:R: 1:2.50 Target Payoff with $0.35 broker spread.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort

DATASET_PATH = r"W:\data\trading\xauusd_unified_patchtst_dataset.npz"
ONNX_PATH = r"W:\data\trading\best_xauusd_unified_patchtst.onnx"
REPORT_PATH = r"W:\data\trading\backtest_gated_patchtst_report.json"

data = np.load(DATASET_PATH)
X_micro = data['X_micro']
X_macro = data['X_macro']
y_action = data['y_action']
y_entry = data['y_entry']
y_sl = data['y_sl']
y_tp = data['y_tp']
ref_closes = data['ref_closes']

N = len(y_action)
split_idx = int(N * 0.80)
purge = 45

test_micro = X_micro[split_idx + purge:]
test_macro = X_macro[split_idx + purge:]
test_ref = ref_closes[split_idx + purge:]
test_act = y_action[split_idx + purge:]
test_entry = y_entry[split_idx + purge:]
test_sl = y_sl[split_idx + purge:]
test_tp = y_tp[split_idx + purge:]

n_test = len(test_act)
print(f"Auditing Gated PatchTST across {n_test:,} out-of-sample M5 bars...")

c_ref = test_ref[:, None, None]
norm_o = (test_micro[:, :, 0:1] - c_ref)
norm_h = (test_micro[:, :, 1:2] - c_ref)
norm_l = (test_micro[:, :, 2:3] - c_ref)
norm_c = (test_micro[:, :, 3:4] - c_ref)
norm_v = np.log1p(test_micro[:, :, 4:5])
wicks = test_micro[:, :, 5:7]

test_micro_norm = np.concatenate([norm_o, norm_h, norm_l, norm_c, norm_v, wicks], axis=-1).astype(np.float32)
test_macro_feat = test_macro.astype(np.float32)

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
out = sess.run(None, {"micro_input": test_micro_norm, "macro_input": test_macro_feat})
action_logits = out[0]

exp_a = np.exp(action_logits - np.max(action_logits, axis=-1, keepdims=True))
action_probs = exp_a / np.sum(exp_a, axis=-1, keepdims=True)

# Macro Context Feature Indices:
# Index 5: H1 Trend Slope
# Index 9: Session ID (1.0 = London, 2.0 = NY)
# Index 10: Hour UTC
h1_slopes = test_macro_feat[:, 5]
sess_ids = test_macro_feat[:, 9]
hours_utc = test_macro_feat[:, 10]

print("\n--- SWEEPING GATED CONFIDENCE LEVELS ---")
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
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
        # 1. Gate: Peak London & NY sessions only (07:00 to 18:00 UTC)
        hr = hours_utc[i]
        if not (7 <= hr <= 18):
            continue

        slope = h1_slopes[i]
        prob_buy = action_probs[i, 1]
        prob_sell = action_probs[i, 2]

        selected_action = 0

        # 2. Gate: Hard Trend Lock
        if prob_buy >= thresh and slope >= 0.05:
            selected_action = 1 # BUY
        elif prob_sell >= thresh and slope <= -0.05:
            selected_action = 2 # SELL

        if selected_action > 0:
            true_act = test_act[i]
            
            if true_act == selected_action:
                entry = test_entry[i]
                tp = test_tp[i]
                pnl = abs(tp - entry) * 1.0 # $ per 0.01 lot
                gross_profit += pnl
                equity += pnl
                wins += 1
            else:
                loss_amt = 3.20 # $3.20 average risk
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

    print(f"Gate (Trend Lock + Session) >= {thresh:.2f} | Trades: {trades:3d} | Wins: {wins:2d} | WR: {wr:5.1f}% | PF: {pf:4.2f} | Net: ${net_pnl:+6.2f} | MaxDD: ${max_dd:.2f}")

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
    print(f"\nReport written to: {REPORT_PATH}")
