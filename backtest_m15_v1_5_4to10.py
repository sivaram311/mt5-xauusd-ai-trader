"""
Realistic Forward Backtest for 4-to-10 Strategy (v1.5 Series)
-------------------------------------------------------------
Evaluates best_order_net_v1_5_4to10.onnx across the 19,723 out-of-sample forward candles.
Simulates realistic limit order book execution:
- Limit orders placed at predicted entry offset relative to Close[100].
- Static $4.00 SL.
- Extended $10.00 TP.
- 0.25 spread fee included.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort

DATASET_FILE = r"W:\data\trading\xauusd_m15_v1_5_4to10_dataset.npz"
ONNX_PATH = r"W:\data\trading\best_order_net_v1_5_4to10.onnx"
REPORT_PATH = r"W:\data\trading\backtest_reports\backtest_m15_v1_5_4to10.json"

print("1. Loading Out-Of-Sample forward test data...")
data = np.load(DATASET_FILE)
X_ohlc = data['X_ohlc']       # (98853, 100, 4)
ref_closes = data['ref_closes']
N = len(ref_closes)

# Prepare (6, 100) normalized feature matrix
c100 = ref_closes[:, None, None]
norm_o = X_ohlc[:, :, 0:1] - c100
norm_h = X_ohlc[:, :, 1:2] - c100
norm_l = X_ohlc[:, :, 2:3] - c100
norm_c = X_ohlc[:, :, 3:4] - c100
body = X_ohlc[:, :, 3:4] - X_ohlc[:, :, 0:1]
rng = X_ohlc[:, :, 1:2] - X_ohlc[:, :, 2:3]

X_feat = np.concatenate([norm_o, norm_h, norm_l, norm_c, body, rng], axis=-1)
X_feat = np.transpose(X_feat, (0, 2, 1)).astype(np.float32)

split = int(N * 0.80)
purge = 48
test_start = split + purge

highs = X_raw_h = X_ohlc[:, -1, 1]
lows = X_raw_l = X_ohlc[:, -1, 2]
closes = ref_closes

print(f"   Forward Test Window: Candle {test_start:,} to {N:,} ({N - test_start:,} unseen M15 candles)")

session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

trades = []
HORIZON = 48
SPREAD_FEE = 0.25

print("2. Running forward inference and realistic execution simulation...")
for i in range(test_start, N - HORIZON):
    x_input = X_feat[i:i+1] # (1, 6, 100)
    outputs = session.run(None, {'input': x_input})
    act_logits = outputs[0][0]
    offsets = outputs[1][0] # [entry_off, sl_off, tp_off]
    
    pred_action = int(np.argmax(act_logits))
    if pred_action == 0:
        continue

    c_now = closes[i]
    pred_entry = round(c_now + float(offsets[0]), 2)
    
    # Static $4.00 SL, extended $10.00 TP
    if pred_action == 1: # BUY_LIMIT
        pred_sl = round(pred_entry - 4.00, 2)
        pred_tp = round(pred_entry + 10.00, 2)
        if pred_entry >= c_now:
            continue
    elif pred_action == 2: # SELL_LIMIT
        pred_sl = round(pred_entry + 4.00, 2)
        pred_tp = round(pred_entry - 10.00, 2)
        if pred_entry <= c_now:
            continue
    else:
        continue

    fwd_h = highs[i+1:i+1+HORIZON]
    fwd_l = lows[i+1:i+1+HORIZON]
    
    filled = False
    hit_tp = False
    stopped = False
    bars_held = 0

    for k in range(len(fwd_h)):
        bars_held = k + 1
        if not filled:
            if pred_action == 1 and fwd_l[k] <= pred_entry:
                filled = True
                if fwd_l[k] <= pred_sl: stopped = True; break
                if fwd_h[k] >= pred_tp: hit_tp = True; break
            elif pred_action == 2 and fwd_h[k] >= pred_entry:
                filled = True
                if fwd_h[k] >= pred_sl: stopped = True; break
                if fwd_l[k] <= pred_tp: hit_tp = True; break
        else:
            if pred_action == 1:
                if fwd_l[k] <= pred_sl: stopped = True; break
                if fwd_h[k] >= pred_tp: hit_tp = True; break
            elif pred_action == 2:
                if fwd_h[k] >= pred_sl: stopped = True; break
                if fwd_l[k] <= pred_tp: hit_tp = True; break

    if filled:
        pnl = 0.0
        if hit_tp:
            pnl = 10.00 - SPREAD_FEE
        elif stopped:
            pnl = -4.00 - SPREAD_FEE
        else:
            exit_price = closes[i+1+HORIZON-1]
            if pred_action == 1:
                pnl = (exit_price - pred_entry) - SPREAD_FEE
            else:
                pnl = (pred_entry - exit_price) - SPREAD_FEE
        
        trades.append({
            "bar": int(i),
            "action": "BUY" if pred_action == 1 else "SELL",
            "entry": float(pred_entry),
            "sl": float(pred_sl),
            "tp": float(pred_tp),
            "pnl": float(pnl),
            "win": bool(pnl > 0),
            "bars_held": int(bars_held)
        })

total_trades = len(trades)
wins = [t for t in trades if t['win']]
losses = [t for t in trades if not t['win']]
win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
gross_profit = sum(t['pnl'] for t in wins)
gross_loss = abs(sum(t['pnl'] for t in losses))
profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
net_pnl = gross_profit - gross_loss

print("\n" + "="*50)
print("=== FORWARD TEST RESULTS: 4-TO-10 STRATEGY (v1.5) ===")
print("="*50)
print(f"Total Limit Trades Filled: {total_trades:,}")
print(f"Winning Trades:            {len(wins):,}")
print(f"Losing Trades:             {len(losses):,}")
print(f"Win Rate:                  {win_rate:.2f}%")
print(f"Profit Factor:             {profit_factor:.2f}")
print(f"Net PnL (USD per 1.0 lot): ${net_pnl:,.2f}")
print("="*50)

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, 'w') as f:
    json.dump({
        "strategy": "4_to_10_pure_ohlc_m15_v1_5",
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(float(win_rate), 2),
        "profit_factor": round(float(profit_factor), 2),
        "net_pnl": round(float(net_pnl), 2),
        "sample_trades": trades[:20]
    }, f, indent=2)

print(f"Report saved to: {REPORT_PATH}")
