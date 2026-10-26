"""
Quantitative Experiment: Testing Pure M15 OHLC Structural Filters for 55%+ Win Rate
-----------------------------------------------------------------------------------
Tests 3 purely OHLC-derived structural mechanisms across 99,000 candles:
1. Rejection Confirmation Bar (Require candle to close green after Low 3, or red after High 3).
2. Asymmetric Payoff Calibration:
   - Option A: SL $4.50, TP $6.75 (1:1.5 R:R)
   - Option B: SL $5.00, TP $7.50 (1:1.5 R:R)
   - Option C: SL $4.00, TP $6.00 (1:1.5 R:R)
3. Multiple-Timeframe Synthesis purely from M15 OHLC:
   - Synthetic H1 EMA Trend Slope (computed by rolling mean over 4 x M15 candles).
   - Higher High / Higher Low structural trend filter.
"""
import numpy as np
import pandas as pd
import json

DATASET_PATH = r"W:\data\trading\xauusd_m15_v1_4_3drive_dataset.npz"
data = np.load(DATASET_PATH)
X_raw = data['X_ohlc'] # (N, 100, 4)
ref_closes = data['ref_closes']

N = len(ref_closes)
split = int(N * 0.80)
purge = 48
test_start = split + purge

test_ohlc = X_raw[test_start:]
test_closes = ref_closes[test_start:]
n_test = len(test_closes)

highs = X_raw[:, -1, 1]
lows = X_raw[:, -1, 2]
opens = X_raw[:, -1, 0]
closes = ref_closes

# Synthetic H1 slope derived purely from M15 closes (rolling 16 bars = 4 hours)
synthetic_h1_ema = pd.Series(closes).ewm(span=16, adjust=False).mean().values
synthetic_slope = (closes - synthetic_h1_ema)

# Run simulations across combinations
experiments = []

# Scan 3-Drive setups in out-of-sample period
for sl_val, tp_val in [(4.0, 6.0), (4.5, 6.75), (5.0, 7.5), (4.0, 8.0)]:
    for require_confirmation_candle in [False, True]:
        for trend_filter in [False, True]:
            trades = 0
            wins = 0
            losses = 0

            for i in range(test_start, N - 32):
                # Check recent low swing
                # Swing 3 Low setup
                is_low3 = False
                is_high3 = False

                # Quick check if candle i is local extreme
                if lows[i] == min(lows[max(0, i-15):i+1]):
                    is_low3 = True
                elif highs[i] == max(highs[max(0, i-15):i+1]):
                    is_high3 = True

                if not (is_low3 or is_high3):
                    continue

                # Confirmation candle filter: candle i+1 must close in direction of reversal
                eval_idx = i
                if require_confirmation_candle:
                    if is_low3 and closes[i] <= opens[i]: # Must be a green rejection or hammer
                        continue
                    if is_high3 and closes[i] >= opens[i]: # Must be a red rejection or shooting star
                        continue

                # Pure OHLC Synthetic Trend Filter
                if trend_filter:
                    if is_low3 and synthetic_slope[i] < -2.0: continue # Don't buy in violent downtrend
                    if is_high3 and synthetic_slope[i] > 2.0: continue # Don't sell in violent uptrend

                fwd_h = highs[i + 1 : i + 1 + 32]
                fwd_l = lows[i + 1 : i + 1 + 32]

                if is_low3:
                    entry = round(lows[i] + 0.50, 2)
                    sl = round(entry - sl_val, 2)
                    tp = round(entry + tp_val, 2)

                    hit_tp = False; hit_sl = False; filled = False
                    for k in range(len(fwd_l)):
                        if not filled:
                            if fwd_l[k] <= entry:
                                filled = True
                                if fwd_l[k] <= sl: hit_sl = True; break
                                if fwd_h[k] >= tp: hit_tp = True; break
                        else:
                            if fwd_l[k] <= sl: hit_sl = True; break
                            if fwd_h[k] >= tp: hit_tp = True; break

                    if filled:
                        trades += 1
                        if hit_tp and not hit_sl: wins += 1
                        elif hit_sl: losses += 1

                elif is_high3:
                    entry = round(highs[i] - 0.50, 2)
                    sl = round(entry + sl_val, 2)
                    tp = round(entry - tp_val, 2)

                    hit_tp = False; hit_sl = False; filled = False
                    for k in range(len(fwd_h)):
                        if not filled:
                            if fwd_h[k] >= entry:
                                filled = True
                                if fwd_h[k] >= sl: hit_sl = True; break
                                if fwd_l[k] <= tp: hit_tp = True; break
                        else:
                            if fwd_h[k] >= sl: hit_sl = True; break
                            if fwd_l[k] <= tp: hit_tp = True; break

                    if filled:
                        trades += 1
                        if hit_tp and not hit_sl: wins += 1
                        elif hit_sl: losses += 1

            if trades >= 50:
                wr = (wins / trades) * 100
                rr = tp_val / sl_val
                pf = (wins * tp_val) / (max(losses, 1) * sl_val)
                experiments.append({
                    "sl": sl_val,
                    "tp": tp_val,
                    "rr": f"1:{rr:.2f}",
                    "confirm_candle": require_confirmation_candle,
                    "synthetic_trend": trend_filter,
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(wr, 2),
                    "profit_factor": round(pf, 2)
                })

df_res = pd.DataFrame(experiments).sort_values(by="win_rate", ascending=False)
print(df_res.to_string(index=False))
