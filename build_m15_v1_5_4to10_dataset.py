"""
Dataset Generator for 4-to-10 Strategy (v1.5 Series)
---------------------------------------------------
Parameters:
  - Timeframe: M15 Pure OHLC only (6 channels: norm_o, norm_h, norm_l, norm_c, body, range)
  - Lookback: 100 bars relative to Close[100]
  - Logic: 3-Drive Swing Extrema (L1->L2->L3 or H1->H2->H3, 8 to 28 bars spacing)
  - SL: Static $4.00 (Logical stop anchored below/above swing 3 wick)
  - TP: Extended Target $10.00 (1:2.50 payoff)
  - Multi-target scaleout milestone at $4.00 (TP1) -> Breakeven protection
"""
import sys
import os
import json
import numpy as np

DATASET_FILE = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"
OUTPUT_FILE = r"W:\data\trading\xauusd_m15_v1_5_4to10_dataset.npz"

print("1. Loading 99,000 pure M15 historical candles...")
data = np.load(DATASET_FILE)
X_raw = data['X_ohlc'] # (N, 100, 4)
ref_closes = data['ref_closes']

N = len(ref_closes)
print(f"   Loaded {N:,} candles. Scanning 3-Drive Extrema for 4-to-10 Strategy...")

highs = X_raw[:, -1, 1]
lows = X_raw[:, -1, 2]
opens = X_raw[:, -1, 0]
closes = ref_closes

LOOKBACK = 100
HORIZON = 48  # 12 hours forward evaluation
SL_STATIC = 4.00
TP_STATIC = 10.00 # 4 to 10 strategy (1:2.50)

# Precompute 5-bar swing extrema
is_swing_l = np.zeros(N, dtype=bool)
is_swing_h = np.zeros(N, dtype=bool)
for i in range(5, N - 5):
    if lows[i] == np.min(lows[i-5:i+6]):
        is_swing_l[i] = True
    if highs[i] == np.max(highs[i-5:i+6]):
        is_swing_h[i] = True

sw_l_indices = np.where(is_swing_l)[0]
sw_h_indices = np.where(is_swing_h)[0]
print(f"   Extrema detected: {len(sw_l_indices):,} swing lows, {len(sw_h_indices):,} swing highs.")

X_ohlc_list = []
y_action_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
ref_closes_list = []

valid_buys = 0
valid_sells = 0

for i in range(LOOKBACK, N - HORIZON):
    c100 = closes[i]
    ohlc_100 = X_raw[i] # (100, 4)
    
    found_action = 0
    t_entry = 0.0
    t_sl = 0.0
    t_tp = 0.0

    fwd_h = highs[i+1:i+1+HORIZON]
    fwd_l = lows[i+1:i+1+HORIZON]

    # 1. Check if current bar is at/near Swing 3 of a 3-Drive Low pattern
    recent_low_swings = [idx for idx in sw_l_indices if i - 60 <= idx <= i]
    if len(recent_low_swings) >= 3:
        idx3, idx2, idx1 = recent_low_swings[-1], recent_low_swings[-2], recent_low_swings[-3]
        dt12 = idx2 - idx1
        dt23 = idx3 - idx2

        if 8 <= dt12 <= 28 and 8 <= dt23 <= 28 and (i - idx3) <= 4:
            ref_low = min(lows[idx1], lows[idx2])
            if abs(lows[idx3] - ref_low) <= 4.50:
                cand_entry = round(lows[idx3] + 0.40, 2)
                cand_sl = round(lows[idx3] - SL_STATIC, 2)
                cand_tp = round(cand_entry + TP_STATIC, 2)

                if cand_entry < c100:
                    filled = False; hit_tp = False; stopped = False
                    for k in range(len(fwd_h)):
                        if not filled:
                            if fwd_l[k] <= cand_entry:
                                filled = True
                                if fwd_l[k] <= cand_sl: stopped = True; break
                                if fwd_h[k] >= cand_tp: hit_tp = True; break
                        else:
                            if fwd_l[k] <= cand_sl: stopped = True; break
                            if fwd_h[k] >= cand_tp: hit_tp = True; break
                    
                    if filled and hit_tp and not stopped:
                        found_action = 1 # BUY_LIMIT
                        t_entry = cand_entry
                        t_sl = cand_sl
                        t_tp = cand_tp
                        valid_buys += 1

    # 2. Check if current bar is at/near Swing 3 of a 3-Drive High pattern
    if found_action == 0:
        recent_high_swings = [idx for idx in sw_h_indices if i - 60 <= idx <= i]
        if len(recent_high_swings) >= 3:
            idx3, idx2, idx1 = recent_high_swings[-1], recent_high_swings[-2], recent_high_swings[-3]
            dt12 = idx2 - idx1
            dt23 = idx3 - idx2

            if 8 <= dt12 <= 28 and 8 <= dt23 <= 28 and (i - idx3) <= 4:
                ref_high = max(highs[idx1], highs[idx2])
                if abs(highs[idx3] - ref_high) <= 4.50:
                    cand_entry = round(highs[idx3] - 0.40, 2)
                    cand_sl = round(highs[idx3] + SL_STATIC, 2)
                    cand_tp = round(cand_entry - TP_STATIC, 2)

                    if cand_entry > c100:
                        filled = False; hit_tp = False; stopped = False
                        for k in range(len(fwd_h)):
                            if not filled:
                                if fwd_h[k] >= cand_entry:
                                    filled = True
                                    if fwd_h[k] >= cand_sl: stopped = True; break
                                    if fwd_l[k] <= cand_tp: hit_tp = True; break
                            else:
                                if fwd_h[k] >= cand_sl: stopped = True; break
                                if fwd_l[k] <= cand_tp: hit_tp = True; break

                        if filled and hit_tp and not stopped:
                            found_action = 2 # SELL_LIMIT
                            t_entry = cand_entry
                            t_sl = cand_sl
                            t_tp = cand_tp
                            valid_sells += 1

    X_ohlc_list.append(ohlc_100)
    y_action_list.append(found_action)
    y_entry_list.append(t_entry)
    y_sl_list.append(t_sl)
    y_tp_list.append(t_tp)
    ref_closes_list.append(c100)

print(f"2. Compiling 4-to-10 Dataset: {len(y_action_list):,} total samples.")
print(f"   Target $10 Hit BUY setups:  {valid_buys:,}")
print(f"   Target $10 Hit SELL setups: {valid_sells:,}")

np.savez_compressed(
    OUTPUT_FILE,
    X_ohlc=np.array(X_ohlc_list, dtype=np.float32),
    y_action=np.array(y_action_list, dtype=np.int64),
    y_entry=np.array(y_entry_list, dtype=np.float32),
    y_sl=np.array(y_sl_list, dtype=np.float32),
    y_tp=np.array(y_tp_list, dtype=np.float32),
    ref_closes=np.array(ref_closes_list, dtype=np.float32)
)

print(f"Dataset successfully saved to: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.2f} MB)")
