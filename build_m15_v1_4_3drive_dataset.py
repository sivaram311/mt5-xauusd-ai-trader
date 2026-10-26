"""
Dataset Generator for 3-Drive Swing Liquidity Model (v1.4 Series)
-----------------------------------------------------------------
Features: Exact 6 pure-OHLC relative channels:
    0: norm_o = Open - c100
    1: norm_h = High - c100
    2: norm_l = Low  - c100
    3: norm_c = Close - c100
    4: body   = Close - Open
    5: range  = High  - Low
Lookback: 100 historical M15 candles (captures full 27-28 bar 3-drive patterns).
Targets:
    - Anchored on 3-drive swing sweeps (L1->L2->L3 or H1->H2->H3).
    - SL: Static $4.00 (Immune to Gold wick sweeps).
    - TP: Dynamic 1:2.0 to 1:2.5 Mean Reversion ($8.00 to $10.00).
    - Max Hold: 32 bars (8 hours).
"""
import sys
import os
import json
import numpy as np
import pandas as pd

DATASET_FILE = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"
OUTPUT_FILE = r"W:\data\trading\xauusd_m15_v1_4_3drive_dataset.npz"

print("1. Loading 99,000 pure M15 historical candles...")
data = np.load(DATASET_FILE)
X_raw = data['X_ohlc'] # (N, 100, 4)
ref_closes = data['ref_closes']
timestamps = data.get('end_times', [str(i) for i in range(len(ref_closes))])

N = len(ref_closes)
print(f"   Loaded {N:,} candles. Scanning 3-Drive Swing patterns...")

# Reconstruct continuous high/lows
highs = X_raw[:, -1, 1]
lows = X_raw[:, -1, 2]
opens = X_raw[:, -1, 0]
closes = ref_closes

LOOKBACK = 100
HORIZON = 32
SL_STATIC = 4.00
TP_STATIC = 8.50 # 1:2.125 R:R
SPREAD_FEE = 0.25

# Identify 5-bar swing extrema
swing_low_indices = []
for i in range(5, N - 5):
    if lows[i] == min(lows[i-5:i+6]):
        swing_low_indices.append(i)

swing_high_indices = []
for i in range(5, N - 5):
    if highs[i] == max(highs[i-5:i+6]):
        swing_high_indices.append(i)

swing_low_set = set(swing_low_indices)
swing_high_set = set(swing_high_indices)

print(f"   Identified {len(swing_low_indices):,} swing lows and {len(swing_high_indices):,} swing highs.")

X_ohlc_list = []
y_action_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
ref_closes_list = []

valid_buys = 0
valid_sells = 0

for i in range(LOOKBACK - 1, N - HORIZON):
    c100 = closes[i]

    # Lookback window (100, 4) pure OHLC
    ohlc_100 = X_raw[i]

    fwd_h = highs[i + 1 : i + 1 + HORIZON]
    fwd_l = lows[i + 1 : i + 1 + HORIZON]

    found_action = 0
    t_entry = 0.0
    t_sl = 0.0
    t_tp = 0.0

    # 1. Check if current candle forms Swing 3 of a 3-Drive Low Pattern
    # Look back for prior 2 swing lows in recent 60 bars (spacing 8 to 28 bars)
    recent_low_swings = [idx for idx in swing_low_indices if i - 60 <= idx <= i]
    
    if len(recent_low_swings) >= 3:
        idx3, idx2, idx1 = recent_low_swings[-1], recent_low_swings[-2], recent_low_swings[-3]
        dt12 = idx2 - idx1
        dt23 = idx3 - idx2
        
        # Spacing matches empirical 8-25 bar rhythms (approx 2 to 6 hours between taps)
        if 8 <= dt12 <= 28 and 8 <= dt23 <= 28 and (i - idx3) <= 4:
            # L3 is near or sweeping L1/L2
            ref_low = min(lows[idx1], lows[idx2])
            if abs(lows[idx3] - ref_low) <= 4.50:
                cand_entry = round(lows[idx3] + 0.40, 2)
                cand_sl = round(lows[idx3] - SL_STATIC, 2)
                cand_tp = round(cand_entry + TP_STATIC, 2)

                if cand_entry < c100:
                    filled = False; hit_tp = False; stopped = False
                    for k in range(len(fwd_l)):
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

    # 2. Check if current candle forms Swing 3 of a 3-Drive High Pattern
    if found_action == 0:
        recent_high_swings = [idx for idx in swing_high_indices if i - 60 <= idx <= i]
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

print(f"2. Compiling 3-Drive Dataset: {len(y_action_list):,} total samples.")
print(f"   High-Quality 3-Drive BUY setups:  {valid_buys:,}")
print(f"   High-Quality 3-Drive SELL setups: {valid_sells:,}")

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
