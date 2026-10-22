"""
Dataset Generator for Raw Price 1:5 Asymmetric M1 Liquidity Sniper
Inputs:
- 120 Raw M1 Candles: [Open, High, Low, Close, Minute (0-59), Hour (0-23), DayOfWeek (0-4)]
- No volume, no normalization.

Outputs (Multi-Head Ground Truth):
1. confirm: 0 (No trade) or 1 (Valid setup)
2. direction: 0 (BUY_LIMIT) or 1 (SELL_LIMIT)
3. entry_price: Raw float
4. sl_price: Raw float (tight structural stop)
5. tp_price: Raw float (strictly >= 5.0x Risk, accounting for $0.35 broker spread)
6. time_to_fill: Minutes (1 to 25) until the pending order was picked up by market wicks.
"""
import sys
import os
import time
import datetime
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
N_BARS = 85000           # ~60 trading days of M1 history
LOOKBACK_WINDOW = 120    # 120 M1 bars lookback (2 hours)
CONFIRMATION_MAX = 7     # Confirmation window: 1 to 7 candles
MAX_FILL_WINDOW = 25     # Order must fill within 25 minutes
MAX_TRADE_HORIZON = 60   # Max trade lifespan to hit TP: 60 minutes
SPREAD_FEE = 0.35        # Real broker spread ($0.35)
MIN_RISK = 1.40          # Minimum structural risk points to prevent spread eating stop
MAX_RISK = 3.50          # Maximum tight stop loss risk points
RR_TARGET = 5.0          # Strict minimum 1:5 Risk-to-Reward

SAVE_PATH_NPZ = r"W:\data\trading\xauusd_m1_raw_price_1to5_dataset.npz"
SAVE_PATH_PREVIEW = r"W:\data\trading\m1_raw_price_1to5_preview.json"

print("1. Connecting to MT5 and fetching historical M1 candles...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, N_BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("Error: Could not retrieve rates.")
    sys.exit(1)

print(f"   Successfully fetched {len(rates):,} M1 bars.")
df = pd.DataFrame(rates)
df['dt'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('dt').reset_index(drop=True)

opens = df['open'].values.astype(np.float32)
highs = df['high'].values.astype(np.float32)
lows = df['low'].values.astype(np.float32)
closes = df['close'].values.astype(np.float32)
minutes = df['dt'].dt.minute.values.astype(np.float32)
hours = df['dt'].dt.hour.values.astype(np.float32)
dows = df['dt'].dt.dayofweek.values.astype(np.float32)

n_total = len(df)
print("2. Constructing raw inputs and scanning asymmetric 1:5 pending order outcomes...")

X_raw_list = []
y_confirm_list = []
y_direction_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
y_time_to_fill_list = []
preview_samples = []

start_idx = LOOKBACK_WINDOW
end_idx = n_total - MAX_TRADE_HORIZON - 10

# Scan candles
for i in range(start_idx, end_idx):
    # Raw input window: (120, 7)
    # [Open, High, Low, Close, Minute, Hour, DayOfWeek]
    w_o = opens[i - LOOKBACK_WINDOW + 1 : i + 1]
    w_h = highs[i - LOOKBACK_WINDOW + 1 : i + 1]
    w_l = lows[i - LOOKBACK_WINDOW + 1 : i + 1]
    w_c = closes[i - LOOKBACK_WINDOW + 1 : i + 1]
    w_m = minutes[i - LOOKBACK_WINDOW + 1 : i + 1]
    w_hr = hours[i - LOOKBACK_WINDOW + 1 : i + 1]
    w_dow = dows[i - LOOKBACK_WINDOW + 1 : i + 1]

    raw_feat = np.stack([w_o, w_h, w_l, w_c, w_m, w_hr, w_dow], axis=-1) # (120, 7)

    c_now = closes[i]

    # Future lookahead window for trade resolution
    fwd_h = highs[i + 1 : i + 1 + MAX_TRADE_HORIZON]
    fwd_l = lows[i + 1 : i + 1 + MAX_TRADE_HORIZON]

    found_trade = False
    trade_meta = None

    # Check for BUY_LIMIT setup:
    # Liquidity sweep/swing low within recent 3-5 candles
    recent_low_idx = np.argmin(w_l[-5:])
    swing_low = float(w_l[-5:][recent_low_idx])
    
    # Target pending entry slightly above swing low
    cand_entry = round(swing_low + 0.15, 2)
    # Target tight SL just below swing low + spread buffer
    cand_sl = round(swing_low - 1.50 - SPREAD_FEE, 2)
    risk_pts = round(cand_entry - cand_sl, 2)

    if MIN_RISK <= risk_pts <= MAX_RISK and cand_entry < c_now:
        cand_tp = round(cand_entry + (risk_pts * RR_TARGET) + SPREAD_FEE, 2)
        
        # Simulate pending order pickup and outcome
        filled = False
        time_to_fill = 0
        stopped = False
        hit_tp = False

        for k in range(len(fwd_l)):
            if not filled:
                if k < MAX_FILL_WINDOW and fwd_l[k] <= cand_entry:
                    filled = True
                    time_to_fill = k + 1  # 1 to 25 minutes
                    # Check if bar also blew through stop immediately
                    if fwd_l[k] <= cand_sl:
                        stopped = True
                        break
                    if fwd_h[k] >= cand_tp:
                        hit_tp = True
                        break
                elif k >= MAX_FILL_WINDOW:
                    # Order expired before being picked up
                    break
            else:
                # Trade is active, check SL/TP
                if fwd_l[k] <= cand_sl:
                    stopped = True
                    break
                if fwd_h[k] >= cand_tp:
                    hit_tp = True
                    break

        if filled and hit_tp and not stopped:
            found_trade = True
            trade_meta = {
                "confirm": 1,
                "direction": 0, # BUY_LIMIT
                "entry": cand_entry,
                "sl": cand_sl,
                "tp": cand_tp,
                "time_to_fill": float(time_to_fill),
                "risk": risk_pts,
                "reward": round(cand_tp - cand_entry, 2),
                "rr": round((cand_tp - cand_entry) / risk_pts, 2)
            }

    # If no buy, check SELL_LIMIT setup:
    if not found_trade:
        recent_high_idx = np.argmax(w_h[-5:])
        swing_high = float(w_h[-5:][recent_high_idx])
        cand_entry = round(swing_high - 0.15, 2)
        cand_sl = round(swing_high + 1.50 + SPREAD_FEE, 2)
        risk_pts = round(cand_sl - cand_entry, 2)

        if MIN_RISK <= risk_pts <= MAX_RISK and cand_entry > c_now:
            cand_tp = round(cand_entry - (risk_pts * RR_TARGET) - SPREAD_FEE, 2)

            filled = False
            time_to_fill = 0
            stopped = False
            hit_tp = False

            for k in range(len(fwd_h)):
                if not filled:
                    if k < MAX_FILL_WINDOW and fwd_h[k] >= cand_entry:
                        filled = True
                        time_to_fill = k + 1
                        if fwd_h[k] >= cand_sl:
                            stopped = True
                            break
                        if fwd_l[k] <= cand_tp:
                            hit_tp = True
                            break
                    elif k >= MAX_FILL_WINDOW:
                        break
                else:
                    if fwd_h[k] >= cand_sl:
                        stopped = True
                        break
                    if fwd_l[k] <= cand_tp:
                        hit_tp = True
                        break

            if filled and hit_tp and not stopped:
                found_trade = True
                trade_meta = {
                    "confirm": 1,
                    "direction": 1, # SELL_LIMIT
                    "entry": cand_entry,
                    "sl": cand_sl,
                    "tp": cand_tp,
                    "time_to_fill": float(time_to_fill),
                    "risk": risk_pts,
                    "reward": round(cand_entry - cand_tp, 2),
                    "rr": round((cand_entry - cand_tp) / risk_pts, 2)
                }

    X_raw_list.append(raw_feat)

    if found_trade and trade_meta is not None:
        y_confirm_list.append(1)
        y_direction_list.append(trade_meta["direction"])
        y_entry_list.append(trade_meta["entry"])
        y_sl_list.append(trade_meta["sl"])
        y_tp_list.append(trade_meta["tp"])
        y_time_to_fill_list.append(trade_meta["time_to_fill"])
        if len(preview_samples) < 5:
            preview_samples.append({
                "bar_idx": i,
                "current_close": float(c_now),
                "trade": trade_meta
            })
    else:
        y_confirm_list.append(0)
        y_direction_list.append(0)
        y_entry_list.append(float(c_now))
        y_sl_list.append(float(c_now))
        y_tp_list.append(float(c_now))
        y_time_to_fill_list.append(0.0)

    if (i - start_idx + 1) % 20000 == 0:
        pos_cnt = sum(y_confirm_list)
        print(f"   Processed {i - start_idx + 1:,} bars | Positive 1:5 setups found: {pos_cnt:,} ({pos_cnt / len(y_confirm_list) * 100:.2f}%)")

X_raw = np.array(X_raw_list, dtype=np.float32)
y_confirm = np.array(y_confirm_list, dtype=np.int64)
y_direction = np.array(y_direction_list, dtype=np.int64)
y_entry = np.array(y_entry_list, dtype=np.float32)
y_sl = np.array(y_sl_list, dtype=np.float32)
y_tp = np.array(y_tp_list, dtype=np.float32)
y_time_to_fill = np.array(y_time_to_fill_list, dtype=np.float32)

print("3. Saving dataset to compressed npz...")
np.savez_compressed(
    SAVE_PATH_NPZ,
    X_raw=X_raw,
    y_confirm=y_confirm,
    y_direction=y_direction,
    y_entry=y_entry,
    y_sl=y_sl,
    y_tp=y_tp,
    y_time_to_fill=y_time_to_fill
)

with open(SAVE_PATH_PREVIEW, "w") as f:
    json.dump({
        "total_samples": len(y_confirm),
        "positive_setups": int((y_confirm == 1).sum()),
        "positive_pct": float(round((y_confirm == 1).sum() / len(y_confirm) * 100, 2)),
        "samples_preview": preview_samples
    }, f, indent=2)

print(f"Done! Dataset saved to {SAVE_PATH_NPZ}")
print(f"Total samples: {len(y_confirm):,} | 1:5 Trades: {(y_confirm == 1).sum():,} ({round((y_confirm == 1).sum()/len(y_confirm)*100, 2)}%)")
