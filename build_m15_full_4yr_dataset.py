"""
Full 4.2-Year Dataset Generator for XAUUSD M15 (v1.3 Series)
------------------------------------------------------------
- Full MT5 Depth: 99,000 Candles (Aug 2022 to Oct 2026)
- 7 Feature Channels:
    0: norm_open  (Open - c100)
    1: norm_high  (High - c100)
    2: norm_low   (Low  - c100)
    3: norm_close (Close - c100)
    4: body       (Close - Open)
    5: range      (High  - Low)
    6: volume     (log1p(tick_volume))
- Ground Truth: Asymmetric Pending Limits (R:R >= 1:3 within 32 forward bars)
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100         # 100 bars lookback
HORIZON = 32           # 32 bars (8 hours) forward horizon
MIN_RISK = 2.00        # $2.00 min SL
MAX_RISK = 6.00        # $6.00 max SL
TARGET_RR = 3.00       # 1:3.00 R:R
MAX_FILL_BARS = 6      # 1.5-hour fill window
MAX_BARS = 99000       # Full 4.2-year history

OUTPUT_FILE = r"W:\data\trading\xauusd_m15_full_4yr_dataset.npz"

print("1. Connecting to MT5 and fetching full 99,000 M15 history...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, MAX_BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("Error: Could not retrieve rates from MT5.")
    sys.exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].sort_values('time').reset_index(drop=True)
print(f"   Retrieved {len(df):,} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

highs = df['high'].values.astype(np.float32)
lows = df['low'].values.astype(np.float32)
opens = df['open'].values.astype(np.float32)
closes = df['close'].values.astype(np.float32)
volumes = df['tick_volume'].values.astype(np.float32)
times = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S').values

n_bars = len(df)
print(f"2. Scanning 4.2-year historical windows (Lookback={LOOKBACK}, Horizon={HORIZON})...")

X_raw_list = []
y_action_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
ref_closes_list = []
timestamps_list = []

valid_buys = 0
valid_sells = 0

for i in range(LOOKBACK - 1, n_bars - HORIZON):
    c100 = closes[i]

    win_o = opens[i - LOOKBACK + 1 : i + 1]
    win_h = highs[i - LOOKBACK + 1 : i + 1]
    win_l = lows[i - LOOKBACK + 1 : i + 1]
    win_c = closes[i - LOOKBACK + 1 : i + 1]
    win_v = volumes[i - LOOKBACK + 1 : i + 1]

    # Shape: (100, 5) -> [O, H, L, C, V]
    raw_100 = np.stack([win_o, win_h, win_l, win_c, win_v], axis=-1)

    fwd_h = highs[i + 1 : i + 1 + HORIZON]
    fwd_l = lows[i + 1 : i + 1 + HORIZON]

    found_action = 0
    t_entry = 0.0
    t_sl = 0.0
    t_tp = 0.0

    # 1. Test BUY_LIMIT
    recent_low = float(min(win_l[-4:]))
    cand_entry = round(recent_low + 0.30, 2)
    cand_sl = round(recent_low - 2.50, 2)
    risk = cand_entry - cand_sl

    if MIN_RISK <= risk <= MAX_RISK and cand_entry < c100:
        cand_tp = round(cand_entry + (risk * TARGET_RR), 2)
        filled = False
        hit_tp = False
        stopped = False

        for k in range(len(fwd_l)):
            if not filled:
                if k < MAX_FILL_BARS and fwd_l[k] <= cand_entry:
                    filled = True
                    if fwd_l[k] <= cand_sl: stopped = True; break
                    if fwd_h[k] >= cand_tp: hit_tp = True; break
                elif k >= MAX_FILL_BARS: break
            else:
                if fwd_l[k] <= cand_sl: stopped = True; break
                if fwd_h[k] >= cand_tp: hit_tp = True; break

        if filled and hit_tp and not stopped:
            found_action = 1
            t_entry = cand_entry
            t_sl = cand_sl
            t_tp = cand_tp
            valid_buys += 1

    # 2. Test SELL_LIMIT
    if found_action == 0:
        recent_high = float(max(win_h[-4:]))
        cand_entry = round(recent_high - 0.30, 2)
        cand_sl = round(recent_high + 2.50, 2)
        risk = cand_sl - cand_entry

        if MIN_RISK <= risk <= MAX_RISK and cand_entry > c100:
            cand_tp = round(cand_entry - (risk * TARGET_RR), 2)
            filled = False
            hit_tp = False
            stopped = False

            for k in range(len(fwd_h)):
                if not filled:
                    if k < MAX_FILL_BARS and fwd_h[k] >= cand_entry:
                        filled = True
                        if fwd_h[k] >= cand_sl: stopped = True; break
                        if fwd_l[k] <= cand_tp: hit_tp = True; break
                    elif k >= MAX_FILL_BARS: break
                else:
                    if fwd_h[k] >= cand_sl: stopped = True; break
                    if fwd_l[k] <= cand_tp: hit_tp = True; break

            if filled and hit_tp and not stopped:
                found_action = 2
                t_entry = cand_entry
                t_sl = cand_sl
                t_tp = cand_tp
                valid_sells += 1

    X_raw_list.append(raw_100)
    y_action_list.append(found_action)
    y_entry_list.append(t_entry)
    y_sl_list.append(t_sl)
    y_tp_list.append(t_tp)
    ref_closes_list.append(c100)
    timestamps_list.append(times[i])

print(f"3. Compiling Full 4.2-Year Dataset: {len(y_action_list):,} total samples.")
print(f"   Valid BUY_LIMIT setups:  {valid_buys:,}")
print(f"   Valid SELL_LIMIT setups: {valid_sells:,}")

np.savez_compressed(
    OUTPUT_FILE,
    X_raw=np.array(X_raw_list, dtype=np.float32),
    y_action=np.array(y_action_list, dtype=np.int64),
    y_entry=np.array(y_entry_list, dtype=np.float32),
    y_sl=np.array(y_sl_list, dtype=np.float32),
    y_tp=np.array(y_tp_list, dtype=np.float32),
    ref_closes=np.array(ref_closes_list, dtype=np.float32),
    timestamps=np.array(timestamps_list)
)

print(f"Dataset successfully saved to: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.2f} MB)")
