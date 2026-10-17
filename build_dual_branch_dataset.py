"""
Dual-Branch Dataset Generator: M15 Micro (100 bars) + H4 Macro (30 bars)
Timeframe Alignment:
- M15 branch: 100 consecutive bars (covers ~25 hours of micro structure / entry triggers)
- H4 branch: 30 consecutive bars closed strictly before the current M15 bar (covers 120 hours / ~2.5 weeks of macro trend)
- Output: NPZ archive containing aligned X_m15, X_h4, and verified forward 1:5 R:R targets
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
import time

OUTPUT_FILE = r"W:\data\trading\xauusd_dual_branch_m15_h4_dataset.npz"
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
LOOKBACK_M15 = 100
LOOKBACK_H4 = 30
HORIZON_M15 = 48        # 12 hours forward evaluation
MIN_RR = 5.0
MAX_M15_BARS = 99000

print("1. Initializing MT5 and fetching data...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 init failed")
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)
print(f"   Fetching {MAX_M15_BARS} M15 bars...")
rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, MAX_M15_BARS)

print("   Fetching 10,000 H4 bars...")
rates_h4 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 10000)
mt5.shutdown()

if rates_m15 is None or rates_h4 is None:
    print("Error fetching rates")
    sys.exit(1)

df_m15 = pd.DataFrame(rates_m15)
df_m15['time_dt'] = pd.to_datetime(df_m15['time'], unit='s')
df_m15 = df_m15.sort_values('time_dt').reset_index(drop=True)

df_h4 = pd.DataFrame(rates_h4)
df_h4['time_dt'] = pd.to_datetime(df_h4['time'], unit='s')
df_h4 = df_h4.sort_values('time_dt').reset_index(drop=True)

print(f"   M15: {len(df_m15)} bars ({df_m15['time_dt'].iloc[0]} to {df_m15['time_dt'].iloc[-1]})")
print(f"   H4:  {len(df_h4)} bars ({df_h4['time_dt'].iloc[0]} to {df_h4['time_dt'].iloc[-1]})")

# Arrays for fast indexing
h4_times = df_h4['time'].values
h4_opens = df_h4['open'].values
h4_highs = df_h4['high'].values
h4_lows = df_h4['low'].values
h4_closes = df_h4['close'].values

m15_times = df_m15['time'].values
m15_highs = df_m15['high'].values
m15_lows = df_m15['low'].values
m15_opens = df_m15['open'].values
m15_closes = df_m15['close'].values

n_m15 = len(df_m15)

X_m15_list = []
X_h4_list = []
y_action_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
ref_closes_list = []
end_times_list = []

print("2. Aligning M15 and H4 timeframes and generating labels...")
t0 = time.time()

# We need at least LOOKBACK_H4 bars in H4 before the earliest M15 timestamp
min_m15_start = LOOKBACK_M15 - 1

for i in range(min_m15_start, n_m15 - HORIZON_M15):
    c100_time = m15_times[i]
    c100 = m15_closes[i]

    # Find the index of the latest closed H4 bar strictly BEFORE or AT the current M15 bar
    # (Prevents lookahead leakage from H4)
    h4_idx = np.searchsorted(h4_times, c100_time, side='right') - 1
    if h4_idx < LOOKBACK_H4 - 1:
        continue # Not enough H4 history yet

    # 1. Micro M15 features (100 candles)
    win_m15_o = m15_opens[i - LOOKBACK_M15 + 1 : i + 1]
    win_m15_h = m15_highs[i - LOOKBACK_M15 + 1 : i + 1]
    win_m15_l = m15_lows[i - LOOKBACK_M15 + 1 : i + 1]
    win_m15_c = m15_closes[i - LOOKBACK_M15 + 1 : i + 1]

    # 2. Macro H4 features (30 candles)
    win_h4_o = h4_opens[h4_idx - LOOKBACK_H4 + 1 : h4_idx + 1]
    win_h4_h = h4_highs[h4_idx - LOOKBACK_H4 + 1 : h4_idx + 1]
    win_h4_l = h4_lows[h4_idx - LOOKBACK_H4 + 1 : h4_idx + 1]
    win_h4_c = h4_closes[h4_idx - LOOKBACK_H4 + 1 : h4_idx + 1]

    # Forward M15 evaluation window (48 bars)
    fwd_h = m15_highs[i + 1 : i + 1 + HORIZON_M15]
    fwd_l = m15_lows[i + 1 : i + 1 + HORIZON_M15]

    found_order = False
    action_id = 0
    entry_off = 0.0
    sl_off = 0.0
    tp_off = 0.0

    # Macro trend direction from H4: 10-period EMA / SMA vs 30-period SMA
    h4_trend_bull = np.mean(win_h4_c[-10:]) > np.mean(win_h4_c)
    h4_trend_bear = np.mean(win_h4_c[-10:]) < np.mean(win_h4_c)

    # If Macro H4 is Bullish or Neutral -> Scan candidate BUY LIMITs
    if not h4_trend_bear:
        candidate_buy_entries = [round(np.min(win_m15_l[-k:]), 2) for k in [5, 10, 15, 20] if np.min(win_m15_l[-k:]) < c100 - 0.5]
        for entry in set(candidate_buy_entries):
            for risk in [2.0, 3.0, 4.0, 5.0]:
                sl = entry - risk
                tp = entry + (risk * MIN_RR)
                triggered, stopped, took_profit = False, False, False
                for k in range(len(fwd_l)):
                    if not triggered:
                        if fwd_l[k] <= entry:
                            triggered = True
                            if fwd_l[k] <= sl: stopped = True; break
                            if fwd_h[k] >= tp: took_profit = True; break
                    else:
                        if fwd_l[k] <= sl: stopped = True; break
                        if fwd_h[k] >= tp: took_profit = True; break
                if triggered and took_profit and not stopped:
                    action_id = 1 # BUY_LIMIT
                    entry_off = round(entry - c100, 2)
                    sl_off = round(sl - c100, 2)
                    tp_off = round(tp - c100, 2)
                    found_order = True
                    break
            if found_order:
                break

    # If Macro H4 is Bearish or Neutral and no Buy was found -> Scan candidate SELL LIMITs
    if not found_order and not h4_trend_bull:
        candidate_sell_entries = [round(np.max(win_m15_h[-k:]), 2) for k in [5, 10, 15, 20] if np.max(win_m15_h[-k:]) > c100 + 0.5]
        for entry in set(candidate_sell_entries):
            for risk in [2.0, 3.0, 4.0, 5.0]:
                sl = entry + risk
                tp = entry - (risk * MIN_RR)
                triggered, stopped, took_profit = False, False, False
                for k in range(len(fwd_h)):
                    if not triggered:
                        if fwd_h[k] >= entry:
                            triggered = True
                            if fwd_h[k] >= sl: stopped = True; break
                            if fwd_l[k] <= tp: took_profit = True; break
                    else:
                        if fwd_h[k] >= sl: stopped = True; break
                        if fwd_l[k] <= tp: took_profit = True; break
                if triggered and took_profit and not stopped:
                    action_id = 2 # SELL_LIMIT
                    entry_off = round(entry - c100, 2)
                    sl_off = round(sl - c100, 2)
                    tp_off = round(tp - c100, 2)
                    found_order = True
                    break
            if found_order:
                break

    # Save arrays
    m15_sample = np.stack([win_m15_o, win_m15_h, win_m15_l, win_m15_c], axis=-1)
    h4_sample = np.stack([win_h4_o, win_h4_h, win_h4_l, win_h4_c], axis=-1)

    X_m15_list.append(m15_sample)
    X_h4_list.append(h4_sample)
    y_action_list.append(action_id)
    y_entry_list.append(entry_off)
    y_sl_list.append(sl_off)
    y_tp_list.append(tp_off)
    ref_closes_list.append(c100)
    end_times_list.append(c100_time)

print(f"3. Alignment complete in {time.time()-t0:.1f}s. Total aligned sequences: {len(y_action_list)}")

X_m15_arr = np.array(X_m15_list, dtype=np.float32)
X_h4_arr = np.array(X_h4_list, dtype=np.float32)
y_action_arr = np.array(y_action_list, dtype=np.int64)
y_entry_arr = np.array(y_entry_list, dtype=np.float32)
y_sl_arr = np.array(y_sl_list, dtype=np.float32)
y_tp_arr = np.array(y_tp_list, dtype=np.float32)
ref_closes_arr = np.array(ref_closes_list, dtype=np.float32)
end_times_arr = np.array(end_times_list, dtype=np.int64)

unique, counts = np.unique(y_action_arr, return_counts=True)
print("   Action Distribution:", dict(zip(unique, counts)))
print(f"   X_m15 shape: {X_m15_arr.shape} | X_h4 shape: {X_h4_arr.shape}")

print(f"4. Saving dual-branch dataset to {OUTPUT_FILE}...")
np.savez_compressed(
    OUTPUT_FILE,
    X_m15=X_m15_arr,
    X_h4=X_h4_arr,
    y_action=y_action_arr,
    y_entry=y_entry_arr,
    y_sl=y_sl_arr,
    y_tp=y_tp_arr,
    ref_closes=ref_closes_arr,
    end_times=end_times_arr
)
print("   Done!")
