"""
Dataset Generator V3: High-Accuracy Liquidity Anchor & Realistic 1:3 R:R Target
Inputs: 100 historical M15 candles [time, open, high, low, close]
Enhancements:
1. Liquidity Anchor Labeling: Only label setups at structural swing points / rejection levels.
2. Realistic Intraday Targets: R:R = 1:3.0 (Risk $2.00 to Gain $6.00 on Gold).
3. Forward Horizon: 32 bars (8 hours) for faster capital turnover.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
import time

OUTPUT_FILE = r"W:\data\trading\xauusd_m15_v3_high_acc_dataset.npz"
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100
HORIZON = 32           # 8 hours forward evaluation
TARGET_RR = 3.0        # Realistic 1:3 Risk-Reward
RISK_PTS = 2.0         # Tight $2.00 Gold Stop Loss
MAX_M15_BARS = 99000   # 4+ years of history

print("1. Initializing MT5 and retrieving historical bars...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 init failed")
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, MAX_M15_BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("No rates retrieved")
    sys.exit(1)

df = pd.DataFrame(rates)
df['time_dt'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('time_dt').reset_index(drop=True)
print(f"   Fetched {len(df)} bars from {df['time_dt'].iloc[0]} to {df['time_dt'].iloc[-1]}")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
times = df['time'].values
n_bars = len(df)

X_m15_list = []
y_action_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
ref_closes_list = []
end_times_list = []

print("2. Scanning with Liquidity Anchor & 1:3 R:R Target...")
t0 = time.time()

for i in range(LOOKBACK - 1, n_bars - HORIZON):
    c100 = closes[i]
    c100_time = times[i]

    win_o = opens[i - LOOKBACK + 1 : i + 1]
    win_h = highs[i - LOOKBACK + 1 : i + 1]
    win_l = lows[i - LOOKBACK + 1 : i + 1]
    win_c = closes[i - LOOKBACK + 1 : i + 1]

    fwd_h = highs[i + 1 : i + 1 + HORIZON]
    fwd_l = lows[i + 1 : i + 1 + HORIZON]

    # Liquidity Anchor Detection in the last 20 candles
    # Swing low = minimum of prior bars
    swing_low_20 = np.min(win_l[-20:-3])
    swing_high_20 = np.max(win_h[-20:-3])

    # Recent rejection check:
    # Bullish anchor: current candle pierced below swing low or near it and closed higher
    bull_anchor = (np.min(win_l[-3:]) <= swing_low_20 + 0.5) and (win_c[-1] > win_o[-1])
    # Bearish anchor: current candle pierced above swing high or near it and closed lower
    bear_anchor = (np.max(win_h[-3:]) >= swing_high_20 - 0.5) and (win_c[-1] < win_o[-1])

    action_id = 0
    entry_off = 0.0
    sl_off = 0.0
    tp_off = 0.0
    found_trade = False

    # 1. Evaluate BUY LIMIT if Bullish Anchor is present
    if bull_anchor:
        entry = round(swing_low_20, 2)
        if entry < c100 - 0.30: # At least $0.30 below market
            sl = entry - RISK_PTS
            tp = entry + (RISK_PTS * TARGET_RR)
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
                found_trade = True

    # 2. Evaluate SELL LIMIT if Bearish Anchor is present
    if not found_trade and bear_anchor:
        entry = round(swing_high_20, 2)
        if entry > c100 + 0.30: # At least $0.30 above market
            sl = entry + RISK_PTS
            tp = entry - (RISK_PTS * TARGET_RR)
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
                found_trade = True

    sample_ohlc = np.stack([win_o, win_h, win_l, win_c], axis=-1)
    X_m15_list.append(sample_ohlc)
    y_action_list.append(action_id)
    y_entry_list.append(entry_off)
    y_sl_list.append(sl_off)
    y_tp_list.append(tp_off)
    ref_closes_list.append(c100)
    end_times_list.append(c100_time)

print(f"3. Scanning complete in {time.time()-t0:.1f}s. Total samples: {len(y_action_list)}")

X_m15_arr = np.array(X_m15_list, dtype=np.float32)
y_action_arr = np.array(y_action_list, dtype=np.int64)
y_entry_arr = np.array(y_entry_list, dtype=np.float32)
y_sl_arr = np.array(y_sl_list, dtype=np.float32)
y_tp_arr = np.array(y_tp_list, dtype=np.float32)
ref_closes_arr = np.array(ref_closes_list, dtype=np.float32)
end_times_arr = np.array(end_times_list, dtype=np.int64)

unique, counts = np.unique(y_action_arr, return_counts=True)
print("   Action Distribution (0=NO_ORDER, 1=BUY_LIMIT, 2=SELL_LIMIT):", dict(zip(unique, counts)))

print(f"4. Saving V3 dataset to {OUTPUT_FILE}...")
np.savez_compressed(
    OUTPUT_FILE,
    X_m15=X_m15_arr,
    y_action=y_action_arr,
    y_entry=y_entry_arr,
    y_sl=y_sl_arr,
    y_tp=y_tp_arr,
    ref_closes=ref_closes_arr,
    end_times=end_times_arr
)
print("   Dataset V3 successfully created!")
