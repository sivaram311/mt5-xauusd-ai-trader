"""
Dataset Generator for XAUUSD M15 Pending Orders (Buy Limit / Sell Limit only)
Inputs: 100 historical M15 candles [time, open, high, low, close]
Targets: Buy Limit / Sell Limit setups with R:R >= 1:5 within forward 48 bars (12 hours)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import sys

OUTPUT_FILE = r"W:\data\trading\xauusd_m15_100bars_dataset.npz"
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100         # 100 input bars
HORIZON = 48           # 48 bars (12 hours) forward window
MIN_RR = 5.0           # Minimum Risk-Reward 1:5
MIN_RISK_PTS = 1.5     # Minimum stop loss in points ($1.50) to prevent micro-scalp noise
MAX_RISK_PTS = 8.0     # Maximum stop loss in points ($8.00) to keep stops tight
MAX_TOTAL_BARS = 99000 # Full 4+ years of available MT5 history (Aug 2022 to Oct 2026)

print("1. Connecting to MT5...")
if not mt5.initialize(path=TERMINAL_PATH):
    print(f"Error: MT5 initialization failed: {mt5.last_error()}")
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)
print(f"2. Fetching {MAX_TOTAL_BARS} historical M15 bars for {SYMBOL}...")
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, MAX_TOTAL_BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("Error: Could not retrieve rates from MT5.")
    sys.exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df[['time', 'open', 'high', 'low', 'close']].sort_values('time').reset_index(drop=True)
print(f"   Fetched {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
times = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S').values
n_bars = len(df)

samples = []
step_stride = 1  # evaluate every candle step

print("3. Scanning sliding windows (Lookback=100, Forward Horizon=48, R:R >= 1:5)...")

for i in range(LOOKBACK - 1, n_bars - HORIZON, step_stride):
    # Current bar is at index i (100th candle)
    c100 = closes[i]
    
    # 1. Lookback window: [i - 99 to i] inclusive
    win_h = highs[i - LOOKBACK + 1 : i + 1]
    win_l = lows[i - LOOKBACK + 1 : i + 1]
    win_c = closes[i - LOOKBACK + 1 : i + 1]
    win_o = opens[i - LOOKBACK + 1 : i + 1]
    
    # Forward simulation window: [i + 1 to i + HORIZON]
    fwd_h = highs[i + 1 : i + 1 + HORIZON]
    fwd_l = lows[i + 1 : i + 1 + HORIZON]
    fwd_c = closes[i + 1 : i + 1 + HORIZON]
    
    # Candidate order finding
    found_order = False
    best_trade = None
    
    # Identify swing lows & swing highs in the lookback window as candidate limit entry levels
    # For BUY LIMIT: Price must dip below c100 to an entry level, then rise to TP with RR >= 5
    candidate_buy_entries = []
    # Test recent swing lows or pullbacks below c100
    for look in [5, 10, 15, 20]:
        lvl = np.min(win_l[-look:])
        if lvl < c100 - 0.5: # At least $0.50 below current close
            candidate_buy_entries.append(round(lvl, 2))
            
    # For SELL LIMIT: Price must rally above c100 to an entry level, then fall to TP with RR >= 5
    candidate_sell_entries = []
    for look in [5, 10, 15, 20]:
        lvl = np.max(win_h[-look:])
        if lvl > c100 + 0.5: # At least $0.50 above current close
            candidate_sell_entries.append(round(lvl, 2))
            
    # Test candidate BUY LIMITs
    for entry in set(candidate_buy_entries):
        # Scan SL levels from 1.5 to 6.0 points below entry
        for risk in [2.0, 3.0, 4.0, 5.0]:
            sl = entry - risk
            tp = entry + (risk * MIN_RR)
            
            # Simulate forward in time
            triggered = False
            stopped = False
            take_profit = False
            
            for k in range(len(fwd_l)):
                # Before trigger, check if price hits entry
                if not triggered:
                    if fwd_l[k] <= entry:
                        triggered = True
                        # On trigger candle, did it also hit SL?
                        if fwd_l[k] <= sl:
                            stopped = True
                            break
                        if fwd_h[k] >= tp:
                            take_profit = True
                            break
                else:
                    # After trigger
                    if fwd_l[k] <= sl:
                        stopped = True
                        break
                    if fwd_h[k] >= tp:
                        take_profit = True
                        break
                        
            if triggered and take_profit and not stopped:
                best_trade = {
                    "action": "BUY_LIMIT",
                    "entry_offset": round(entry - c100, 2),
                    "sl_offset": round(sl - c100, 2),
                    "tp_offset": round(tp - c100, 2),
                    "rr": MIN_RR
                }
                found_order = True
                break
        if found_order:
            break

    # If no buy limit found, test candidate SELL LIMITs
    if not found_order:
        for entry in set(candidate_sell_entries):
            for risk in [2.0, 3.0, 4.0, 5.0]:
                sl = entry + risk
                tp = entry - (risk * MIN_RR)
                
                triggered = False
                stopped = False
                take_profit = False
                
                for k in range(len(fwd_h)):
                    if not triggered:
                        if fwd_h[k] >= entry:
                            triggered = True
                            if fwd_h[k] >= sl:
                                stopped = True
                                break
                            if fwd_l[k] <= tp:
                                take_profit = True
                                break
                    else:
                        if fwd_h[k] >= sl:
                            stopped = True
                            break
                        if fwd_l[k] <= tp:
                            take_profit = True
                            break
                            
                if triggered and take_profit and not stopped:
                    best_trade = {
                        "action": "SELL_LIMIT",
                        "entry_offset": round(entry - c100, 2),
                        "sl_offset": round(sl - c100, 2),
                        "tp_offset": round(tp - c100, 2),
                        "rr": MIN_RR
                    }
                    found_order = True
                    break
            if found_order:
                break
                
    if not found_order:
        best_trade = {
            "action": "NO_ORDER",
            "entry_offset": 0.0,
            "sl_offset": 0.0,
            "tp_offset": 0.0,
            "rr": 0.0
        }

    # Store sample
    sample_record = {
        "end_time": times[i],
        "ref_close": round(c100, 2),
        "candles_time": times[i - LOOKBACK + 1 : i + 1].tolist(),
        "candles_open": win_o.round(2).tolist(),
        "candles_high": win_h.round(2).tolist(),
        "candles_low": win_l.round(2).tolist(),
        "candles_close": win_c.round(2).tolist(),
        "action": best_trade["action"],
        "entry_offset": best_trade["entry_offset"],
        "sl_offset": best_trade["sl_offset"],
        "tp_offset": best_trade["tp_offset"]
    }
    samples.append(sample_record)

print(f"4. Total samples generated: {len(samples)}")
dataset_df = pd.DataFrame(samples)
counts = dataset_df['action'].value_counts()
print("   Action distribution:")
print(counts)

print(f"5. Saving dataset to {OUTPUT_FILE} (Compressed NPZ)...")
# Vectorize into numpy arrays for direct fast ML training
X_open = np.array(dataset_df['candles_open'].tolist(), dtype=np.float32)
X_high = np.array(dataset_df['candles_high'].tolist(), dtype=np.float32)
X_low = np.array(dataset_df['candles_low'].tolist(), dtype=np.float32)
X_close = np.array(dataset_df['candles_close'].tolist(), dtype=np.float32)
# Shape: (N, 100, 4)
X_ohlc = np.stack([X_open, X_high, X_low, X_close], axis=-1)

action_map = {"NO_ORDER": 0, "BUY_LIMIT": 1, "SELL_LIMIT": 2}
y_action = np.array([action_map[a] for a in dataset_df['action']], dtype=np.int64)
y_entry = np.array(dataset_df['entry_offset'], dtype=np.float32)
y_sl = np.array(dataset_df['sl_offset'], dtype=np.float32)
y_tp = np.array(dataset_df['tp_offset'], dtype=np.float32)
ref_closes = np.array(dataset_df['ref_close'], dtype=np.float32)
end_times = np.array(dataset_df['end_time'].tolist())

np.savez_compressed(
    OUTPUT_FILE,
    X_ohlc=X_ohlc,
    y_action=y_action,
    y_entry=y_entry,
    y_sl=y_sl,
    y_tp=y_tp,
    ref_closes=ref_closes,
    end_times=end_times
)

# Also save a 10-sample JSON human preview file
preview_file = r"W:\data\trading\sample_preview.json"
with open(preview_file, "w") as f:
    json.dump(dataset_df.head(10).to_dict(orient="records"), f, indent=2)

print(f"   Saved compressed NPZ: {OUTPUT_FILE}")
print(f"   Saved human preview JSON: {preview_file}")
print("   Done!")
