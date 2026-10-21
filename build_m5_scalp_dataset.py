"""
M5 Scalping Dataset Generator for XAUUSD (All Hours / Spread-Buffered)
- Ingests all available continuous M5 bars without session filtering
- Bakes in explicit spread friction ($0.35 USD) directly into target and fill conditions
- Lookback: 60 bars (5 hours)
- Horizon: 18 bars (90 minutes max duration)
- Risk:Reward: 1:2.0 to 1:2.5
- Saves compressed NPZ + preview JSON
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import sys

OUTPUT_FILE = r"W:\data\trading\xauusd_m5_scalp_dataset.npz"
PREVIEW_FILE = r"W:\data\trading\m5_scalp_sample_preview.json"
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
LOOKBACK = 60          # 60 M5 bars (5 hours micro-structure)
HORIZON = 18           # Forward 18 bars (90 minutes maximum trade lifespan)
MIN_RR = 2.0           # Scalping Risk-Reward 1:2.0
SPREAD_FRICTION = 0.35 # Fixed spread penalty ($0.35/oz)
TOTAL_BARS = 99000     # Maximum historical bars available in MT5

print(f"1. Connecting to MT5 and fetching {TOTAL_BARS} M5 bars for {SYMBOL}...")
if not mt5.initialize(path=TERMINAL_PATH):
    print(f"Error: MT5 init failed: {mt5.last_error()}")
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
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
step_stride = 1

print(f"2. Generating continuous samples across ALL hours (Lookback={LOOKBACK}, Horizon={HORIZON}, Spread=${SPREAD_FRICTION})...")

for i in range(LOOKBACK - 1, n_bars - HORIZON, step_stride):
    c_current = closes[i]
    
    # Lookback window
    win_h = highs[i - LOOKBACK + 1 : i + 1]
    win_l = lows[i - LOOKBACK + 1 : i + 1]
    win_c = closes[i - LOOKBACK + 1 : i + 1]
    win_o = opens[i - LOOKBACK + 1 : i + 1]
    
    # Forward simulation window
    fwd_h = highs[i + 1 : i + 1 + HORIZON]
    fwd_l = lows[i + 1 : i + 1 + HORIZON]
    
    found_order = False
    best_trade = None
    
    # Candidate entry pullbacks below/above current close for scalping
    candidate_buy_entries = []
    for look in [3, 6, 12, 18]:
        lvl = np.min(win_l[-look:])
        if lvl < c_current - 0.25: # At least $0.25 below current close
            candidate_buy_entries.append(round(lvl, 2))
            
    candidate_sell_entries = []
    for look in [3, 6, 12, 18]:
        lvl = np.max(win_h[-look:])
        if lvl > c_current + 0.25: # At least $0.25 above current close
            candidate_sell_entries.append(round(lvl, 2))
            
    # Test BUY LIMIT candidates
    for entry in set(candidate_buy_entries):
        for risk in [1.25, 1.50, 2.00]:
            sl = entry - risk
            # For BUY: Target must clear TP + SPREAD_FRICTION
            gross_tp = entry + (risk * MIN_RR)
            req_high = gross_tp + SPREAD_FRICTION
            
            triggered = False
            stopped = False
            take_profit = False
            
            for k in range(len(fwd_l)):
                if not triggered:
                    # Entry must be hit on Ask/Bid dip
                    if fwd_l[k] <= entry:
                        triggered = True
                        if fwd_l[k] <= sl:
                            stopped = True
                            break
                        if fwd_h[k] >= req_high:
                            take_profit = True
                            break
                else:
                    if fwd_l[k] <= sl:
                        stopped = True
                        break
                    if fwd_h[k] >= req_high:
                        take_profit = True
                        break
                        
            if triggered and take_profit and not stopped:
                best_trade = {
                    "action": "BUY_LIMIT",
                    "entry_offset": round(entry - c_current, 2),
                    "sl_offset": round(sl - c_current, 2),
                    "tp_offset": round(gross_tp - c_current, 2),
                    "rr": MIN_RR
                }
                found_order = True
                break
        if found_order:
            break
            
    # Test SELL LIMIT candidates
    if not found_order:
        for entry in set(candidate_sell_entries):
            for risk in [1.25, 1.50, 2.00]:
                sl = entry + risk
                gross_tp = entry - (risk * MIN_RR)
                req_low = gross_tp - SPREAD_FRICTION
                
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
                            if fwd_l[k] <= req_low:
                                take_profit = True
                                break
                    else:
                        if fwd_h[k] >= sl:
                            stopped = True
                            break
                        if fwd_l[k] <= req_low:
                            take_profit = True
                            break
                            
                if triggered and take_profit and not stopped:
                    best_trade = {
                        "action": "SELL_LIMIT",
                        "entry_offset": round(entry - c_current, 2),
                        "sl_offset": round(sl - c_current, 2),
                        "tp_offset": round(gross_tp - c_current, 2),
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
        
    sample_record = {
        "end_time": times[i],
        "ref_close": round(c_current, 2),
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

print(f"3. Total samples processed: {len(samples)}")
dataset_df = pd.DataFrame(samples)
counts = dataset_df['action'].value_counts()
print("   Action distribution:")
print(counts)

print(f"4. Saving dataset to {OUTPUT_FILE} (Compressed NPZ)...")
X_open = np.array(dataset_df['candles_open'].tolist(), dtype=np.float32)
X_high = np.array(dataset_df['candles_high'].tolist(), dtype=np.float32)
X_low = np.array(dataset_df['candles_low'].tolist(), dtype=np.float32)
X_close = np.array(dataset_df['candles_close'].tolist(), dtype=np.float32)
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

with open(PREVIEW_FILE, "w") as f:
    json.dump(dataset_df.head(10).to_dict(orient="records"), f, indent=2)

print(f"   Successfully generated {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB)")
print(f"   Saved preview sample to {PREVIEW_FILE}")
print("   Done!")
