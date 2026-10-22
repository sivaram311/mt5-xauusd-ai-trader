# Multi-Timeframe Hierarchical Dataset Generator for XAUUSD M1 Sniper V2
# - Macro Hierarchy: 2-Week, Previous Week (PWH/PWL), Previous Day (PDH/PDL), H4, M15 levels
# - Micro Engine: 180 M1 candles (3 hours) + Candle Color + Wicks + Body
# - Targets: Asymmetric R:R (1:3.0 to 1:5.0+) with exact $0.35 bid/ask spread mechanics
# - Output: W:/data/trading/xauusd_m1_v2_hierarchical_dataset.npz
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import sys

OUTPUT_FILE = r"W:\data\trading\xauusd_m1_v2_hierarchical_dataset.npz"
PREVIEW_FILE = r"W:\data\trading\m1_v2_sample_preview.json"
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
LOOKBACK_M1 = 180       # 180 M1 bars (3 hours of micro-macro price action)
HORIZON_M1 = 45         # Forward 45 M1 bars max trade duration
SPREAD_FEE = 0.35       # $0.35/oz gold spread buffer
TOTAL_M1_BARS = 99000   # ~3.5 months of continuous micro history

print(f"1. Connecting to MT5 and fetching historical data across timeframes for {SYMBOL}...")
if not mt5.initialize(path=TERMINAL_PATH):
    print(f"Error: MT5 init failed: {mt5.last_error()}")
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)

# 1. Fetch M1 base rates
rates_m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, TOTAL_M1_BARS)
# 2. Fetch H1 rates for fast rolling higher-timeframe levels
rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 3000)
# 3. Fetch D1 rates for daily levels
rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 200)
mt5.shutdown()

if rates_m1 is None or len(rates_m1) == 0:
    print("Error: Could not retrieve M1 rates.")
    sys.exit(1)

print(f"   Fetched {len(rates_m1)} M1 bars, {len(rates_h1)} H1 bars, {len(rates_d1)} D1 bars.")

# Dataframes
df_m1 = pd.DataFrame(rates_m1)
df_m1['dt'] = pd.to_datetime(df_m1['time'], unit='s')
df_m1 = df_m1.sort_values('dt').reset_index(drop=True)

df_d1 = pd.DataFrame(rates_d1)
df_d1['dt'] = pd.to_datetime(df_d1['time'], unit='s').dt.date

highs = df_m1['high'].values
lows = df_m1['low'].values
opens = df_m1['open'].values
closes = df_m1['close'].values
times_epoch = df_m1['time'].values
times_str = df_m1['dt'].dt.strftime('%Y-%m-%d %H:%M:%S').values
n_bars = len(df_m1)

# Precalculate rolling macro levels along M1 sequence for high-speed generation
print("2. Calculating multi-timeframe rolling levels (2-Week, 1-Week, Daily, H4, M15)...")
# Rolling 15-minute level: min/max of last 15 bars
m15_high = pd.Series(highs).rolling(15).max().bfill().values
m15_low = pd.Series(lows).rolling(15).min().bfill().values

# Rolling 4-hour level: min/max of last 240 bars
h4_high = pd.Series(highs).rolling(240).max().bfill().values
h4_low = pd.Series(lows).rolling(240).min().bfill().values

# Rolling 1-day (24-hour) level: min/max of last 1440 bars
d1_high = pd.Series(highs).rolling(1440).max().bfill().values
d1_low = pd.Series(lows).rolling(1440).min().bfill().values

# Rolling 1-week (5 trading days = 7200 bars)
w1_high = pd.Series(highs).rolling(7200).max().bfill().values
w1_low = pd.Series(lows).rolling(7200).min().bfill().values

# Rolling 2-week (10 trading days = 14400 bars)
w2_high = pd.Series(highs).rolling(14400).max().bfill().values
w2_low = pd.Series(lows).rolling(14400).min().bfill().values

print("3. Generating hierarchical samples with exact swing wicks and spread mechanics...")
samples = []
# Start after 2-week warmup (14,400 bars)
start_idx = max(LOOKBACK_M1, 14400)
step_stride = 1

for i in range(start_idx, n_bars - HORIZON_M1, step_stride):
    c180 = closes[i]

    # Micro 180 M1 candles
    win_o = opens[i - LOOKBACK_M1 + 1 : i + 1]
    win_h = highs[i - LOOKBACK_M1 + 1 : i + 1]
    win_l = lows[i - LOOKBACK_M1 + 1 : i + 1]
    win_c = closes[i - LOOKBACK_M1 + 1 : i + 1]

    # Candle color (+1 for Green, -1 for Red, 0 for Doji)
    win_color = np.where(win_c > win_o, 1.0, np.where(win_c < win_o, -1.0, 0.0)).astype(np.float32)

    # Forward simulation window (45 bars = 45 mins)
    fwd_h = highs[i + 1 : i + 1 + HORIZON_M1]
    fwd_l = lows[i + 1 : i + 1 + HORIZON_M1]

    # Macro level distances relative to current close c180
    macro_vec = [
        round(w2_high[i] - c180, 2),
        round(w2_low[i] - c180, 2),
        round(w1_high[i] - c180, 2),
        round(w1_low[i] - c180, 2),
        round(d1_high[i] - c180, 2),
        round(d1_low[i] - c180, 2),
        round(h4_high[i] - c180, 2),
        round(h4_low[i] - c180, 2),
        round(m15_high[i] - c180, 2),
        round(m15_low[i] - c180, 2),
        # Time of day (UTC hour / 24.0)
        round((times_epoch[i] % 86400) / 86400.0, 4)
    ]

    # Identify clean swing low / swing high formed 2 to 5 candles ago
    # For BUY: Look for swing low formed between i-4 and i-1
    found_order = False
    best_trade = None

    # Candidate buy pivot
    for p_offset in [1, 2, 3, 4]:
        p = i - p_offset
        if win_l[p - (i - LOOKBACK_M1 + 1)] == min(win_l[max(0, p - 2 - (i - LOOKBACK_M1 + 1)) : p + 3 - (i - LOOKBACK_M1 + 1)]):
            tip_low = win_l[p - (i - LOOKBACK_M1 + 1)]
            # Buy Entry is slightly above or right at tip, + SPREAD_FEE for Ask fill
            entry = tip_low + 0.10
            actual_fill = entry + SPREAD_FEE
            
            # Test tight SL: $1.00 to $1.50 below tip
            for risk in [1.00, 1.25, 1.50]:
                sl = tip_low - risk - SPREAD_FEE # with -0.35 buffer as requested
                
                # Test target RR tiers from 1:3.0 up to 1:5.5
                for rr in [3.0, 4.0, 5.0]:
                    gross_tp = entry + (risk * rr) + SPREAD_FEE # +0.35 buffer for target
                    
                    # Forward simulation
                    triggered = False
                    stopped = False
                    hit_tp = False
                    
                    for k in range(len(fwd_l)):
                        if not triggered:
                            if fwd_l[k] <= entry:
                                triggered = True
                                if fwd_l[k] <= sl:
                                    stopped = True
                                    break
                                if fwd_h[k] >= gross_tp:
                                    hit_tp = True
                                    break
                        else:
                            if fwd_l[k] <= sl:
                                stopped = True
                                break
                            if fwd_h[k] >= gross_tp:
                                hit_tp = True
                                break
                                
                    if triggered and hit_tp and not stopped:
                        best_trade = {
                            "action": "BUY_LIMIT",
                            "entry_offset": round(entry - c180, 2),
                            "sl_offset": round(sl - c180, 2),
                            "tp_offset": round(gross_tp - c180, 2),
                            "rr": rr
                        }
                        found_order = True
                        break
                if found_order:
                    break
        if found_order:
            break

    # If no buy, test candidate sell pivot
    if not found_order:
        for p_offset in [1, 2, 3, 4]:
            p = i - p_offset
            if win_h[p - (i - LOOKBACK_M1 + 1)] == max(win_h[max(0, p - 2 - (i - LOOKBACK_M1 + 1)) : p + 3 - (i - LOOKBACK_M1 + 1)]):
                tip_high = win_h[p - (i - LOOKBACK_M1 + 1)]
                entry = tip_high - 0.10
                
                for risk in [1.00, 1.25, 1.50]:
                    sl = tip_high + risk + SPREAD_FEE # +0.35 buffer for sell stop trigger
                    
                    for rr in [3.0, 4.0, 5.0]:
                        gross_tp = entry - (risk * rr) - SPREAD_FEE # -0.35 buffer for sell target
                        
                        triggered = False
                        stopped = False
                        hit_tp = False
                        
                        for k in range(len(fwd_h)):
                            if not triggered:
                                if fwd_h[k] >= entry:
                                    triggered = True
                                    if fwd_h[k] >= sl:
                                        stopped = True
                                        break
                                    if fwd_l[k] <= gross_tp:
                                        hit_tp = True
                                        break
                            else:
                                if fwd_h[k] >= sl:
                                    stopped = True
                                    break
                                if fwd_l[k] <= gross_tp:
                                    hit_tp = True
                                    break
                                    
                        if triggered and hit_tp and not stopped:
                            best_trade = {
                                "action": "SELL_LIMIT",
                                "entry_offset": round(entry - c180, 2),
                                "sl_offset": round(sl - c180, 2),
                                "tp_offset": round(gross_tp - c180, 2),
                                "rr": rr
                            }
                            found_order = True
                            break
                    if found_order:
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
        "end_time": times_str[i],
        "ref_close": round(c180, 2),
        "candles_open": win_o.round(2).tolist(),
        "candles_high": win_h.round(2).tolist(),
        "candles_low": win_l.round(2).tolist(),
        "candles_close": win_c.round(2).tolist(),
        "candles_color": win_color.tolist(),
        "macro_features": macro_vec,
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
X_open = np.array(dataset_df['candles_open'].tolist(), dtype=np.float32)
X_high = np.array(dataset_df['candles_high'].tolist(), dtype=np.float32)
X_low = np.array(dataset_df['candles_low'].tolist(), dtype=np.float32)
X_close = np.array(dataset_df['candles_close'].tolist(), dtype=np.float32)
X_color = np.array(dataset_df['candles_color'].tolist(), dtype=np.float32)
# Micro tensor: (N, 180, 5)
X_micro = np.stack([X_open, X_high, X_low, X_close, X_color], axis=-1)

# Macro tensor: (N, 11)
X_macro = np.array(dataset_df['macro_features'].tolist(), dtype=np.float32)

action_map = {"NO_ORDER": 0, "BUY_LIMIT": 1, "SELL_LIMIT": 2}
y_action = np.array([action_map[a] for a in dataset_df['action']], dtype=np.int64)
y_entry = np.array(dataset_df['entry_offset'], dtype=np.float32)
y_sl = np.array(dataset_df['sl_offset'], dtype=np.float32)
y_tp = np.array(dataset_df['tp_offset'], dtype=np.float32)
ref_closes = np.array(dataset_df['ref_close'], dtype=np.float32)
end_times = np.array(dataset_df['end_time'].tolist())

np.savez_compressed(
    OUTPUT_FILE,
    X_micro=X_micro,
    X_macro=X_macro,
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
print(f"   Saved preview to {PREVIEW_FILE}")
print("   Done!")
