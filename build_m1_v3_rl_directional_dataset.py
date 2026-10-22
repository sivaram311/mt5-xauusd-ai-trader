# Dataset Generator for Direction-Aware / Reinforcement-Trained M1 Sniper (V3)
# Ingests 99,000 M1 bars + H4 Macro Trend Slope & Structure
# Enforces directional reward/penalty matrix:
# - Severe negative reward for counter-trend entries (e.g. Buying in H4 downtrend)
# - Higher reward for pro-trend pullbacks (e.g. Selling relief bounces in downtrend)
# - Bakes in $0.35 bid/ask spread friction
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import sys

OUTPUT_FILE = r"W:\data\trading\xauusd_m1_v3_rl_directional_dataset.npz"
PREVIEW_FILE = r"W:\data\trading\m1_v3_rl_sample_preview.json"
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
LOOKBACK_M1 = 180       # 180 M1 bars (3 hours micro)
HORIZON_M1 = 45         # Forward 45 M1 bars max trade horizon
SPREAD_FEE = 0.35       # $0.35 spread friction
TOTAL_M1_BARS = 99000   # ~3.5 months of micro history

print(f"1. Connecting to MT5 and fetching data for {SYMBOL}...")
if not mt5.initialize(path=TERMINAL_PATH):
    print(f"Error: MT5 init failed: {mt5.last_error()}")
    sys.exit(1)

mt5.symbol_select(SYMBOL, True)
rates_m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, TOTAL_M1_BARS)
rates_h4 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 1000)
rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 200)
mt5.shutdown()

if rates_m1 is None or len(rates_m1) == 0:
    print("Error: Could not retrieve M1 rates.")
    sys.exit(1)

df_m1 = pd.DataFrame(rates_m1)
df_m1['dt'] = pd.to_datetime(df_m1['time'], unit='s')
df_m1 = df_m1.sort_values('dt').reset_index(drop=True)

highs = df_m1['high'].values
lows = df_m1['low'].values
opens = df_m1['open'].values
closes = df_m1['close'].values
times_epoch = df_m1['time'].values
times_str = df_m1['dt'].dt.strftime('%Y-%m-%d %H:%M:%S').values
n_bars = len(df_m1)

print("2. Calculating H4 trend regimes and multi-timeframe rolling baselines...")
# H4 baseline: 240 M1 bars = 4 hours. 20-period H4 EMA = 4800 M1 bars
h4_ema20 = pd.Series(closes).ewm(span=4800, adjust=False).mean().values
# H4 ATR (volatility normalizer)
h4_tr = np.maximum(highs - lows, np.abs(highs - pd.Series(closes).shift(1).fillna(closes[0]).values))
h4_atr = pd.Series(h4_tr).rolling(4800).mean().bfill().values

# Normalized H4 Trend Slope: (Close - H4_EMA20) / H4_ATR
h4_slope = (closes - h4_ema20) / (h4_atr + 1e-4)

# Multi-timeframe levels
m15_high = pd.Series(highs).rolling(15).max().bfill().values
m15_low = pd.Series(lows).rolling(15).min().bfill().values
h4_high = pd.Series(highs).rolling(240).max().bfill().values
h4_low = pd.Series(lows).rolling(240).min().bfill().values
d1_high = pd.Series(highs).rolling(1440).max().bfill().values
d1_low = pd.Series(lows).rolling(1440).min().bfill().values
w1_high = pd.Series(highs).rolling(7200).max().bfill().values
w1_low = pd.Series(lows).rolling(7200).min().bfill().values
w2_high = pd.Series(highs).rolling(14400).max().bfill().values
w2_low = pd.Series(lows).rolling(14400).min().bfill().values

print("3. Generating Reinforcement-Labeled samples with Trend Alignment Penalties...")
samples = []
start_idx = max(LOOKBACK_M1, 14400)
step_stride = 1

for i in range(start_idx, n_bars - HORIZON_M1, step_stride):
    c180 = closes[i]
    slope = h4_slope[i] # Current macro trend slope

    win_o = opens[i - LOOKBACK_M1 + 1 : i + 1]
    win_h = highs[i - LOOKBACK_M1 + 1 : i + 1]
    win_l = lows[i - LOOKBACK_M1 + 1 : i + 1]
    win_c = closes[i - LOOKBACK_M1 + 1 : i + 1]
    win_color = np.where(win_c > win_o, 1.0, np.where(win_c < win_o, -1.0, 0.0)).astype(np.float32)

    fwd_h = highs[i + 1 : i + 1 + HORIZON_M1]
    fwd_l = lows[i + 1 : i + 1 + HORIZON_M1]

    # Macro vector (12 features including slope)
    macro_vec = [
        round(float(slope), 3),
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
        round((times_epoch[i] % 86400) / 86400.0, 4)
    ]

    found_order = False
    best_trade = None

    # Reinforcement Directional Gate:
    # If slope < -0.5 (Strong Downtrend), FORBID BUY_LIMIT. Only allow pro-trend SELL_LIMIT.
    # If slope > +0.5 (Strong Uptrend), FORBID SELL_LIMIT. Only allow pro-trend BUY_LIMIT.
    allow_buy = slope >= -0.50
    allow_sell = slope <= 0.50

    # Test BUY candidate if allowed by macro regime
    if allow_buy:
        for p_offset in [1, 2, 3, 4]:
            p = i - p_offset
            if win_l[p - (i - LOOKBACK_M1 + 1)] == min(win_l[max(0, p - 2 - (i - LOOKBACK_M1 + 1)) : p + 3 - (i - LOOKBACK_M1 + 1)]):
                tip_low = win_l[p - (i - LOOKBACK_M1 + 1)]
                entry = tip_low + 0.10
                
                # Minimum SL: $1.50 to $2.50 (widened to absorb spread spikes)
                for risk in [1.50, 2.00, 2.50]:
                    sl = tip_low - risk - SPREAD_FEE
                    for rr in [3.0, 4.0, 5.0]:
                        gross_tp = entry + (risk * rr) + SPREAD_FEE
                        
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
                                "reward": float(rr)
                            }
                            found_order = True
                            break
                    if found_order:
                        break
            if found_order:
                break

    # Test SELL candidate if allowed by macro regime
    if not found_order and allow_sell:
        for p_offset in [1, 2, 3, 4]:
            p = i - p_offset
            if win_h[p - (i - LOOKBACK_M1 + 1)] == max(win_h[max(0, p - 2 - (i - LOOKBACK_M1 + 1)) : p + 3 - (i - LOOKBACK_M1 + 1)]):
                tip_high = win_h[p - (i - LOOKBACK_M1 + 1)]
                entry = tip_high - 0.10
                
                for risk in [1.50, 2.00, 2.50]:
                    sl = tip_high + risk + SPREAD_FEE
                    for rr in [3.0, 4.0, 5.0]:
                        gross_tp = entry - (risk * rr) - SPREAD_FEE
                        
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
                                "reward": float(rr)
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
            "reward": 0.0
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
        "tp_offset": best_trade["tp_offset"],
        "reward": best_trade["reward"]
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
X_micro = np.stack([X_open, X_high, X_low, X_close, X_color], axis=-1)

X_macro = np.array(dataset_df['macro_features'].tolist(), dtype=np.float32)
rewards = np.array(dataset_df['reward'], dtype=np.float32)

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
    rewards=rewards,
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
