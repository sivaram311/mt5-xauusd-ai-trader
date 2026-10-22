"""
Dataset Generator for High-Alpha Dual-Stream Session Model (Target: Profit Factor 3.0 - 4.0)
Architecture:
- Micro Stream: 60 M1 Candles (Shape: 60, 5) -> [Open, High, Low, Close, Normalized Momentum]
- Macro Stream: 12 Structural Features -> [PDH dist, PDL dist, PWH dist, PWL dist, H1 trend slope, 
                                           Session 30m High dist, Session 30m Low dist, Session ID, 
                                           Minutes in Session, Hour UTC, DayOfWeek, Spread]
Ground Truth Filter:
- Strict Session Gate: 12:00 to 17:00 UTC (Peak institutional momentum)
- Strict Trend Alignment: Only Buy when H1 Trend is Bullish; Only Sell when Bearish
- Tight Risk: $1.80 - $2.00 Stop Loss (incorporating $0.35 broker spread)
- Asymmetric Reward: Flat $10.00 Take Profit (Strict 1:5 Risk-to-Reward)
- Top-1 Daily Constraint: Maximum 1 high-conviction trade labeled per day
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
N_M1_BARS = 85000       # ~60 trading days of M1 history
SPREAD_FEE = 0.35       # Real broker spread ($0.35)
RISK_AMOUNT = 2.00      # $2.00 tight structural SL
REWARD_AMOUNT = 10.00   # $10.00 flat TP (1:5 Asymmetric Payoff)
MAX_FILL_MINUTES = 12   # Limit order must fill within 12 minutes
MAX_TRADE_MINUTES = 90  # Trade must complete $10 expansion within 90 minutes

SAVE_PATH_NPZ = r"W:\data\trading\xauusd_alpha_pf4_dataset.npz"
SAVE_PATH_PREVIEW = r"W:\data\trading\m1_alpha_pf4_preview.json"

print("1. Connecting to MT5 and fetching multi-timeframe candles...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

rates_m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, N_M1_BARS)
rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 1500)
rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 100)
rates_w1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_W1, 0, 20)
mt5.shutdown()

if rates_m1 is None or len(rates_m1) == 0:
    print("Error: Could not retrieve M1 rates.")
    sys.exit(1)

print(f"   Retrieved {len(rates_m1):,} M1 bars, {len(rates_h1):,} H1 bars, {len(rates_d1):,} Daily bars.")

df_m1 = pd.DataFrame(rates_m1)
df_m1['dt'] = pd.to_datetime(df_m1['time'], unit='s', utc=True)
df_m1['date'] = df_m1['dt'].dt.date
df_m1['hour'] = df_m1['dt'].dt.hour
df_m1['minute'] = df_m1['dt'].dt.minute
df_m1['dow'] = df_m1['dt'].dt.dayofweek

# 2. Compute H1 Trend Direction and Slope
df_h1 = pd.DataFrame(rates_h1)
df_h1['dt'] = pd.to_datetime(df_h1['time'], unit='s', utc=True)
df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
df_h1['h1_slope'] = (df_h1['close'] - df_h1['ema50']) / (df_h1['high'] - df_h1['low'] + 1e-4)

# Map H1 trend to M1 by timestamp merge_asof
df_m1 = pd.merge_asof(
    df_m1.sort_values('dt'),
    df_h1[['dt', 'h1_slope']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df_m1['h1_slope'] = df_m1['h1_slope'].fillna(0.0)

# 3. Compute Multi-Timeframe Levels (PDH, PDL, PWH, PWL)
df_d1 = pd.DataFrame(rates_d1)
df_d1['dt'] = pd.to_datetime(df_d1['time'], unit='s', utc=True)
df_d1['pdh'] = df_d1['high'].shift(1)
df_d1['pdl'] = df_d1['low'].shift(1)

df_m1 = pd.merge_asof(
    df_m1.sort_values('dt'),
    df_d1[['dt', 'pdh', 'pdl']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df_m1['pdh'] = df_m1['pdh'].bfill()
df_m1['pdl'] = df_m1['pdl'].bfill()

# Weekly Levels
df_w1 = pd.DataFrame(rates_w1)
df_w1['dt'] = pd.to_datetime(df_w1['time'], unit='s', utc=True)
df_w1['pwh'] = df_w1['high'].shift(1)
df_w1['pwl'] = df_w1['low'].shift(1)

df_m1 = pd.merge_asof(
    df_m1.sort_values('dt'),
    df_w1[['dt', 'pwh', 'pwl']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df_m1['pwh'] = df_m1['pwh'].bfill()
df_m1['pwl'] = df_m1['pwl'].bfill()

print("2. Scanning Golden High-Alpha setups (12:00-17:00 UTC with 1:5 Target)...")

n_total = len(df_m1)
opens = df_m1['open'].values.astype(np.float32)
highs = df_m1['high'].values.astype(np.float32)
lows = df_m1['low'].values.astype(np.float32)
closes = df_m1['close'].values.astype(np.float32)
hours = df_m1['hour'].values
minutes = df_m1['minute'].values
dows = df_m1['dow'].values
dates = df_m1['date'].values
h1_slopes = df_m1['h1_slope'].values.astype(np.float32)
pdh_arr = df_m1['pdh'].values.astype(np.float32)
pdl_arr = df_m1['pdl'].values.astype(np.float32)
pwh_arr = df_m1['pwh'].values.astype(np.float32)
pwl_arr = df_m1['pwl'].values.astype(np.float32)

# Precalculate session opening range (12:00 to 12:30 UTC) for each date
session_30m_ranges = {}
for dt_date, grp in df_m1.groupby('date'):
    s_grp = grp[(grp['hour'] == 12) & (grp['minute'] < 30)]
    if len(s_grp) > 0:
        session_30m_ranges[dt_date] = {
            'high': float(s_grp['high'].max()),
            'low': float(s_grp['low'].min())
        }
    else:
        session_30m_ranges[dt_date] = {'high': 0.0, 'low': 0.0}

# Store all candidate days
day_candidates = {}

start_idx = 60
end_idx = n_total - MAX_TRADE_MINUTES - 10

for i in range(start_idx, end_idx):
    hr = hours[i]
    # STRICT SESSION GATE: Only 12:30 to 17:00 UTC (After the 30m opening range)
    if not (12 <= hr < 17):
        continue
    if hr == 12 and minutes[i] < 30:
        continue

    cur_date = dates[i]
    sess_range = session_30m_ranges.get(cur_date, {'high': 0.0, 'low': 0.0})
    s_high = sess_range['high']
    s_low = sess_range['low']
    if s_high == 0.0 or s_low == 0.0:
        continue

    c_now = closes[i]
    slope = h1_slopes[i]

    # Look forward up to MAX_TRADE_MINUTES
    fwd_h = highs[i + 1 : i + 1 + MAX_TRADE_MINUTES]
    fwd_l = lows[i + 1 : i + 1 + MAX_TRADE_MINUTES]

    found_trade = False
    trade_meta = None

    # BULLISH SETUP: H1 Trend Bullish + Sweep below Session Low
    if slope > 0.10:
        # Check if the recent 3 bars swept below Session 30m Low and rejected back
        recent_l = lows[i - 3 : i + 1]
        if min(recent_l) < s_low and c_now > s_low - 0.50:
            entry = round(s_low + 0.10, 2)
            sl = round(min(recent_l) - RISK_AMOUNT - SPREAD_FEE, 2)
            tp = round(entry + REWARD_AMOUNT + SPREAD_FEE, 2)
            
            # Simulate pending limit order
            filled = False
            hit_tp = False
            stopped = False
            fill_time = 0

            for k in range(len(fwd_l)):
                if not filled:
                    if k < MAX_FILL_MINUTES and fwd_l[k] <= entry:
                        filled = True
                        fill_time = k + 1
                        if fwd_l[k] <= sl:
                            stopped = True; break
                        if fwd_h[k] >= tp:
                            hit_tp = True; break
                    elif k >= MAX_FILL_MINUTES:
                        break
                else:
                    if fwd_l[k] <= sl:
                        stopped = True; break
                    if fwd_h[k] >= tp:
                        hit_tp = True; break

            if filled and hit_tp and not stopped:
                found_trade = True
                trade_meta = {
                    "direction": 0, # BUY_LIMIT
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "time_to_fill": float(fill_time),
                    "fill_step_idx": i
                }

    # BEARISH SETUP: H1 Trend Bearish + Sweep above Session High
    elif slope < -0.10:
        recent_h = highs[i - 3 : i + 1]
        if max(recent_h) > s_high and c_now < s_high + 0.50:
            entry = round(s_high - 0.10, 2)
            sl = round(max(recent_h) + RISK_AMOUNT + SPREAD_FEE, 2)
            tp = round(entry - REWARD_AMOUNT - SPREAD_FEE, 2)

            filled = False
            hit_tp = False
            stopped = False
            fill_time = 0

            for k in range(len(fwd_h)):
                if not filled:
                    if k < MAX_FILL_MINUTES and fwd_h[k] >= entry:
                        filled = True
                        fill_time = k + 1
                        if fwd_h[k] >= sl:
                            stopped = True; break
                        if fwd_l[k] <= tp:
                            hit_tp = True; break
                    elif k >= MAX_FILL_MINUTES:
                        break
                else:
                    if fwd_h[k] >= sl:
                        stopped = True; break
                    if fwd_l[k] <= tp:
                        hit_tp = True; break

            if filled and hit_tp and not stopped:
                found_trade = True
                trade_meta = {
                    "direction": 1, # SELL_LIMIT
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "time_to_fill": float(fill_time),
                    "fill_step_idx": i
                }

    if found_trade:
        if cur_date not in day_candidates:
            day_candidates[cur_date] = []
        day_candidates[cur_date].append(trade_meta)

print(f"   Identified potential 1:5 setups across {len(day_candidates)} active trading days.")

# Enforce Top-1 Daily Constraint: Pick the earliest/cleanest fill per day
top1_positive_indices = set()
for d, candidates in day_candidates.items():
    best_cand = min(candidates, key=lambda c: c['time_to_fill'])
    top1_positive_indices.add(best_cand['fill_step_idx'])

print(f"   Constrained to exactly {len(top1_positive_indices)} pristine Top-1 daily setups.")

# 4. Build Input & Target Arrays
X_micro_list = []
X_macro_list = []
y_confirm_list = []
y_direction_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
y_ttf_list = []
preview_samples = []

for i in range(start_idx, end_idx):
    cur_date = dates[i]
    c_now = closes[i]

    # Micro features: (60, 5)
    w_o = opens[i - 60 + 1 : i + 1]
    w_h = highs[i - 60 + 1 : i + 1]
    w_l = lows[i - 60 + 1 : i + 1]
    w_c = closes[i - 60 + 1 : i + 1]
    w_mom = (w_c - w_o) / (w_h - w_l + 1e-4)

    micro_feat = np.stack([w_o, w_h, w_l, w_c, w_mom], axis=-1) # (60, 5)

    # Macro features: (12)
    s_range = session_30m_ranges.get(cur_date, {'high': c_now, 'low': c_now})
    mins_in_session = float((hours[i] - 12) * 60 + minutes[i]) if hours[i] >= 12 else 0.0

    macro_feat = np.array([
        pdh_arr[i] - c_now,
        c_now - pdl_arr[i],
        pwh_arr[i] - c_now,
        c_now - pwl_arr[i],
        h1_slopes[i],
        s_range['high'] - c_now,
        c_now - s_range['low'],
        2.0 if 12 <= hours[i] < 17 else (1.0 if 7 <= hours[i] < 12 else 0.0), # Session ID
        mins_in_session,
        float(hours[i]),
        float(dows[i]),
        SPREAD_FEE
    ], dtype=np.float32)

    X_micro_list.append(micro_feat)
    X_macro_list.append(macro_feat)

    if i in top1_positive_indices:
        # Find meta
        meta = [c for c in day_candidates[cur_date] if c['fill_step_idx'] == i][0]
        y_confirm_list.append(1)
        y_direction_list.append(meta['direction'])
        y_entry_list.append(meta['entry'])
        y_sl_list.append(meta['sl'])
        y_tp_list.append(meta['tp'])
        y_ttf_list.append(meta['time_to_fill'])
        if len(preview_samples) < 5:
            preview_samples.append({
                "date": str(cur_date),
                "time_utc": f"{hours[i]:02d}:{minutes[i]:02d}",
                "ref_close": float(c_now),
                "setup": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in meta.items()}
            })
    else:
        y_confirm_list.append(0)
        y_direction_list.append(0)
        y_entry_list.append(float(c_now))
        y_sl_list.append(float(c_now))
        y_tp_list.append(float(c_now))
        y_ttf_list.append(0.0)

X_micro = np.array(X_micro_list, dtype=np.float32)
X_macro = np.array(X_macro_list, dtype=np.float32)
y_confirm = np.array(y_confirm_list, dtype=np.int64)
y_direction = np.array(y_direction_list, dtype=np.int64)
y_entry = np.array(y_entry_list, dtype=np.float32)
y_sl = np.array(y_sl_list, dtype=np.float32)
y_tp = np.array(y_tp_list, dtype=np.float32)
y_ttf = np.array(y_ttf_list, dtype=np.float32)

print("3. Saving Alpha dataset...")
np.savez_compressed(
    SAVE_PATH_NPZ,
    X_micro=X_micro,
    X_macro=X_macro,
    y_confirm=y_confirm,
    y_direction=y_direction,
    y_entry=y_entry,
    y_sl=y_sl,
    y_tp=y_tp,
    y_ttf=y_ttf
)

with open(SAVE_PATH_PREVIEW, "w") as f:
    json.dump({
        "total_samples": len(y_confirm),
        "positive_top1_trades": int((y_confirm == 1).sum()),
        "samples_preview": preview_samples
    }, f, indent=2)

print(f"Dataset successfully created and saved to {SAVE_PATH_NPZ}!")
print(f"Total bars: {len(y_confirm):,} | Pristine Top-1 Setups: {(y_confirm == 1).sum():,}")
