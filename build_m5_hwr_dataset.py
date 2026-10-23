"""
Dataset Generator for M5 High-Win-Rate Liquidity Engine (1:2.5 to 1:3 R:R)
Architecture:
- Micro Stream: 48 M5 Candles (4 Hours of Structure) -> [Open, High, Low, Close, Minute, Hour, DayOfWeek]
- Macro Stream: 10 Context Features -> [Distance to PDH, Distance to PDL, Distance to PWH, Distance to PWL,
                                        H1 Trend Slope, Session 30m High dist, Session 30m Low dist,
                                        Session ID, Hour UTC, DayOfWeek]
Target Geometry:
- Structural SL: $2.80 to $3.50 (incorporating $0.35 broker spread)
- Target TP: $8.00 to $10.00 (1:2.5 to 1:3 Asymmetric Payoff)
- Max Fill Window: 6 M5 bars (30 minutes)
- Max Trade Horizon: 24 M5 bars (2 hours)
- Active Trading Hours: London & NY (07:00 to 18:00 UTC)
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
N_M5_BARS = 35000       # ~6 months of continuous M5 data
SPREAD_FEE = 0.35       # Real broker spread ($0.35)
LOOKBACK_M5 = 48        # 48 M5 bars (4 hours lookback)
MAX_FILL_BARS = 6       # 30-minute auto-expiry on pending limit
MAX_HOLD_BARS = 24      # 2-hour max trade duration
MIN_RISK = 2.50         # $2.50 min SL (immunizes against M5 wicks)
MAX_RISK = 3.80         # $3.80 max SL
TARGET_RR = 2.60        # 1:2.60 R:R (~$8.00 to $9.50 TP)

SAVE_PATH_NPZ = r"W:\data\trading\xauusd_m5_hwr_dataset.npz"
SAVE_PATH_PREVIEW = r"W:\data\trading\m5_hwr_preview.json"

print("1. Connecting to MT5 and fetching M5, H1, D1, W1 candles...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, N_M5_BARS)
rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 3000)
rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 200)
rates_w1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_W1, 0, 40)
mt5.shutdown()

if rates_m5 is None or len(rates_m5) == 0:
    print("Error: Could not retrieve M5 rates.")
    sys.exit(1)

print(f"   Retrieved {len(rates_m5):,} M5 bars, {len(rates_h1):,} H1 bars, {len(rates_d1):,} D1 bars.")

df_m5 = pd.DataFrame(rates_m5)
df_m5['dt'] = pd.to_datetime(df_m5['time'], unit='s', utc=True)
df_m5['date'] = df_m5['dt'].dt.date
df_m5['hour'] = df_m5['dt'].dt.hour
df_m5['minute'] = df_m5['dt'].dt.minute
df_m5['dow'] = df_m5['dt'].dt.dayofweek

# 2. Macro Indicators: H1 Trend Direction
df_h1 = pd.DataFrame(rates_h1)
df_h1['dt'] = pd.to_datetime(df_h1['time'], unit='s', utc=True)
df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
df_h1['h1_slope'] = (df_h1['close'] - df_h1['ema50']) / (df_h1['high'] - df_h1['low'] + 1e-4)

df_m5 = pd.merge_asof(
    df_m5.sort_values('dt'),
    df_h1[['dt', 'h1_slope']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df_m5['h1_slope'] = df_m5['h1_slope'].fillna(0.0)

# Multi-Timeframe Levels (PDH, PDL, PWH, PWL)
df_d1 = pd.DataFrame(rates_d1)
df_d1['dt'] = pd.to_datetime(df_d1['time'], unit='s', utc=True)
df_d1['pdh'] = df_d1['high'].shift(1)
df_d1['pdl'] = df_d1['low'].shift(1)

df_m5 = pd.merge_asof(
    df_m5.sort_values('dt'),
    df_d1[['dt', 'pdh', 'pdl']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df_m5['pdh'] = df_m5['pdh'].bfill()
df_m5['pdl'] = df_m5['pdl'].bfill()

df_w1 = pd.DataFrame(rates_w1)
df_w1['dt'] = pd.to_datetime(df_w1['time'], unit='s', utc=True)
df_w1['pwh'] = df_w1['high'].shift(1)
df_w1['pwl'] = df_w1['low'].shift(1)

df_m5 = pd.merge_asof(
    df_m5.sort_values('dt'),
    df_w1[['dt', 'pwh', 'pwl']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df_m5['pwh'] = df_m5['pwh'].bfill()
df_m5['pwl'] = df_m5['pwl'].bfill()

print("2. Scanning M5 Liquidity Reversals and Continuations...")

n_total = len(df_m5)
opens = df_m5['open'].values.astype(np.float32)
highs = df_m5['high'].values.astype(np.float32)
lows = df_m5['low'].values.astype(np.float32)
closes = df_m5['close'].values.astype(np.float32)
hours = df_m5['hour'].values
minutes = df_m5['minute'].values
dows = df_m5['dow'].values
h1_slopes = df_m5['h1_slope'].values.astype(np.float32)
pdh_arr = df_m5['pdh'].values.astype(np.float32)
pdl_arr = df_m5['pdl'].values.astype(np.float32)
pwh_arr = df_m5['pwh'].values.astype(np.float32)
pwl_arr = df_m5['pwl'].values.astype(np.float32)

X_micro_list = []
X_macro_list = []
y_confirm_list = []
y_direction_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []
y_ttf_list = []
preview_samples = []

start_idx = LOOKBACK_M5
end_idx = n_total - MAX_HOLD_BARS - 5

for i in range(start_idx, end_idx):
    hr = hours[i]
    c_now = closes[i]

    # Session gate: 06:00 to 19:00 UTC (Skip dead rollover)
    in_session = (6 <= hr <= 19)

    w_o = opens[i - LOOKBACK_M5 + 1 : i + 1]
    w_h = highs[i - LOOKBACK_M5 + 1 : i + 1]
    w_l = lows[i - LOOKBACK_M5 + 1 : i + 1]
    w_c = closes[i - LOOKBACK_M5 + 1 : i + 1]
    w_m = minutes[i - LOOKBACK_M5 + 1 : i + 1].astype(np.float32)
    w_hr = hours[i - LOOKBACK_M5 + 1 : i + 1].astype(np.float32)
    w_dow = dows[i - LOOKBACK_M5 + 1 : i + 1].astype(np.float32)

    micro_feat = np.stack([w_o, w_h, w_l, w_c, w_m, w_hr, w_dow], axis=-1) # (48, 7)

    # Macro vector (10 features)
    macro_feat = np.array([
        pdh_arr[i] - c_now,
        c_now - pdl_arr[i],
        pwh_arr[i] - c_now,
        c_now - pwl_arr[i],
        h1_slopes[i],
        max(w_h[-12:]) - c_now,  # Recent 1-hour high dist
        c_now - min(w_l[-12:]),  # Recent 1-hour low dist
        float(hr),
        float(dows[i]),
        SPREAD_FEE
    ], dtype=np.float32)

    found_trade = False
    trade_meta = None

    if in_session:
        fwd_h = highs[i + 1 : i + 1 + MAX_HOLD_BARS]
        fwd_l = lows[i + 1 : i + 1 + MAX_HOLD_BARS]
        slope = h1_slopes[i]

        # 1. Test BUY_LIMIT Setup (Sweep of recent 3-bar low in bullish/neutral trend)
        if slope >= -0.20:
            swing_low = float(min(w_l[-4:]))
            cand_entry = round(swing_low + 0.20, 2)
            cand_sl = round(swing_low - 2.80 - SPREAD_FEE, 2)
            risk = round(cand_entry - cand_sl, 2)

            if MIN_RISK <= risk <= MAX_RISK and cand_entry < c_now:
                cand_tp = round(cand_entry + (risk * TARGET_RR) + SPREAD_FEE, 2)

                filled = False
                hit_tp = False
                stopped = False
                fill_bar = 0

                for k in range(len(fwd_l)):
                    if not filled:
                        if k < MAX_FILL_BARS and fwd_l[k] <= cand_entry:
                            filled = True
                            fill_bar = k + 1
                            if fwd_l[k] <= cand_sl:
                                stopped = True; break
                            if fwd_h[k] >= cand_tp:
                                hit_tp = True; break
                        elif k >= MAX_FILL_BARS:
                            break
                    else:
                        if fwd_l[k] <= cand_sl:
                            stopped = True; break
                        if fwd_h[k] >= cand_tp:
                            hit_tp = True; break

                if filled and hit_tp and not stopped:
                    found_trade = True
                    trade_meta = {
                        "direction": 0, # BUY_LIMIT
                        "entry": cand_entry,
                        "sl": cand_sl,
                        "tp": cand_tp,
                        "ttf_bars": fill_bar,
                        "risk": risk,
                        "reward": round(cand_tp - cand_entry, 2),
                        "rr": round((cand_tp - cand_entry) / risk, 2)
                    }

        # 2. Test SELL_LIMIT Setup (Sweep of recent 3-bar high in bearish/neutral trend)
        if not found_trade and slope <= 0.20:
            swing_high = float(max(w_h[-4:]))
            cand_entry = round(swing_high - 0.20, 2)
            cand_sl = round(swing_high + 2.80 + SPREAD_FEE, 2)
            risk = round(cand_sl - cand_entry, 2)

            if MIN_RISK <= risk <= MAX_RISK and cand_entry > c_now:
                cand_tp = round(cand_entry - (risk * TARGET_RR) - SPREAD_FEE, 2)

                filled = False
                hit_tp = False
                stopped = False
                fill_bar = 0

                for k in range(len(fwd_h)):
                    if not filled:
                        if k < MAX_FILL_BARS and fwd_h[k] >= cand_entry:
                            filled = True
                            fill_bar = k + 1
                            if fwd_h[k] >= cand_sl:
                                stopped = True; break
                            if fwd_l[k] <= cand_tp:
                                hit_tp = True; break
                        elif k >= MAX_FILL_BARS:
                            break
                    else:
                        if fwd_h[k] >= cand_sl:
                            stopped = True; break
                        if fwd_l[k] <= cand_tp:
                            hit_tp = True; break

                if filled and hit_tp and not stopped:
                    found_trade = True
                    trade_meta = {
                        "direction": 1, # SELL_LIMIT
                        "entry": cand_entry,
                        "sl": cand_sl,
                        "tp": cand_tp,
                        "ttf_bars": fill_bar,
                        "risk": risk,
                        "reward": round(cand_entry - cand_tp, 2),
                        "rr": round((cand_entry - cand_tp) / risk, 2)
                    }

    X_micro_list.append(micro_feat)
    X_macro_list.append(macro_feat)

    if found_trade and trade_meta is not None:
        y_confirm_list.append(1)
        y_direction_list.append(trade_meta['direction'])
        y_entry_list.append(trade_meta['entry'])
        y_sl_list.append(trade_meta['sl'])
        y_tp_list.append(trade_meta['tp'])
        y_ttf_list.append(float(trade_meta['ttf_bars']))
        if len(preview_samples) < 5:
            preview_samples.append({
                "bar_idx": i,
                "current_close": float(c_now),
                "setup": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in trade_meta.items()}
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

print("3. Saving M5 HWR Dataset...")
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
        "positive_setups": int((y_confirm == 1).sum()),
        "positive_pct": float(round((y_confirm == 1).sum() / len(y_confirm) * 100, 2)),
        "samples_preview": preview_samples
    }, f, indent=2)

print(f"Dataset successfully saved to {SAVE_PATH_NPZ}!")
print(f"Total M5 bars: {len(y_confirm):,} | Positive 1:2.6 Setups: {(y_confirm == 1).sum():,} ({round((y_confirm == 1).sum()/len(y_confirm)*100, 2)}%)")
