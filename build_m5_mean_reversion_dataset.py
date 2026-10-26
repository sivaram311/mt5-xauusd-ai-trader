"""
Mean Reversion Dataset Generator for MT5 XAUUSD (M5 Timeframe)
---------------------------------------------------------------
Pillars integrated:
1. Dynamic Equilibrium & Standardized Deviation (Daily VWAP & 20-bar ATR Z-Score)
2. Session Liquidity & Exhaustion (Distances to Asian H/L, PDH/PDL, Wick Rejection, Volume Climax)
3. Triple-Barrier Target Labeling with Time Decay (Target = Return to Equilibrium; Stop = Extreme Wick + Spread)
4. Cost-Sensitive Penalty Ground Truth
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
N_M5_BARS = 40000        # ~7 months of continuous M5 data
SPREAD_FEE = 0.35        # Real broker spread ($0.35 per oz)
LOOKBACK_M5 = 48         # 48 M5 bars (4 hours lookback)
MAX_FILL_BARS = 4        # 20-minute fill window for pending limit
MAX_HOLD_BARS = 24       # 2-hour max hold before time decay expiry
MIN_Z_SCORE = 1.80       # Minimum statistical overextension (|Z| >= 1.8)

SAVE_PATH_NPZ = r"W:\data\trading\xauusd_m5_mean_reversion_dataset.npz"
SAVE_PATH_PREVIEW = r"W:\data\trading\m5_mean_reversion_preview.json"

print("1. Connecting to MT5 and fetching historical M5, H1, D1 bars...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, N_M5_BARS)
rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 3500)
rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 250)
mt5.shutdown()

if rates_m5 is None or len(rates_m5) == 0:
    print("Error: Could not retrieve M5 rates.")
    sys.exit(1)

print(f"   Retrieved {len(rates_m5):,} M5 bars. Building Mean Reversion metrics...")

df = pd.DataFrame(rates_m5)
df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
df['date'] = df['dt'].dt.date
df['hour'] = df['dt'].dt.hour
df['minute'] = df['dt'].dt.minute
df['dow'] = df['dt'].dt.dayofweek

# 1. Compute Daily Anchored VWAP (Volume-Weighted Average Price)
typical_price = (df['high'] + df['low'] + df['close']) / 3.0
pv = typical_price * df['tick_volume']

df['pv'] = pv
df['cum_pv'] = df.groupby('date')['pv'].cumsum()
df['cum_vol'] = df.groupby('date')['tick_volume'].cumsum()
df['vwap'] = df['cum_pv'] / (df['cum_vol'] + 1e-6)

# 2. Rolling ATR(20) and Standardized Z-Score from VWAP
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = tr.rolling(window=20).mean().bfill()
df['z_score'] = (df['close'] - df['vwap']) / (df['atr'] + 1e-4)

# 3. Macro Multi-Timeframe Levels (PDH, PDL)
df_d1 = pd.DataFrame(rates_d1)
df_d1['dt'] = pd.to_datetime(df_d1['time'], unit='s', utc=True)
df_d1['pdh'] = df_d1['high'].shift(1)
df_d1['pdl'] = df_d1['low'].shift(1)

df = pd.merge_asof(
    df.sort_values('dt'),
    df_d1[['dt', 'pdh', 'pdl']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df['pdh'] = df['pdh'].bfill()
df['pdl'] = df['pdl'].bfill()

# 4. Volume Exhaustion Ratio (current tick volume vs 20-bar SMA)
df['vol_sma20'] = df['tick_volume'].rolling(20).mean().bfill()
df['vol_exhaustion'] = df['tick_volume'] / (df['vol_sma20'] + 1e-4)

print("2. Scanning Triple-Barrier Mean Reversion Ground Truth...")

opens = df['open'].values.astype(np.float32)
highs = df['high'].values.astype(np.float32)
lows = df['low'].values.astype(np.float32)
closes = df['close'].values.astype(np.float32)
volumes = df['tick_volume'].values.astype(np.float32)
hours = df['hour'].values
minutes = df['minute'].values
dows = df['dow'].values
vwaps = df['vwap'].values.astype(np.float32)
atrs = df['atr'].values.astype(np.float32)
z_scores = df['z_score'].values.astype(np.float32)
vol_exs = df['vol_exhaustion'].values.astype(np.float32)
pdh_arr = df['pdh'].values.astype(np.float32)
pdl_arr = df['pdl'].values.astype(np.float32)

n_total = len(df)
start_idx = max(LOOKBACK_M5, 30)
end_idx = n_total - MAX_HOLD_BARS - 2

X_micro_list = []
X_macro_list = []
y_confirm_list = []
y_direction_list = []
y_entry_list = []
y_sl_list = []
y_tp_list = []

valid_reversions = 0
continuation_fails = 0

for i in range(start_idx, end_idx):
    c_now = closes[i]
    vwap_now = vwaps[i]
    atr_now = atrs[i]
    z_now = z_scores[i]
    hr = hours[i]

    # Session filter: Active liquidity windows (06:00 to 20:00 UTC)
    in_session = (6 <= hr <= 20)

    # Micro window (48 bars)
    w_o = opens[i - LOOKBACK_M5 + 1 : i + 1]
    w_h = highs[i - LOOKBACK_M5 + 1 : i + 1]
    w_l = lows[i - LOOKBACK_M5 + 1 : i + 1]
    w_c = closes[i - LOOKBACK_M5 + 1 : i + 1]
    w_v = volumes[i - LOOKBACK_M5 + 1 : i + 1]
    w_hr = hours[i - LOOKBACK_M5 + 1 : i + 1].astype(np.float32)
    w_dow = dows[i - LOOKBACK_M5 + 1 : i + 1].astype(np.float32)

    micro_feat = np.stack([w_o, w_h, w_l, w_c, w_v, w_hr, w_dow], axis=-1)

    # Macro Context vector (10 features)
    macro_feat = np.array([
        float(z_now),
        float(c_now - vwap_now),
        float(pdh_arr[i] - c_now),
        float(c_now - pdl_arr[i]),
        float(vol_exs[i]),
        float(atr_now),
        float(c_now - min(w_l[-6:])),  # 30-min low distance
        float(max(w_h[-6:]) - c_now),  # 30-min high distance
        float(hr),
        float(dows[i])
    ], dtype=np.float32)

    found_trade = False
    trade_dir = 0
    trade_entry = 0.0
    trade_sl = 0.0
    trade_tp = 0.0

    if in_session and abs(z_now) >= MIN_Z_SCORE:
        fwd_h = highs[i + 1 : i + 1 + MAX_HOLD_BARS]
        fwd_l = lows[i + 1 : i + 1 + MAX_HOLD_BARS]

        # Case A: Bearish Mean Reversion (Overextended to the upside: Z >= +1.8)
        if z_now >= MIN_Z_SCORE:
            extreme_high = float(max(w_h[-3:]))
            cand_entry = round(extreme_high - 0.20, 2)
            cand_sl = round(extreme_high + (1.2 * atr_now) + SPREAD_FEE, 2)
            cand_tp = round(vwap_now + SPREAD_FEE, 2)

            if cand_entry > cand_tp and (cand_sl - cand_entry) >= 2.0:
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
                    found_trade = True
                    trade_dir = 1 # SELL_LIMIT (Revert down to VWAP)
                    trade_entry = cand_entry
                    trade_sl = cand_sl
                    trade_tp = cand_tp
                    valid_reversions += 1
                elif filled and stopped:
                    continuation_fails += 1

        # Case B: Bullish Mean Reversion (Overextended to the downside: Z <= -1.8)
        elif z_now <= -MIN_Z_SCORE:
            extreme_low = float(min(w_l[-3:]))
            cand_entry = round(extreme_low + 0.20, 2)
            cand_sl = round(extreme_low - (1.2 * atr_now) - SPREAD_FEE, 2)
            cand_tp = round(vwap_now - SPREAD_FEE, 2)

            if cand_entry < cand_tp and (cand_entry - cand_sl) >= 2.0:
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
                    found_trade = True
                    trade_dir = 0 # BUY_LIMIT (Revert up to VWAP)
                    trade_entry = cand_entry
                    trade_sl = cand_sl
                    trade_tp = cand_tp
                    valid_reversions += 1
                elif filled and stopped:
                    continuation_fails += 1

    X_micro_list.append(micro_feat)
    X_macro_list.append(macro_feat)
    y_confirm_list.append(1 if found_trade else 0)
    y_direction_list.append(trade_dir)
    y_entry_list.append(trade_entry if found_trade else 0.0)
    y_sl_list.append(trade_sl if found_trade else 0.0)
    y_tp_list.append(trade_tp if found_trade else 0.0)

print(f"3. Compiling dataset: {len(y_confirm_list):,} total samples.")
print(f"   Successful Mean Reversions: {valid_reversions:,} | Runaway Fails: {continuation_fails:,}")

X_micro_arr = np.array(X_micro_list, dtype=np.float32)
X_macro_arr = np.array(X_macro_list, dtype=np.float32)
y_confirm_arr = np.array(y_confirm_list, dtype=np.int64)
y_direction_arr = np.array(y_direction_list, dtype=np.int64)
y_entry_arr = np.array(y_entry_list, dtype=np.float32)
y_sl_arr = np.array(y_sl_list, dtype=np.float32)
y_tp_arr = np.array(y_tp_list, dtype=np.float32)
ref_closes = closes[start_idx:end_idx].astype(np.float32)

np.savez_compressed(
    SAVE_PATH_NPZ,
    X_micro=X_micro_arr,
    X_macro=X_macro_arr,
    y_confirm=y_confirm_arr,
    y_direction=y_direction_arr,
    y_entry=y_entry_arr,
    y_sl=y_sl_arr,
    y_tp=y_tp_arr,
    ref_closes=ref_closes
)

print(f"Dataset successfully saved to: {SAVE_PATH_NPZ} ({os.path.getsize(SAVE_PATH_NPZ)/(1024*1024):.2f} MB)")
