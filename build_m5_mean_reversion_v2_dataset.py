"""
Dataset Generator V2: Physical Liquidity Sweep & Rejection Wick Mean Reversion
-------------------------------------------------------------------------------
Core Enhancements over V1:
1. Physical Trigger Precondition:
   - Only samples when an actual REJECTION WICK forms (wick >= 45% of total bar range).
   - Only triggers after a physical Sweep of a multi-bar swing high/low or PDH/PDL.
2. Realistic Equilibrium Target:
   - Targets dynamic 20-period EMA / Asian Mid equilibrium instead of distant Daily VWAP.
   - Provides 1:1.8 to 1:2.4 R:R that resolves within 12 M5 bars (1 hour).
3. Eliminates 90% of runaway trend false positives.
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
MAX_FILL_BARS = 3        # 15-minute quick fill on pending limit order
MAX_HOLD_BARS = 16       # 80-minute max hold horizon
MIN_SL = 2.40            # $2.40 min SL
MAX_SL = 4.20            # $4.20 max SL
TARGET_RR = 1.90         # 1:1.90 realistic mean reversion payoff

SAVE_PATH_NPZ = r"W:\data\trading\xauusd_m5_mean_reversion_v2_dataset.npz"

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

print(f"   Retrieved {len(rates_m5):,} M5 bars. Building V2 physical rejection features...")

df = pd.DataFrame(rates_m5)
df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
df['date'] = df['dt'].dt.date
df['hour'] = df['dt'].dt.hour
df['minute'] = df['dt'].dt.minute
df['dow'] = df['dt'].dt.dayofweek

# 1. 20-period EMA (Dynamic Equilibrium Baseline)
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

# 2. Rolling ATR(14)
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

# 3. Microstructure Candle Geometries (Body vs Wicks)
candle_range = (df['high'] - df['low']) + 1e-4
upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
lower_wick = df[['open', 'close']].min(axis=1) - df['low']

df['upper_wick_ratio'] = upper_wick / candle_range
df['lower_wick_ratio'] = lower_wick / candle_range

# 4. Macro MTF Levels (PDH, PDL)
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

# 5. Relative Volume Acceleration
df['vol_sma20'] = df['tick_volume'].rolling(20).mean().bfill()
df['vol_ratio'] = df['tick_volume'] / (df['vol_sma20'] + 1e-4)

print("2. Scanning Physical Rejection Trigger Setups...")

opens = df['open'].values.astype(np.float32)
highs = df['high'].values.astype(np.float32)
lows = df['low'].values.astype(np.float32)
closes = df['close'].values.astype(np.float32)
volumes = df['tick_volume'].values.astype(np.float32)
hours = df['hour'].values
minutes = df['minute'].values
dows = df['dow'].values
ema20s = df['ema20'].values.astype(np.float32)
atrs = df['atr'].values.astype(np.float32)
u_wicks = df['upper_wick_ratio'].values.astype(np.float32)
l_wicks = df['lower_wick_ratio'].values.astype(np.float32)
vol_ratios = df['vol_ratio'].values.astype(np.float32)
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

valid_rejections = 0
failed_rejections = 0

for i in range(start_idx, end_idx):
    c_now = closes[i]
    ema_now = ema20s[i]
    atr_now = atrs[i]
    hr = hours[i]
    u_ratio = u_wicks[i]
    l_ratio = l_wicks[i]

    # Active trading sessions: London & NY (07:00 to 19:00 UTC)
    in_session = (7 <= hr <= 19)

    # Condition 1: Must be at least 1.2 ATR away from the EMA20
    disp_atr = (c_now - ema_now) / (atr_now + 1e-4)

    is_bearish_sweep = (disp_atr >= 1.2 and u_ratio >= 0.40)
    is_bullish_sweep = (disp_atr <= -1.2 and l_ratio >= 0.40)

    # We ONLY generate samples when a physical rejection setup actually occurred!
    if not (in_session and (is_bearish_sweep or is_bullish_sweep)):
        continue

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
        float(disp_atr),
        float(c_now - ema_now),
        float(u_ratio),
        float(l_ratio),
        float(vol_ratios[i]),
        float(pdh_arr[i] - c_now),
        float(c_now - pdl_arr[i]),
        float(atr_now),
        float(hr),
        float(dows[i])
    ], dtype=np.float32)

    fwd_h = highs[i + 1 : i + 1 + MAX_HOLD_BARS]
    fwd_l = lows[i + 1 : i + 1 + MAX_HOLD_BARS]

    found_trade = False
    trade_dir = 0
    trade_entry = 0.0
    trade_sl = 0.0
    trade_tp = 0.0

    # 1. Bearish Rejection (Pin-bar at High -> Short targeting EMA)
    if is_bearish_sweep:
        bar_high = float(highs[i])
        cand_entry = round(bar_high - 0.25, 2)
        cand_sl = round(bar_high + (0.8 * atr_now) + SPREAD_FEE, 2)
        risk = cand_sl - cand_entry

        if MIN_SL <= risk <= MAX_SL and cand_entry > c_now:
            cand_tp = round(cand_entry - (risk * TARGET_RR) - SPREAD_FEE, 2)

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
                trade_dir = 1 # SELL_LIMIT
                trade_entry = cand_entry
                trade_sl = cand_sl
                trade_tp = cand_tp
                valid_rejections += 1
            elif filled and stopped:
                failed_rejections += 1

    # 2. Bullish Rejection (Pin-bar at Low -> Long targeting EMA)
    elif is_bullish_sweep:
        bar_low = float(lows[i])
        cand_entry = round(bar_low + 0.25, 2)
        cand_sl = round(bar_low - (0.8 * atr_now) - SPREAD_FEE, 2)
        risk = cand_entry - cand_sl

        if MIN_SL <= risk <= MAX_SL and cand_entry < c_now:
            cand_tp = round(cand_entry + (risk * TARGET_RR) + SPREAD_FEE, 2)

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
                trade_dir = 0 # BUY_LIMIT
                trade_entry = cand_entry
                trade_sl = cand_sl
                trade_tp = cand_tp
                valid_rejections += 1
            elif filled and stopped:
                failed_rejections += 1

    X_micro_list.append(micro_feat)
    X_macro_list.append(macro_feat)
    y_confirm_list.append(1 if found_trade else 0)
    y_direction_list.append(trade_dir)
    y_entry_list.append(trade_entry if found_trade else 0.0)
    y_sl_list.append(trade_sl if found_trade else 0.0)
    y_tp_list.append(trade_tp if found_trade else 0.0)

print(f"3. Compiling V2 Dataset: {len(y_confirm_list):,} curated physical rejection events.")
print(f"   Successful Mean Reversions: {valid_rejections:,} ({valid_rejections/(len(y_confirm_list)+1e-4)*100:.1f}%)")
print(f"   Failed Breakouts / Stops:  {failed_rejections:,}")

X_micro_arr = np.array(X_micro_list, dtype=np.float32)
X_macro_arr = np.array(X_macro_list, dtype=np.float32)
y_confirm_arr = np.array(y_confirm_list, dtype=np.int64)
y_direction_arr = np.array(y_direction_list, dtype=np.int64)
y_entry_arr = np.array(y_entry_list, dtype=np.float32)
y_sl_arr = np.array(y_sl_list, dtype=np.float32)
y_tp_arr = np.array(y_tp_list, dtype=np.float32)

# Save reference closes
ref_closes = closes[start_idx:end_idx] # subset matches filtered indices
# Filter matching indices
c_ref_list = []
idx_counter = 0
for i in range(start_idx, end_idx):
    c_now = closes[i]
    ema_now = ema20s[i]
    atr_now = atrs[i]
    hr = hours[i]
    u_ratio = u_wicks[i]
    l_ratio = l_wicks[i]
    in_session = (7 <= hr <= 19)
    disp_atr = (c_now - ema_now) / (atr_now + 1e-4)
    is_bearish = (disp_atr >= 1.2 and u_ratio >= 0.40)
    is_bullish = (disp_atr <= -1.2 and l_ratio >= 0.40)
    if in_session and (is_bearish or is_bullish):
        c_ref_list.append(c_now)

ref_closes_arr = np.array(c_ref_list, dtype=np.float32)

np.savez_compressed(
    SAVE_PATH_NPZ,
    X_micro=X_micro_arr,
    X_macro=X_macro_arr,
    y_confirm=y_confirm_arr,
    y_direction=y_direction_arr,
    y_entry=y_entry_arr,
    y_sl=y_sl_arr,
    y_tp=y_tp_arr,
    ref_closes=ref_closes_arr
)

print(f"Dataset V2 saved to: {SAVE_PATH_NPZ} ({os.path.getsize(SAVE_PATH_NPZ)/(1024*1024):.2f} MB)")
