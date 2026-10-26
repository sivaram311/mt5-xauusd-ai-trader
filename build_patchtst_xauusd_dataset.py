"""
Institutional Dataset Generator for Unified Multi-Task PatchTST Engine (Idea #7)
---------------------------------------------------------------------------------
Pillars Integrated:
1. Feature Engineering:
   - Micro Sequence: 64 M5 Bars (5+ hours) -> [O, H, L, C, Volume, UpperWick, LowerWick]
   - Macro Context Vector:
     * Distance to Asian Session High, Low, and Mid
     * Distance to Prior Day High (PDH) and Low (PDL)
     * London Open Reference Offset
     * 14-period ATR Volatility
     * H1 EMA50 Trend Slope
     * Session ID (Asian: 0, London: 1, NY: 2, Rollover: 3)
     * Hour UTC & Day of Week
2. Supervision:
   - Triple-Barrier Method:
     * Upper Barrier (TP): +2.5 * ATR (Guaranteed 1:2.5+ Asymmetric Payoff)
     * Lower Barrier (SL): -1.0 * ATR
     * Vertical Barrier: 24 M5 bars (2-hour maximum hold horizon to eliminate dead capital)
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
N_M5_BARS = 50000        # ~9 months of continuous M5 data
SPREAD_FEE = 0.35        # Real broker spread ($0.35 per oz)
LOOKBACK_M5 = 64         # 64 M5 bars (ideal patch length: 8 patches x 8 bars)
MAX_FILL_BARS = 4        # 20-minute limit order fill window
MAX_HOLD_BARS = 24       # 2-hour trade horizon
TARGET_RR = 2.50         # Institutional 1:2.50 Risk-to-Reward ratio

SAVE_PATH_NPZ = r"W:\data\trading\xauusd_unified_patchtst_dataset.npz"

print("1. Connecting to MT5 and fetching historical M5, H1, D1 bars...")
if not mt5.initialize(path=TERMINAL_PATH):
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, N_M5_BARS)
rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 4500)
rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 300)
mt5.shutdown()

if rates_m5 is None or len(rates_m5) == 0:
    print("Error: Could not retrieve M5 rates.")
    sys.exit(1)

print(f"   Retrieved {len(rates_m5):,} M5 bars. Computing Session Liquidity features...")

df = pd.DataFrame(rates_m5)
df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
df['date'] = df['dt'].dt.date
df['hour'] = df['dt'].dt.hour
df['minute'] = df['dt'].dt.minute
df['dow'] = df['dt'].dt.dayofweek

# 1. Macro Indicators: H1 Trend Slope
df_h1 = pd.DataFrame(rates_h1)
df_h1['dt'] = pd.to_datetime(df_h1['time'], unit='s', utc=True)
df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
df_h1['h1_slope'] = (df_h1['close'] - df_h1['ema50']) / (df_h1['high'] - df_h1['low'] + 1e-4)

df = pd.merge_asof(
    df.sort_values('dt'),
    df_h1[['dt', 'h1_slope']].sort_values('dt'),
    on='dt',
    direction='backward'
)
df['h1_slope'] = df['h1_slope'].fillna(0.0)

# 2. Prior Day High (PDH) & Prior Day Low (PDL)
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

# 3. ATR(14)
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

# 4. Asian Session Range (00:00 to 07:00 UTC)
# Compute daily Asian High, Low, Mid
asian_mask = (df['hour'] >= 0) & (df['hour'] < 7)
df_asian = df[asian_mask].groupby('date').agg(
    asian_high=('high', 'max'),
    asian_low=('low', 'min')
).reset_index()
df_asian['asian_mid'] = (df_asian['asian_high'] + df_asian['asian_low']) / 2.0

df = pd.merge(df, df_asian, on='date', how='left')
df['asian_high'] = df['asian_high'].bfill()
df['asian_low'] = df['asian_low'].bfill()
df['asian_mid'] = df['asian_mid'].bfill()

# 5. Microstructure Candle Wicks
candle_range = (df['high'] - df['low']) + 1e-4
df['u_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
df['l_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range

print("2. Scanning Triple-Barrier Institutional Setups...")

opens = df['open'].values.astype(np.float32)
highs = df['high'].values.astype(np.float32)
lows = df['low'].values.astype(np.float32)
closes = df['close'].values.astype(np.float32)
volumes = df['tick_volume'].values.astype(np.float32)
u_wicks = df['u_wick'].values.astype(np.float32)
l_wicks = df['l_wick'].values.astype(np.float32)
hours = df['hour'].values
minutes = df['minute'].values
dows = df['dow'].values
atrs = df['atr'].values.astype(np.float32)
h1_slopes = df['h1_slope'].values.astype(np.float32)
pdh_arr = df['pdh'].values.astype(np.float32)
pdl_arr = df['pdl'].values.astype(np.float32)
asia_h = df['asian_high'].values.astype(np.float32)
asia_l = df['asian_low'].values.astype(np.float32)
asia_m = df['asian_mid'].values.astype(np.float32)

n_total = len(df)
start_idx = LOOKBACK_M5
end_idx = n_total - MAX_HOLD_BARS - 2

X_micro_list = []
X_macro_list = []
y_action_list = []      # 0: NO_ORDER, 1: BUY, 2: SELL
y_entry_list = []
y_sl_list = []
y_tp_list = []

valid_buys = 0
valid_sells = 0

for i in range(start_idx, end_idx):
    c_now = closes[i]
    atr_now = atrs[i]
    hr = hours[i]
    slope = h1_slopes[i]

    # Session ID: Asian (0), London (1), NY (2), Rollover/Dead (3)
    if 0 <= hr < 7: sess_id = 0.0
    elif 7 <= hr < 13: sess_id = 1.0
    elif 13 <= hr < 20: sess_id = 2.0
    else: sess_id = 3.0

    # Trade during active institutional sessions
    in_session = (sess_id in [1.0, 2.0])

    # Micro stream (64 bars, 7 features) -> [O, H, L, C, Vol, U_Wick, L_Wick]
    w_o = opens[i - LOOKBACK_M5 + 1 : i + 1]
    w_h = highs[i - LOOKBACK_M5 + 1 : i + 1]
    w_l = lows[i - LOOKBACK_M5 + 1 : i + 1]
    w_c = closes[i - LOOKBACK_M5 + 1 : i + 1]
    w_v = volumes[i - LOOKBACK_M5 + 1 : i + 1]
    w_u = u_wicks[i - LOOKBACK_M5 + 1 : i + 1]
    w_lw = l_wicks[i - LOOKBACK_M5 + 1 : i + 1]

    micro_feat = np.stack([w_o, w_h, w_l, w_c, w_v, w_u, w_lw], axis=-1) # (64, 7)

    # Macro Context Vector (12 features)
    macro_feat = np.array([
        float(c_now - asia_h[i]),      # Dist to Asian High
        float(c_now - asia_l[i]),      # Dist to Asian Low
        float(c_now - asia_m[i]),      # Dist to Asian Mid
        float(pdh_arr[i] - c_now),     # Dist to Prior Day High
        float(c_now - pdl_arr[i]),     # Dist to Prior Day Low
        float(slope),                  # H1 Trend Slope
        float(atr_now),                # Dynamic Volatility
        float(max(w_h[-12:]) - c_now), # 1-hour High dist
        float(c_now - min(w_l[-12:])), # 1-hour Low dist
        float(sess_id),                # Session ID
        float(hr),                     # Hour
        float(dows[i])                 # Day of Week
    ], dtype=np.float32)

    found_action = 0 # Default NO_ORDER
    t_entry = 0.0
    t_sl = 0.0
    t_tp = 0.0

    if in_session:
        fwd_h = highs[i + 1 : i + 1 + MAX_HOLD_BARS]
        fwd_l = lows[i + 1 : i + 1 + MAX_HOLD_BARS]

        # 1. Evaluate BUY Setup (Liquidity Grab / Trend Pullback)
        if slope >= -0.15:
            cand_entry = round(float(min(w_l[-3:])) + 0.20, 2)
            cand_sl = round(cand_entry - (1.1 * atr_now) - SPREAD_FEE, 2)
            risk = cand_entry - cand_sl

            if 2.20 <= risk <= 4.20 and cand_entry < c_now:
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
                    found_action = 1 # BUY
                    t_entry = cand_entry
                    t_sl = cand_sl
                    t_tp = cand_tp
                    valid_buys += 1

        # 2. Evaluate SELL Setup (Liquidity Grab / Trend Pullback)
        if found_action == 0 and slope <= 0.15:
            cand_entry = round(float(max(w_h[-3:])) - 0.20, 2)
            cand_sl = round(cand_entry + (1.1 * atr_now) + SPREAD_FEE, 2)
            risk = cand_sl - cand_entry

            if 2.20 <= risk <= 4.20 and cand_entry > c_now:
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
                    found_action = 2 # SELL
                    t_entry = cand_entry
                    t_sl = cand_sl
                    t_tp = cand_tp
                    valid_sells += 1

    X_micro_list.append(micro_feat)
    X_macro_list.append(macro_feat)
    y_action_list.append(found_action)
    y_entry_list.append(t_entry)
    y_sl_list.append(t_sl)
    y_tp_list.append(t_tp)

print(f"3. Compiling Unified Dataset: {len(y_action_list):,} total samples.")
print(f"   Triple-Barrier Valid BUYs:  {valid_buys:,}")
print(f"   Triple-Barrier Valid SELLs: {valid_sells:,}")

X_micro_arr = np.array(X_micro_list, dtype=np.float32)
X_macro_arr = np.array(X_macro_list, dtype=np.float32)
y_action_arr = np.array(y_action_list, dtype=np.int64)
y_entry_arr = np.array(y_entry_list, dtype=np.float32)
y_sl_arr = np.array(y_sl_list, dtype=np.float32)
y_tp_arr = np.array(y_tp_list, dtype=np.float32)
ref_closes = closes[start_idx:end_idx].astype(np.float32)

np.savez_compressed(
    SAVE_PATH_NPZ,
    X_micro=X_micro_arr,
    X_macro=X_macro_arr,
    y_action=y_action_arr,
    y_entry=y_entry_arr,
    y_sl=y_sl_arr,
    y_tp=y_tp_arr,
    ref_closes=ref_closes
)

print(f"Dataset successfully saved to: {SAVE_PATH_NPZ} ({os.path.getsize(SAVE_PATH_NPZ)/(1024*1024):.2f} MB)")
