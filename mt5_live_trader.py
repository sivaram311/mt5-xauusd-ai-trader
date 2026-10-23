"""
Live Automated Triple-Engine MT5 Trading Daemon for XAUUSD
Engine 1: M15 Intraday Trend Model  (Magic: 778899, Comment: M15_CONF_XX)
Engine 2: M5 Continuous Scalper     (Magic: 778855, Comment: M5_SCALP_XX)
Engine 3: M1 Directional & RL Sniper V3 (Magic: 778811, Comment: M1_V3_XX)
- Polling cycle: Every 3-5 seconds
- Multi-pending order support with duplicate tolerance protection ($0.25)
- Auto-reconnection & IPC health monitoring
- Pure static SL/TP targets (Zero trailing stop interference)
- 25-minute auto-timeout cancellation on M1 Sniper orders
- Hard Directional Guardrails: Blocks BUY during steep H4 downtrend (slope < -0.5)
"""
import time
import datetime
import logging
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from predict_order import predict_pending_order_from_100_candles
from predict_m5_hwr import predict_m5_hwr_order
from predict_m1_v3_directional import predict_m1_v3_directional_order

# Terminal & Symbol settings
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
TIMEFRAME_M5 = mt5.TIMEFRAME_M5
TIMEFRAME_M1 = mt5.TIMEFRAME_M1

# Strategy 1: M15 Intraday Settings
M15_MAGIC = 778899
M15_CONF_GATE = 0.43          # Restored balanced M15 gate
M15_EXPIRY_HOURS = 12
M15_RISK_PCT = 1.0

# Strategy 2: M5 High-Win-Rate Liquidity Engine (1:2.6 R:R, PF 2.36)
M5_MAGIC = 778855
M5_CONF_GATE = 0.55           # High-Conviction M5 Gate (42% to 48% Win Rate)
M5_EXPIRY_MINUTES = 30        # 6 bars (30 mins auto-expiry)
M5_RISK_PCT = 0.75            # 0.75% risk per trade

# Strategy 3: M1 Directional & RL Sniper V3 Settings
M1_MAGIC = 778811
M1_CONF_GATE = 0.55           # High selectivity sniper gate
M1_EXPIRY_MINUTES = 25        # Auto-timeout cancellation
M1_RISK_PCT = 0.35            # 0.35% risk per sniper trade

# Execution safeguards
SLIPPAGE_TOLERANCE = 10      # in points
MAX_SPREAD_POINTS = 0.45     # Max acceptable spread ($0.45)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"W:\data\trading\mt5_live_trader.log"),
        logging.StreamHandler()
    ]
)

def connect_mt5():
    if not mt5.initialize(path=TERMINAL_PATH):
        logging.error(f"MT5 Init error: {mt5.last_error()}")
        return False
    if not mt5.symbol_select(SYMBOL, True):
        logging.error(f"Failed to select symbol {SYMBOL}")
        return False
    term_info = mt5.terminal_info()
    logging.info(f"Connected to MT5: {term_info.name} | Account: {mt5.account_info().login} | Server: {term_info.company}")
    return True

def ensure_connection():
    term_info = mt5.terminal_info()
    if term_info is None or not term_info.connected:
        logging.warning("MT5 disconnected or IPC dropped. Re-initializing...")
        mt5.shutdown()
        time.sleep(2)
        return connect_mt5()
    return True

def get_account_balance():
    info = mt5.account_info()
    return info.balance if info else 1000.0

def get_broker_time():
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick and tick.time > 0:
        return tick.time
    return int(time.time()) + 10800  # Fallback: UTC+3 broker offset

def calculate_lot(balance, risk_pct, risk_points):
    risk_dollars = balance * (risk_pct / 100.0)
    lot = risk_dollars / (risk_points * 100.0)
    lot = max(0.01, round(lot, 2))
    return min(lot, 5.0)

def check_spread_safe():
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        return False, 0.0
    spread = round(abs(tick.ask - tick.bid), 2)
    return (spread <= MAX_SPREAD_POINTS), spread

def cancel_pending_orders_by_magic(magic_number, strategy_tag):
    orders = mt5.orders_get(symbol=SYMBOL)
    if not orders:
        return
    for o in orders:
        if o.magic == magic_number and o.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket}
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{strategy_tag}] Canceled pending order #{o.ticket}")
            else:
                logging.warning(f"[{strategy_tag}] Could not cancel order #{o.ticket}: {res.comment}")

def has_duplicate_pending_order(magic_number, entry_price, tolerance=0.25):
    orders = mt5.orders_get(symbol=SYMBOL)
    if not orders:
        return False
    for o in orders:
        if o.magic == magic_number and o.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
            if abs(o.price_open - entry_price) <= tolerance:
                return True
    return False

def get_macro_levels():
    r15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 1)
    rh4 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 50)
    rd1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 1)
    rw1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_W1, 0, 2)
    
    if r15 is None or rh4 is None or rd1 is None or rw1 is None or len(rw1) < 2 or len(rh4) < 20:
        return None

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    time_of_day = (now_utc.hour * 3600 + now_utc.minute * 60 + now_utc.second) / 86400.0

    # Calculate 20-period H4 Trend Slope and ATR normalizer
    h4_closes = pd.Series([r['close'] for r in rh4])
    h4_highs = pd.Series([r['high'] for r in rh4])
    h4_lows = pd.Series([r['low'] for r in rh4])
    h4_ema20 = h4_closes.ewm(span=20, adjust=False).mean()
    h4_tr = np.maximum(h4_highs - h4_lows, np.abs(h4_highs - h4_closes.shift(1).fillna(h4_closes.iloc[0])))
    h4_atr = h4_tr.rolling(20).mean().bfill()
    h4_slope = float((h4_closes.iloc[-1] - h4_ema20.iloc[-1]) / (h4_atr.iloc[-1] + 1e-4))

    return {
        'w2_high': float(max(rw1[0]['high'], rw1[1]['high'])),
        'w2_low': float(min(rw1[0]['low'], rw1[1]['low'])),
        'w1_high': float(rw1[1]['high']),
        'w1_low': float(rw1[1]['low']),
        'd1_high': float(rd1[0]['high']),
        'd1_low': float(rd1[0]['low']),
        'h4_high': float(rh4[-1]['high']),
        'h4_low': float(rh4[-1]['low']),
        'm15_high': float(r15[0]['high']),
        'm15_low': float(r15[0]['low']),
        'h4_slope': h4_slope,
        'time_of_day': time_of_day
    }

# --- M15 Cycle ---
def run_m15_cycle():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) != 100:
        return

    candles = [{"open": float(r['open']), "high": float(r['high']), "low": float(r['low']), "close": float(r['close'])} for r in rates]
    signal = predict_pending_order_from_100_candles(candles)
    bar_time = datetime.datetime.fromtimestamp(rates[-1]['time'], datetime.timezone.utc).strftime('%H:%M:%S UTC')
    logging.info(f"[M15_CYCLE] Candle {bar_time} | Signal: {signal['order_type']} | Conf: {signal.get('confidence', 0)}")

    if signal["order_type"] in ["BUY_LIMIT", "SELL_LIMIT"] and signal.get("confidence", 0) >= M15_CONF_GATE:
        is_safe, spread = check_spread_safe()
        if not is_safe:
            logging.warning(f"[M15] Spread ${spread} exceeds limit. Skipping order.")
            return

        if has_duplicate_pending_order(M15_MAGIC, signal["entry"]):
            logging.info(f"[M15] Order already active at ${signal['entry']} (tolerance $0.25). Skipping duplicate.")
            return

        balance = get_account_balance()
        lot = calculate_lot(balance, M15_RISK_PCT, signal["risk_points"])
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal["order_type"] == "BUY_LIMIT" else mt5.ORDER_TYPE_SELL_LIMIT
        expiration = get_broker_time() + (M15_EXPIRY_HOURS * 3600)
        comment_str = f"M15_CONF_{int(signal['confidence']*100)}"

        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": lot,
            "type": order_type,
            "price": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "deviation": SLIPPAGE_TOLERANCE,
            "magic": M15_MAGIC,
            "comment": comment_str,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiration
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[M15] SUCCESS: Placed {signal['order_type']} #{res.order} | Lot: {lot} | Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} | Comment: {comment_str}")
        else:
            logging.error(f"[M15] Failed order: {res.retcode} ({res.comment})")

# --- M5 High-Win-Rate Liquidity Engine (48 Bars + 1:2.6 R:R) ---
def run_m5_cycle():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M5, 0, 48)
    if rates is None or len(rates) != 48:
        return

    macro_levels = get_macro_levels()
    if macro_levels is None:
        return

    candles = [{"open": float(r['open']), "high": float(r['high']), "low": float(r['low']), "close": float(r['close']), "time": int(r['time'])} for r in rates]
    signal = predict_m5_hwr_order(candles, macro_levels)
    bar_time = datetime.datetime.fromtimestamp(rates[-1]['time'], datetime.timezone.utc).strftime('%H:%M:%S UTC')
    logging.info(f"[M5_HWR] Candle {bar_time} | Signal: {signal['order_type']} | Conf: {signal.get('confidence', 0)} | RR: {signal.get('rr', 'N/A')}")

    if signal["order_type"] in ["BUY_LIMIT", "SELL_LIMIT"] and signal.get("confidence", 0) >= M5_CONF_GATE:
        is_safe, spread = check_spread_safe()
        if not is_safe:
            logging.warning(f"[M5_HWR] Spread ${spread} exceeds limit. Skipping order.")
            return

        if has_duplicate_pending_order(M5_MAGIC, signal["entry"]):
            logging.info(f"[M5_HWR] Order already active at ${signal['entry']} (tolerance $0.25). Skipping duplicate.")
            return

        balance = get_account_balance()
        lot = calculate_lot(balance, M5_RISK_PCT, signal["risk_pts"])
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal["order_type"] == "BUY_LIMIT" else mt5.ORDER_TYPE_SELL_LIMIT
        timeout_mins = signal.get("time_to_fill_mins", M5_EXPIRY_MINUTES)
        expiration = get_broker_time() + (timeout_mins * 60)
        comment_str = f"M5_HWR_{int(signal['confidence']*100)}"

        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": lot,
            "type": order_type,
            "price": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "deviation": SLIPPAGE_TOLERANCE,
            "magic": M5_MAGIC,
            "comment": comment_str,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiration
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[M5_HWR] SUCCESS: Placed {signal['order_type']} #{res.order} | Lot: {lot} | Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} | RR: {signal['rr']} | Comment: {comment_str}")
        else:
            logging.error(f"[M5_HWR] Failed order: {res.retcode} ({res.comment})")

# --- M1 Directional Sniper V3 Cycle (180 Bars + RL Direction Awareness) ---
def run_m1_cycle():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M1, 0, 180)
    if rates is None or len(rates) != 180:
        return

    macro_levels = get_macro_levels()
    if macro_levels is None:
        return

    candles = [{"open": float(r['open']), "high": float(r['high']), "low": float(r['low']), "close": float(r['close'])} for r in rates]
    signal = predict_m1_v3_directional_order(candles, macro_levels)
    bar_time = datetime.datetime.fromtimestamp(rates[-1]['time'], datetime.timezone.utc).strftime('%H:%M:%S UTC')
    logging.info(f"[M1_V3_SNIPER] Candle {bar_time} | Signal: {signal['order_type']} | Conf: {signal.get('confidence', 0)} | H4_Slope: {signal.get('h4_slope', 'N/A')} | RR: {signal.get('rr', 'N/A')}")

    if signal["order_type"] in ["BUY_LIMIT", "SELL_LIMIT"] and signal.get("confidence", 0) >= M1_CONF_GATE:
        is_safe, spread = check_spread_safe()
        if not is_safe:
            logging.warning(f"[M1_V3] Spread ${spread} exceeds limit. Skipping sniper order.")
            return

        if has_duplicate_pending_order(M1_MAGIC, signal["entry"]):
            logging.info(f"[M1_V3] Sniper order already active at ${signal['entry']} (tolerance $0.25). Skipping duplicate.")
            return

        balance = get_account_balance()
        lot = calculate_lot(balance, M1_RISK_PCT, signal["risk_pts"])
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal["order_type"] == "BUY_LIMIT" else mt5.ORDER_TYPE_SELL_LIMIT
        expiration = get_broker_time() + (M1_EXPIRY_MINUTES * 60) # 25-minute auto-timeout
        comment_str = f"M1_V3_{int(signal['confidence']*100)}"

        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": lot,
            "type": order_type,
            "price": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "deviation": SLIPPAGE_TOLERANCE,
            "magic": M1_MAGIC,
            "comment": comment_str,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiration
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[M1_V3] SUCCESS: Placed {signal['order_type']} #{res.order} | Lot: {lot} | Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} | RR: {signal['rr']} | Comment: {comment_str}")
        else:
            logging.error(f"[M1_V3] Failed sniper order: {res.retcode} ({res.comment})")

def main():
    logging.info("Starting Triple-Engine MT5 Trading Daemon (M15 Intraday + M5 Scalper + M1 Sniper V3 RL)...")
    if not connect_mt5():
        return

    last_m15_time = None
    last_m5_time = None
    last_m1_time = None

    try:
        while True:
            ensure_connection()

            # Check M15 bar update
            r_m15 = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M15, 0, 1)
            if r_m15 is not None and len(r_m15) > 0:
                t15 = r_m15[0]['time']
                if t15 != last_m15_time:
                    last_m15_time = t15
                    run_m15_cycle()

            # Check M5 bar update
            r_m5 = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M5, 0, 1)
            if r_m5 is not None and len(r_m5) > 0:
                t5 = r_m5[0]['time']
                if t5 != last_m5_time:
                    last_m5_time = t5
                    run_m5_cycle()

            # Check M1 bar update
            r_m1 = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M1, 0, 1)
            if r_m1 is not None and len(r_m1) > 0:
                t1 = r_m1[0]['time']
                if t1 != last_m1_time:
                    last_m1_time = t1
                    run_m1_cycle()

            time.sleep(3)
    except KeyboardInterrupt:
        logging.info("Daemon stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
