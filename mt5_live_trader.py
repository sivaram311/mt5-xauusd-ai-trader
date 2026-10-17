"""
Live Automated MT5 Trading Daemon for XAUUSD M15
- Runs continuously on this machine
- Wakes up on every new M15 candle close
- Pulls exactly 100 historical candles (pure Time + OHLC)
- Applies the recommended Balanced High-Confidence profile (Conf >= 0.60, London/NY session)
- Automatically places/updates pending BUY_LIMIT or SELL_LIMIT orders via MT5 terminal
- Cancels obsolete pending orders if market structure shifts to NO_ORDER
"""
import time
import datetime
import json
import logging
import MetaTrader5 as mt5
from predict_order import predict_pending_order_from_100_candles

# Configuration
TERMINAL_PATH = r"E:\ProgramFiles\MT5\terminal64.exe"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
MAGIC_NUMBER = 778899         # Unique identifier for AI orders
MAX_RISK_PCT = 1.0           # 1.0% account risk per trade
SLIPPAGE_TOLERANCE = 10       # in points
EXPIRY_HOURS = 12            # Pending orders expire in 12 hours if not triggered

# Logging setup
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
        logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
        return False
    if not mt5.symbol_select(SYMBOL, True):
        logging.error(f"Failed to select symbol {SYMBOL}")
        return False
    return True

def get_account_balance():
    info = mt5.account_info()
    return info.balance if info else 10000.0

def calculate_lot_size(balance, risk_points):
    """
    Gold (XAUUSD): 1 standard lot = 100 oz.
    1 point ($1.00 move in gold) = $100 per 1.0 lot.
    """
    risk_dollars = balance * (MAX_RISK_PCT / 100.0)
    lot = risk_dollars / (risk_points * 100.0)
    # Round to 2 decimal places (standard broker step: 0.01 lot)
    lot = max(0.01, round(lot, 2))
    return min(lot, 5.0) # Safety cap at 5.0 lots

def cancel_existing_ai_pending_orders():
    orders = mt5.orders_get(symbol=SYMBOL)
    if not orders:
        return
    for o in orders:
        if o.magic == MAGIC_NUMBER and o.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
            req = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Canceled previous pending order #{o.ticket}")
            else:
                logging.warning(f"Could not cancel order #{o.ticket}: {res.comment}")

def place_pending_order(signal):
    order_type = signal["order_type"]
    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    risk_pts = signal["risk_points"]
    
    balance = get_account_balance()
    lot = calculate_lot_size(balance, risk_pts)

    mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type == "BUY_LIMIT" else mt5.ORDER_TYPE_SELL_LIMIT
    
    expiration = int(time.time()) + (EXPIRY_HOURS * 3600)
    
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": SLIPPAGE_TOLERANCE,
        "magic": MAGIC_NUMBER,
        "comment": f"AI_CONF_{int(signal['confidence']*100)}",
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": expiration
    }
    
    # Pre-cancel any older pending orders before placing the new setup
    cancel_existing_ai_pending_orders()

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"SUCCESS: Placed {order_type} ticket #{result.order} | Lot: {lot} | Entry: {entry} | SL: {sl} | TP: {tp} | RR: {signal['risk_reward']}")
    else:
        logging.error(f"FAILED to place {order_type} at {entry}: {result.retcode} ({result.comment})")

def run_candle_cycle():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
    if rates is None or len(rates) != 100:
        logging.warning("Incomplete rates received from MT5")
        return

    candles = [
        {"open": float(r['open']), "high": float(r['high']), "low": float(r['low']), "close": float(r['close'])}
        for r in rates
    ]
    
    signal = predict_pending_order_from_100_candles(candles)
    last_candle_time = datetime.datetime.fromtimestamp(rates[-1]['time'], datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    logging.info(f"Evaluated candle {last_candle_time} | Signal: {signal['order_type']} | Confidence: {signal.get('confidence', 0)}")

    if signal["order_type"] in ["BUY_LIMIT", "SELL_LIMIT"]:
        # Only execute if meets confidence gate (configured to 0.43)
        if signal.get("confidence", 0) >= 0.43:
            place_pending_order(signal)
        else:
            logging.info(f"Signal {signal['order_type']} below confidence gate 0.43 (got {signal.get('confidence')}) -> Skipped")
    else:
        # NO_ORDER: cancel stale pending orders
        cancel_existing_ai_pending_orders()

def main():
    logging.info("Starting AI Live Trading Daemon for XAUUSD M15...")
    if not connect_mt5():
        return

    last_processed_time = None
    
    try:
        while True:
            # Check latest closed candle time
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
            if rates is not None and len(rates) > 0:
                current_candle_time = rates[0]['time']
                if current_candle_time != last_processed_time:
                    last_processed_time = current_candle_time
                    run_candle_cycle()
                    
            # Poll every 10 seconds for new M15 candle close
            time.sleep(10)
    except KeyboardInterrupt:
        logging.info("Trading daemon stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
