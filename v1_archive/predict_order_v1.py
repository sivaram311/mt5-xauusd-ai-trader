"""
Inference & Validation Engine for XAUUSD M15 Pending Orders
Takes exactly 100 M15 OHLC candles as input (dict list or dataframe).
Outputs pending order JSON with Entry, SL, TP, and calculated Risk:Reward.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import MetaTrader5 as mt5
import pandas as pd

MODEL_PATH = r"W:\data\trading\best_order_net.pt"

# Model definition matching trained weights
class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.GELU()
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))) + x

class HighAccuracyOrderNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlock1D(64, dilation=1)
        self.block2 = ResBlock1D(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlock1D(128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.trans(h)
        h = self.block3(h)
        pooled = self.pool(h).squeeze(-1)
        feat = self.fc(pooled)
        return self.action_head(feat), self.offset_head(feat)

# Load model into CPU
model = HighAccuracyOrderNet(in_features=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

ACTIONS = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]

def predict_pending_order_from_100_candles(candles_list):
    """
    candles_list: list of 100 dicts: [{'open': float, 'high': float, 'low': float, 'close': float}, ...]
    Returns: JSON response with order details and safety checks
    """
    if len(candles_list) != 100:
        return {"error": f"Expected exactly 100 candles, received {len(candles_list)}"}

    opens = np.array([c['open'] for c in candles_list], dtype=np.float32)
    highs = np.array([c['high'] for c in candles_list], dtype=np.float32)
    lows = np.array([c['low'] for c in candles_list], dtype=np.float32)
    closes = np.array([c['close'] for c in candles_list], dtype=np.float32)
    c100 = closes[-1]

    # Feature engineering without ATR (Relative offsets + anatomy)
    norm_o = opens - c100
    norm_h = highs - c100
    norm_l = lows - c100
    norm_c = closes - c100
    body = closes - opens
    rng = highs - lows

    features = np.stack([norm_o, norm_h, norm_l, norm_c, body, rng], axis=0) # (6, 100)
    tensor_x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)        # (1, 6, 100)

    with torch.no_grad():
        logits, offsets = model(tensor_x)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        action_idx = int(np.argmax(probs))
        action = ACTIONS[action_idx]
        confidence = float(probs[action_idx])
        entry_off, sl_off, tp_off = offsets[0].numpy()

    if action == "NO_ORDER":
        return {
            "order_type": "NO_ORDER",
            "confidence": round(confidence, 3),
            "ref_price": round(float(c100), 2)
        }

    # Reconstruct absolute prices
    entry_price = round(float(c100 + entry_off), 2)
    sl_price = round(float(c100 + sl_off), 2)
    tp_price = round(float(c100 + tp_off), 2)

    risk_pts = round(abs(entry_price - sl_price), 2)
    reward_pts = round(abs(tp_price - entry_price), 2)
    rr = round(reward_pts / (risk_pts + 1e-4), 2)

    # Safety Guardrail: Verify limit direction & minimum R:R
    is_valid = True
    reject_reason = None

    if action == "BUY_LIMIT":
        if entry_price >= c100:
            is_valid = False
            reject_reason = "BUY_LIMIT entry must be below current market price"
        elif sl_price >= entry_price:
            is_valid = False
            reject_reason = "Stop Loss must be below Entry"
        elif tp_price <= entry_price:
            is_valid = False
            reject_reason = "Take Profit must be above Entry"
    elif action == "SELL_LIMIT":
        if entry_price <= c100:
            is_valid = False
            reject_reason = "SELL_LIMIT entry must be above current market price"
        elif sl_price <= entry_price:
            is_valid = False
            reject_reason = "Stop Loss must be above Entry"
        elif tp_price >= entry_price:
            is_valid = False
            reject_reason = "Take Profit must be below Entry"

    if not is_valid:
        return {
            "order_type": "NO_ORDER",
            "confidence": round(confidence, 3),
            "filtered_out": True,
            "filter_reason": reject_reason,
            "ref_price": round(float(c100), 2)
        }

    return {
        "order_type": action,
        "entry": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "confidence": round(confidence, 3),
        "risk_points": risk_pts,
        "reward_points": reward_pts,
        "risk_reward": f"1:{rr}",
        "ref_price": round(float(c100), 2)
    }

if __name__ == "__main__":
    # Test on live MT5 terminal data
    print("Connecting to MT5 to fetch latest 100 live M15 candles...")
    if mt5.initialize(path=r"E:\ProgramFiles\MT5\terminal64.exe"):
        mt5.symbol_select("XAUUSD", True)
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 100)
        mt5.shutdown()
        
        if rates is not None and len(rates) == 100:
            raw_candles = [
                {"open": float(r['open']), "high": float(r['high']), "low": float(r['low']), "close": float(r['close'])}
                for r in rates
            ]
            result = predict_pending_order_from_100_candles(raw_candles)
            print("\n=== Model Prediction for Current MT5 Market State ===")
            print(json.dumps(result, indent=2))
        else:
            print("Failed to pull 100 rates from MT5")
    else:
        print("Could not initialize MT5")
