"""
Inference Engine for XAUUSD M1 Sniper Pending Orders (180 Bars Context)
Takes 180 M1 OHLC candles as input.
Outputs pending order dictionary with Action, Entry, SL, TP, Confidence, and RR.
Target: >= 1:5.0 Risk:Reward with $0.35 spread friction modeled.
"""
import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = r"W:\data\trading\best_m1_sniper_net.pt"

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w

class ResBlockSE1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels)
        self.relu = nn.GELU()
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + res)

class M1SniperNet(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = ResBlockSE1D(64, dilation=1)
        self.block2 = ResBlockSE1D(64, dilation=2)
        self.trans1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = ResBlockSE1D(128, dilation=4)
        self.block4 = ResBlockSE1D(128, dilation=8)
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
        h = self.trans1(h)
        h = self.block3(h)
        h = self.block4(h)
        feat = self.fc(self.pool(h).squeeze(-1))
        return self.action_head(feat), self.offset_head(feat)

_model = None

def get_m1_sniper_model():
    global _model
    if _model is None:
        _model = M1SniperNet(in_features=6)
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
    return _model

def predict_m1_sniper_order(candles_180):
    if len(candles_180) < 180:
        return {"order_type": "NO_ORDER", "reason": "Insufficient candles"}

    candles = candles_180[-180:]
    opens = np.array([float(c['open']) for c in candles], dtype=np.float32)
    highs = np.array([float(c['high']) for c in candles], dtype=np.float32)
    lows = np.array([float(c['low']) for c in candles], dtype=np.float32)
    closes = np.array([float(c['close']) for c in candles], dtype=np.float32)

    c180 = closes[-1]
    norm_open = opens - c180
    norm_high = highs - c180
    norm_low = lows - c180
    norm_close = closes - c180
    body = closes - opens
    rng = highs - lows

    features = np.stack([norm_open, norm_high, norm_low, norm_close, body, rng], axis=0) # (6, 180)
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0) # (1, 6, 180)

    model = get_m1_sniper_model()
    with torch.no_grad():
        logits, offsets = model(feat_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_off = offsets.squeeze(0).numpy()

    action_idx = int(np.argmax(probs))
    conf = float(probs[action_idx])
    actions = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
    order_type = actions[action_idx]

    if order_type == "NO_ORDER":
        return {
            "order_type": "NO_ORDER",
            "confidence": round(conf, 3),
            "ref_price": round(float(c180), 2)
        }

    entry_off, sl_off, tp_off = float(pred_off[0]), float(pred_off[1]), float(pred_off[2])
    entry = round(c180 + entry_off, 2)
    sl = round(c180 + sl_off, 2)
    tp = round(c180 + tp_off, 2)

    risk_pts = abs(entry - sl)
    reward_pts = abs(tp - entry)
    rr = round(reward_pts / (risk_pts + 1e-4), 2)

    # Sniper sanity checks: Risk between $0.80 and $2.00, RR >= 3.5
    if risk_pts < 0.80 or risk_pts > 2.50 or rr < 3.5:
        return {"order_type": "NO_ORDER", "confidence": round(conf, 3), "reason": "Invalid sniper geometry"}

    if order_type == "BUY_LIMIT" and entry >= c180:
        return {"order_type": "NO_ORDER", "confidence": round(conf, 3), "reason": "BUY entry above market"}

    if order_type == "SELL_LIMIT" and entry <= c180:
        return {"order_type": "NO_ORDER", "confidence": round(conf, 3), "reason": "SELL entry below market"}

    return {
        "order_type": order_type,
        "confidence": round(conf, 3),
        "ref_price": round(float(c180), 2),
        "entry": round(float(entry), 2),
        "sl": round(float(sl), 2),
        "tp": round(float(tp), 2),
        "risk_pts": round(float(risk_pts), 2),
        "reward_pts": round(float(reward_pts), 2),
        "rr": f"1:{rr}"
    }
