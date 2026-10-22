"""
Inference Engine for Direction-Aware & Reinforcement M1 Sniper (V3)
Features:
1. micro_candles_180: Exactly 180 M1 OHLC candles.
2. macro_levels: Dict containing:
   - w2_high, w2_low
   - w1_high, w1_low
   - d1_high, d1_low
   - h4_high, h4_low
   - m15_high, m15_low
   - h4_slope (Macro trend direction metric)
   - time_of_day (UTC normalized 0.0 to 1.0)
Outputs pending order dictionary with Action, Entry, SL, TP, Confidence, and RR.
Includes full $0.35 spread mechanics and directional hard filtering.
"""
import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = r"W:\data\trading\best_m1_v3_rl_directional_net.pt"

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
        return x * self.fc(x).unsqueeze(-1)

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
        return self.relu(self.se(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))) + x)

class M1RLDirectionalNet(nn.Module):
    def __init__(self, in_channels_micro=7, in_features_macro=12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels_micro, 64, kernel_size=3, padding=1),
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
        self.micro_fc = nn.Sequential(nn.Linear(128, 64), nn.GELU())

        self.macro_fc = nn.Sequential(
            nn.Linear(in_features_macro, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.action_head = nn.Linear(64, 3)
        self.offset_head = nn.Linear(64, 3)

    def forward(self, micro, macro):
        h_m = self.trans1(self.block2(self.block1(self.stem(micro))))
        feat_micro = self.micro_fc(self.pool(self.block4(self.block3(h_m))).squeeze(-1))
        feat_macro = self.macro_fc(macro)
        feat = self.fusion_fc(torch.cat([feat_micro, feat_macro], dim=-1))
        return self.action_head(feat), self.offset_head(feat)

_GLOBAL_V3_MODEL = None

def get_m1_v3_model():
    global _GLOBAL_V3_MODEL
    if _GLOBAL_V3_MODEL is None:
        model = M1RLDirectionalNet()
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _GLOBAL_V3_MODEL = model
    return _GLOBAL_V3_MODEL

def predict_m1_v3_directional_order(micro_candles_180, macro_levels):
    """
    Predict pending order using Direction-Aware M1 V3 Model.
    """
    if len(micro_candles_180) < 180:
        return {"order_type": "NO_ORDER", "reason": "Insufficient candles"}

    candles = micro_candles_180[-180:]
    opens = np.array([float(c['open']) for c in candles], dtype=np.float32)
    highs = np.array([float(c['high']) for c in candles], dtype=np.float32)
    lows = np.array([float(c['low']) for c in candles], dtype=np.float32)
    closes = np.array([float(c['close']) for c in candles], dtype=np.float32)

    c180 = closes[-1]
    norm_o = opens - c180
    norm_h = highs - c180
    norm_l = lows - c180
    norm_c = closes - c180
    body = closes - opens
    rng = highs - lows
    color = np.where(closes > opens, 1.0, np.where(closes < opens, -1.0, 0.0)).astype(np.float32)

    raw_micro = np.stack([norm_o, norm_h, norm_l, norm_c, body, rng, color], axis=0) # (7, 180)
    std = raw_micro.std(axis=1, keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)
    micro_norm = (raw_micro - raw_micro.mean(axis=1, keepdims=True)) / std
    t_micro = torch.tensor(micro_norm, dtype=torch.float32).unsqueeze(0)

    # Macro vector (12 features)
    h4_slope = float(macro_levels.get('h4_slope', 0.0))
    macro_vec = np.array([
        macro_levels['w2_high'] - c180,
        macro_levels['w2_low'] - c180,
        macro_levels['w1_high'] - c180,
        macro_levels['w1_low'] - c180,
        macro_levels['d1_high'] - c180,
        macro_levels['d1_low'] - c180,
        macro_levels['h4_high'] - c180,
        macro_levels['h4_low'] - c180,
        macro_levels['m15_high'] - c180,
        macro_levels['m15_low'] - c180,
        h4_slope,
        macro_levels.get('time_of_day', 0.5)
    ], dtype=np.float32)
    t_macro = torch.tensor(macro_vec, dtype=torch.float32).unsqueeze(0)

    model = get_m1_v3_model()
    with torch.no_grad():
        logits, offsets = model(t_micro, t_macro)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_off = offsets.squeeze(0).numpy()

    action_idx = int(np.argmax(probs))
    conf = float(probs[action_idx])
    actions = ["NO_ORDER", "BUY_LIMIT", "SELL_LIMIT"]
    order_type = actions[action_idx]

    # Hard Direction Gate: Reject counter-trend signals against steep macro slopes
    if h4_slope < -0.5 and order_type == "BUY_LIMIT":
        return {"order_type": "NO_ORDER", "confidence": round(conf, 3), "reason": f"Downtrend slope {h4_slope:.2f} blocks BUY"}
    if h4_slope > 0.5 and order_type == "SELL_LIMIT":
        return {"order_type": "NO_ORDER", "confidence": round(conf, 3), "reason": f"Uptrend slope {h4_slope:.2f} blocks SELL"}

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

    # Sanity checks ($1.20 - $3.50 risk points to absorb gold noise and $0.35 spread)
    if risk_pts < 1.20 or risk_pts > 4.0 or rr < 2.0:
        return {"order_type": "NO_ORDER", "confidence": round(conf, 3), "reason": f"Invalid geometry (Risk: {risk_pts}, RR: {rr})"}

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
        "rr": f"1:{rr}",
        "h4_slope": round(h4_slope, 2)
    }
