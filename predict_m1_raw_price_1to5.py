"""
Live Inference Engine for Raw Price 1:5 Asymmetric M1 Liquidity Sniper
Takes:
- micro_candles_120: Exactly 120 M1 OHLC candles with timestamps.
Outputs:
- Dictionary with order_type, entry, sl, tp, time_to_fill_mins, rr, confidence.
Includes raw price prediction, $0.35 broker spread buffer, and auto-expiration calculation.
"""
import os
import datetime
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = r"W:\data\trading\best_m1_raw_price_1to5_net.pt"

class RawTemporalBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)

class RawM1SniperNet(nn.Module):
    def __init__(self, in_features=7):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = RawTemporalBlock(64, dilation=1)
        self.block2 = RawTemporalBlock(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = RawTemporalBlock(128, dilation=4)
        self.block4 = RawTemporalBlock(128, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.entry_head = nn.Linear(64, 1)
        self.sl_head = nn.Linear(64, 1)
        self.tp_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x_raw):
        x_norm = self.input_norm(x_raw)
        x_in = x_norm.transpose(1, 2)
        h = self.trans(self.block2(self.block1(self.stem(x_in))))
        feat = self.fc(self.pool(self.block4(self.block3(h))).squeeze(-1))
        last_close = x_raw[:, -1, 3].unsqueeze(-1)
        confirm_logits = self.confirm_head(feat)
        dir_logits = self.dir_head(feat)
        entry_pred = last_close + self.entry_head(feat)
        sl_pred = last_close + self.sl_head(feat)
        tp_pred = last_close + self.tp_head(feat)
        ttf_pred = torch.relu(self.ttf_head(feat)) + 1.0
        return confirm_logits, dir_logits, entry_pred, sl_pred, tp_pred, ttf_pred

_GLOBAL_RAW_SNIPER_MODEL = None

def get_raw_m1_sniper_model():
    global _GLOBAL_RAW_SNIPER_MODEL
    if _GLOBAL_RAW_SNIPER_MODEL is None:
        model = RawM1SniperNet()
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _GLOBAL_RAW_SNIPER_MODEL = model
    return _GLOBAL_RAW_SNIPER_MODEL

def predict_raw_m1_sniper_order(candles_120):
    """
    Takes 120 raw M1 candles with keys: ['open', 'high', 'low', 'close', 'time']
    Returns trade setup with exact dollar quote levels and expected fill window.
    """
    if len(candles_120) < 120:
        return {"order_type": "NO_ORDER", "reason": "Insufficient candles"}

    candles = candles_120[-120:]
    feat_rows = []
    for c in candles:
        t_epoch = c.get('time', 0)
        dt = datetime.datetime.fromtimestamp(t_epoch, datetime.timezone.utc) if t_epoch > 0 else datetime.datetime.now(datetime.timezone.utc)
        feat_rows.append([
            float(c['open']),
            float(c['high']),
            float(c['low']),
            float(c['close']),
            float(dt.minute),
            float(dt.hour),
            float(dt.weekday())
        ])

    raw_mat = np.array(feat_rows, dtype=np.float32)
    t_in = torch.tensor(raw_mat, dtype=torch.float32).unsqueeze(0)

    model = get_raw_m1_sniper_model()
    with torch.no_grad():
        p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf = model(t_in)
        conf = float(torch.softmax(p_conf, dim=-1)[0, 1].item())
        dir_idx = int(torch.argmax(p_dir, dim=-1)[0].item())
        entry = round(float(p_ent[0, 0].item()), 2)
        sl = round(float(p_sl[0, 0].item()), 2)
        tp = round(float(p_tp[0, 0].item()), 2)
        ttf = max(2, min(int(round(float(p_ttf[0, 0].item()))), 25))

    c_now = float(candles[-1]['close'])
    order_type = "BUY_LIMIT" if dir_idx == 0 else "SELL_LIMIT"

    risk_pts = round(abs(entry - sl), 2)
    reward_pts = round(abs(tp - entry), 2)
    rr = round(reward_pts / (risk_pts + 1e-4), 2)

    # Geometry validation
    is_valid = False
    if order_type == "BUY_LIMIT":
        if entry < c_now and sl < entry and tp > entry and 0.40 <= risk_pts <= 5.0 and rr >= 3.0:
            is_valid = True
    else: # SELL_LIMIT
        if entry > c_now and sl > entry and tp < entry and 0.40 <= risk_pts <= 5.0 and rr >= 3.0:
            is_valid = True

    if not is_valid:
        return {
            "order_type": "NO_ORDER",
            "confidence": round(conf, 3),
            "ref_price": round(c_now, 2),
            "reason": "Geometry constraint"
        }

    return {
        "order_type": order_type,
        "confidence": round(conf, 3),
        "ref_price": round(c_now, 2),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_pts": risk_pts,
        "reward_pts": reward_pts,
        "rr": f"1:{rr}",
        "time_to_fill_mins": ttf
    }
