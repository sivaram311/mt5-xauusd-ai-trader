"""
Live Inference Engine for M5 High-Win-Rate Liquidity Engine (1:2.6 R:R)
Model Checkpoint: W:\data\trading\best_m5_hwr_net.pt
Inputs:
- candles_48: Exactly 48 M5 OHLC candles with timestamps.
- macro_levels: Dict containing PDH, PDL, PWH, PWL, H1 Trend Slope, recent 1h high/low.
Outputs:
- Dictionary with order_type, entry, sl, tp, risk_pts, reward_pts, rr, confidence, ttf_minutes.
"""
import os
import datetime
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = r"W:\data\trading\best_m5_hwr_net.pt"

class TemporalConvBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + x)

class M5HWRNet(nn.Module):
    def __init__(self, micro_channels=7, macro_dim=10):
        super().__init__()
        self.micro_norm = nn.LayerNorm(micro_channels)
        self.micro_stem = nn.Sequential(
            nn.Conv1d(micro_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.block1 = TemporalConvBlock(64, dilation=1)
        self.block2 = TemporalConvBlock(64, dilation=2)
        self.trans = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.block3 = TemporalConvBlock(128, dilation=4)
        self.micro_pool = nn.AdaptiveAvgPool1d(1)
        self.micro_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU()
        )
        self.macro_norm = nn.BatchNorm1d(macro_dim)
        self.macro_fc = nn.Sequential(
            nn.Linear(macro_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.confirm_head = nn.Linear(64, 2)
        self.dir_head = nn.Linear(64, 2)
        self.entry_head = nn.Linear(64, 1)
        self.sl_head = nn.Linear(64, 1)
        self.tp_head = nn.Linear(64, 1)
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x_micro, x_macro):
        x_m_norm = self.micro_norm(x_micro)
        x_m_in = x_m_norm.transpose(1, 2)
        h_m = self.trans(self.block2(self.block1(self.micro_stem(x_m_in))))
        feat_micro = self.micro_fc(self.micro_pool(self.block3(h_m)).squeeze(-1))
        feat_macro = self.macro_fc(self.macro_norm(x_macro))
        feat = self.fusion(torch.cat([feat_micro, feat_macro], dim=-1))
        last_close = x_micro[:, -1, 3].unsqueeze(-1)
        p_conf = self.confirm_head(feat)
        p_dir = self.dir_head(feat)
        p_ent = last_close + self.entry_head(feat)
        p_sl = last_close + self.sl_head(feat)
        p_tp = last_close + self.tp_head(feat)
        p_ttf = torch.relu(self.ttf_head(feat)) + 1.0
        return p_conf, p_dir, p_ent, p_sl, p_tp, p_ttf

_GLOBAL_M5_HWR_MODEL = None

def get_m5_hwr_model():
    global _GLOBAL_M5_HWR_MODEL
    if _GLOBAL_M5_HWR_MODEL is None:
        model = M5HWRNet()
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _GLOBAL_M5_HWR_MODEL = model
    return _GLOBAL_M5_HWR_MODEL

def predict_m5_hwr_order(candles_48, macro_context):
    """
    Evaluates 48 M5 candles and macro levels to produce an asymmetric M5 limit order.
    """
    if len(candles_48) < 48:
        return {"order_type": "NO_ORDER", "reason": "Insufficient M5 candles"}

    candles = candles_48[-48:]
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
    t_micro = torch.tensor(raw_mat, dtype=torch.float32).unsqueeze(0)

    c_now = float(candles[-1]['close'])
    recent_1h_high = max(float(c['high']) for c in candles[-12:])
    recent_1h_low = min(float(c['low']) for c in candles[-12:])

    macro_vec = np.array([
        float(macro_context.get('pdh', c_now + 10.0)) - c_now,
        c_now - float(macro_context.get('pdl', c_now - 10.0)),
        float(macro_context.get('pwh', c_now + 25.0)) - c_now,
        c_now - float(macro_context.get('pwl', c_now - 25.0)),
        float(macro_context.get('h1_slope', 0.0)),
        recent_1h_high - c_now,
        c_now - recent_1h_low,
        float(feat_rows[-1][5]), # Hour UTC
        float(feat_rows[-1][6]), # Day of Week
        float(macro_context.get('spread', 0.35))
    ], dtype=np.float32)
    t_macro = torch.tensor(macro_vec, dtype=torch.float32).unsqueeze(0)

    model = get_m5_hwr_model()
    with torch.no_grad():
        pc, pd, pe, ps, pt, ptt = model(t_micro, t_macro)
        conf = float(torch.softmax(pc, dim=-1)[0, 1].item())
        dir_idx = int(torch.argmax(pd, dim=-1)[0].item())
        entry = round(float(pe[0, 0].item()), 2)
        sl = round(float(ps[0, 0].item()), 2)
        tp = round(float(pt[0, 0].item()), 2)
        ttf_bars = max(1, min(int(round(float(ptt[0, 0].item()))), 6))
        ttf_minutes = ttf_bars * 5

    order_type = "BUY_LIMIT" if dir_idx == 0 else "SELL_LIMIT"
    risk_pts = round(abs(entry - sl), 2)
    reward_pts = round(abs(tp - entry), 2)
    rr = round(reward_pts / (risk_pts + 1e-4), 2)

    # Geometry validation (Survives gold noise wicks)
    is_valid = False
    if order_type == "BUY_LIMIT":
        if entry < c_now and sl < entry and tp > entry and 1.50 <= risk_pts <= 5.0 and reward_pts >= 4.0:
            is_valid = True
    else: # SELL_LIMIT
        if entry > c_now and sl > entry and tp < entry and 1.50 <= risk_pts <= 5.0 and reward_pts >= 4.0:
            is_valid = True

    if not is_valid:
        return {
            "order_type": "NO_ORDER",
            "confidence": round(conf, 3),
            "ref_price": round(c_now, 2),
            "reason": "Geometry invalid"
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
        "time_to_fill_mins": ttf_minutes
    }
