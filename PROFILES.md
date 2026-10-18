# Trading Profiles Registry & Version Matrix (MT5 XAUUSD M15)

This registry tracks all operational trading profiles, model architectures, frequency tiers, and out-of-sample verified metrics for the XAUUSD M15 automated trading engine.

---

## 1. Master Profile Comparison Matrix

| Profile ID | Profile Name | Engine Version | Target Frequency | Entry Separation Method | Target R:R | Win Rate | Profit Factor | Max Drawdown | 10-Mo Final (\$10k start) | Primary Use Case |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P-V1-ACTIVE** | Active Day Trader | **V1.0** (`best_order_net.pt`) | **8 to 9 / day** | Static Conf $\ge 0.50$ (London/NY) | $1 : 4.37$ | 25.6% | 1.43 | 43.5% | \$5,372,255 | High compounding volume |
| **P-V1-BALANCED** | Balanced Intraday | **V1.0** (`best_order_net.pt`) | **3 to 4 / day** | Static Conf $\ge 0.60$ (London/NY) | $1 : 4.63$ | 23.0% | 1.36 | 41.9% | \$84,803 | Moderate frequency intraday |
| **P-V1-SWING** | High-Precision Swing | **V1.0** (`best_order_net.pt`) | **3 to 4 / week** | Conf $\ge 0.65$ + Margin $\ge 0.20$ | $1 : 4.63$ | **36.2%** | **2.64** | **12.8%** | \$49,699 | High win-rate, low drawdown |
| **P-V2-MACRO** | Dual-Branch Multi-TF | **V2.0** (`best_order_net_v2.pt`) | **7 to 8 / day** | M15 Micro (100) + H4 Macro (30) | $1 : 4.92$ | 20.5% | 1.15 | 77.5% | \$101,420 | Higher-timeframe macro bias |
| **P-V3-TOP1** | Ultra-Selective Daily | **V3.0** (`best_order_net_v3.pt`) | **1 / day** | Liquidity Anchor + Top-1 Ranker | $1 : 3.00$ | **31.5%** | **1.36** | **12.6%** | \$15,345 | **Safest daily execution** |
| **P-V3-TOP2** | Dual Setup Daily | **V3.0** (`best_order_net_v3.pt`) | **2 / day** | Liquidity Anchor + Top-2 Ranker | $1 : 3.00$ | **29.5%** | **1.24** | **24.1%** | \$17,996 | Balanced 2 trades/day |
| **P-V3-TOP3** | Triple Setup Daily | **V3.0** (`best_order_net_v3.pt`) | **3 / day** | Liquidity Anchor + Top-3 Ranker | $1 : 3.00$ | **28.7%** | **1.19** | **21.9%** | \$20,477 | Standard intraday daily rank |
| **P-V3-TOP5** | High-Turnover Daily | **V3.0** (`best_order_net_v3.pt`) | **4 to 5 / day** | Liquidity Anchor + Top-5 Ranker | $1 : 3.00$ | **28.6%** | **1.17** | **25.8%** | \$30,436 | Maximum daily action |

---

## 2. Profile Configurations & Deep Breakdown

### Profile `P-V3-TOP1` (Top 1 Daily Ranker) — Recommended for Safety
- **Core Strategy:** Evaluates the 44 M15 bars between 07:00 and 18:00 UTC. Picks strictly the single highest conviction liquidity sweep setup each day.
- **Stop Loss:** \$2.00 (200 points).
- **Take Profit:** \$6.00 (600 points).
- **Expected Monthly Volume:** ~16 to 18 trades/month.
- **Drawdown Risk:** **12.6%** (Lowest of all multi-trade setups).
- **Profit Factor:** **1.36**.

### Profile `P-V3-TOP3` (Top 3 Daily Ranker) — Recommended for 2–3 Trades/Day
- **Core Strategy:** Picks the 3 best-ranked structural liquidity anchors per day during London and New York.
- **Stop Loss:** \$2.00.
- **Take Profit:** \$6.00.
- **Expected Monthly Volume:** ~50 to 55 trades/month.
- **Drawdown Risk:** **21.9%**.
- **Net Gain:** **+104% over 10 months** (\$10,000 $\rightarrow$ \$20,477).

### Profile `P-V1-SWING` (High-Precision Swing V1) — Highest Expected Value
- **Core Strategy:** Pure 100-candle M15 1D ResNet with a strict 65% probability gate and 20% margin over the second choice.
- **Target R:R:** $1 : 4.63$ (\$2.00 SL vs \$9.26+ TP).
- **Win Rate:** **36.2%** (against an 18% breakeven barrier).
- **Profit Factor:** **2.64**.
- **Drawdown Risk:** **12.8%**.

---

## 3. Active Production Daemon Mapping

The production background daemon ([`mt5_live_trader.py`](mt5_live_trader.py)) uses the following profile selector:

```python
# CURRENT ACTIVE PROFILE: P-V1-LIVE (Conf >= 0.43 with Dynamic +1.5R Breakeven)
PROFILE_CONFIG = {
    "P-V1-LIVE": {
        "model_file": "best_order_net.pt",
        "confidence_threshold": 0.43,
        "session_filter": (7, 18),
        "breakeven_trigger_r": 1.5,
        "spread_limit": 0.35,
        "max_risk_pct": 1.0
    },
    "P-V3-TOP1": {
        "model_file": "best_order_net_v3.pt",
        "rank_top_k": 1,
        "session_filter": (7, 18),
        "target_rr": 3.0,
        "risk_pts": 2.0
    },
    "P-V3-TOP3": {
        "model_file": "best_order_net_v3.pt",
        "rank_top_k": 3,
        "session_filter": (7, 18),
        "target_rr": 3.0,
        "risk_pts": 2.0
    }
}
```

---

## 4. Live Forward Verified Results (October 2026)

- **Account:** Starting \$718.16 $\rightarrow$ Current **\$740.47 (+3.11% net gain)**
- **Completed Trades:** 5 trades (2 Wins @ full TP, 3 Losses @ tight SL)
- **Live Profit Factor:** **1.83**
- **Live Win Rate:** **40.0%**
