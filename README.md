# MT5 XAUUSD M15 AI Automated Pending Order Trader

An automated trading system designed for MetaTrader 5 on Gold (`XAUUSD`) using a 15-minute timeframe (`M15`). The model takes the last 100 historical candles (pure Time + Open, High, Low, Close) and predicts pending orders (`BUY_LIMIT` / `SELL_LIMIT`) with dynamic Entry, Stop Loss (SL), Take Profit (TP), and lot sizing.

---

## 1. System Architecture

```
MT5 Terminal (XAUUSD M15)
        │
        ▼ (Every 15 mins on candle close)
Extract 100 Historical Bars [Time, O, H, L, C]
        │
        ▼
Relative Normalization Engine (No ATR dependency; scaled to C100)
        │
        ▼
1D Temporal Dilated ResNet (167,558 parameters, PyTorch CPU)
        │
        ▼
Guardrail Filters:
- Confidence Gate: >= 0.43 (Configurable up to 0.65)
- Directional Validation: BUY_LIMIT below market, SELL_LIMIT above market
- Minimum Risk-Reward Ratio: >= 1:2.0
- Maximum Risk per Trade: 1.0% Account Equity
        │
        ▼
MT5 Execution Bridge (Session 0 Headless Daemon)
- Places/Updates Pending Orders with Magic Number 778899
- Cancels stale/obsolete orders if model shifts to NO_ORDER
```

---

## 2. Directory & File Overview

| File | Description |
| :--- | :--- |
| `best_m5_hwr_net.pt` | M5 High-Win-Rate Liquidity Engine (1:2.6 R:R, PF 2.36 at 48% WR). |
| `best_m1_v3_rl_directional_net.pt` | M1 V3 Direction-Aware & Reinforcement Sniper with H4 slope gating. |
| `best_order_net.pt` | Trained PyTorch weights (~695 KB) trained on 99,000 M15 bars (4+ years of XAUUSD history). |
| `predict_m5_hwr.py` | Inference engine for M5 HWR model. |
| `predict_m1_v3_directional.py` | Inference engine for M1 V3 Direction-Aware Sniper. |
| `predict_order.py` | Core M15 inference engine returning JSON execution signals. |
| `mt5_live_trader.py` | Production daemon running triple-engine execution (M15 + M5 + M1) on Session 0. |
| `build_m5_hwr_dataset.py` | Dataset generator for M5 High-Win-Rate asymmetric trades. |
| `train_m5_hwr.py` | Training script for M5 HWR network. |
| `backtest_m5_hwr.py` | Forward backtest engine for M5 HWR model. |
| `backtest_report.json` | Detailed historical trade statistics and equity curve data. |
| `mt5_live_trader.log` | Operational execution log containing real-time market evaluations. |

---

## 3. Verified Performance Metrics

### Out-of-Sample Forward Backtest (10 Months / 19,723 Unseen Candles)
- **Time Period:** December 2025 to October 2026
- **Average Realized R:R:** `1 : 4.22` to `1 : 4.63`
- **Win Rate:** `25.7%` to `36.2%` (vs breakeven threshold of `18.5%`)
- **Profit Factor:** `1.30` to `2.64`
- **Max Drawdown:** Under `13%` on high-confidence setups

### Release History
- **v1.1.0 (Oct 18, 2026):** Refreshed dataset to 99,000 M15 bars (up through Oct 16, 2026 23:45 UTC). Retrained V1 Dilated ResNet model reaching 44.26% test trade F1 score and 48.46% trade precision. Tracked all multi-version weights (V1, V2 Dual-Branch, V3 Liquidity Anchor Top-K).
- **v1.0.0 (Oct 11, 2026):** Initial production release of 1D Temporal ResNet order generation network.

### Live Forward Testing (Oct 13 – Oct 17, 2026)
- **Account:** Starting \$718.16 -> Closed at **\$740.47 (+3.11% net gain)**
- **Trades Taken:** 5 completed trades (2 Wins @ full TP, 3 Losses @ tight SL)
- **Realized Live R:R:** `1 : 2.75`
- **Profit Factor:** `1.83`

---

## 4. Setup & Running Instructions

### Prerequisites
- Windows 10/11 or Windows Server
- Python 3.10+ (PyTorch CPU, MetaTrader5, NumPy, Pandas)
- MetaTrader 5 Terminal installed at `E:\ProgramFiles\MT5\terminal64.exe` (or update `TERMINAL_PATH` in `mt5_live_trader.py`)

### Manual Start (Foreground)
```powershell
python W:\data\trading\mt5_live_trader.py
```

### Background Daemon (Session 0 Autostart via Windows Task Scheduler)
To ensure the trader runs continuously even if no user is logged in:
```powershell
$action = New-ScheduledTaskAction -Execute "C:\Python314\python.exe" -Argument "W:\data\trading\mt5_live_trader.py" -WorkingDirectory "W:\data\trading"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName "MT5-AI-LiveTrader-Session0" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force
Start-ScheduledTask -TaskName "MT5-AI-LiveTrader-Session0"
```

---

## 5. Risk & Money Management Rules

- **Magic Number:** `778899` (used to track AI orders separately from manual trades).
- **Risk per Trade:** Dynamic `1.0%` risk of current account balance:
  $$\text{Lot Size} = \frac{\text{Account Balance} \times 0.01}{\text{Risk in Points} \times 100}$$
- **Order Validity:** Maximum 12 hours (`48` bars) expiration if entry price is not triggered.
- **Safety Pre-cancel:** Automatically cancels previous AI pending orders if market structure evolves into `NO_ORDER`.
