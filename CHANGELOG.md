# CHANGELOG - MT5 XAUUSD AI Trader

All notable changes, architectural iterations, model versions, and live execution modifications are documented here.

---

## [v2.1.0] - 2026-10-21: Multi-Engine Scalper & Execution Overhaul

### 1. Dedicated M5 Continuous Scalper Model
- **Dataset**: `xauusd_m5_scalp_dataset.npz` (98,923 samples).
  - Modeled across **all 24 hours** continuously (May 2025 – October 2026) without session restriction.
  - Hard-coded **$0.35/oz spread penalty** subtracted from all profit targets during labeling.
  - Lookback: 60 bars (5 hours). Max trade horizon: 18 bars (90 minutes).
- **Model Checkpoint**: `best_m5_scalper_net.pt` (180,166 parameters).
  - Architecture: 1D Dilated ResNet + Squeeze-and-Excitation (SE) channel attention.
  - Trained on 4 vCPUs in 22.06 minutes with Cosine Annealing Learning Rate.
  - Out-of-sample test accuracy: **50.51%**, Trade F1: **56.38%**, Trade Precision: **51.80%**.
- **Out-of-Sample Forward Backtest** (`m5_scalper_backtest_report.json`):
  - Evaluated over 19,767 unseen bars (July 13 – October 21, 2026).
  - Balanced Tier (Conf $\ge 0.50$): **43.40% Win Rate**, **1.48 Profit Factor**, **26.22% Max Drawdown**.
  - Ultra-Selective Tier (Conf $\ge 0.60$): **48.18% Win Rate**, **1.81 Profit Factor**, **34.56% Max Drawdown**.
- **Inference Engine**: Created `predict_m5_scalp.py` with standalone tensor normalization and sanity checks.

### 2. Dual-Engine Live Daemon Deployment
- Integrated into `mt5_live_trader.py` running concurrently under Windows Task Scheduler (`MT5-AI-LiveTrader-Session0`).
- **Engine 1 (M15 Intraday Trend)**:
  - Magic Number: `778899`
  - Order Comment Tag: `M15_CONF_XX`
  - Evaluated on M15 candle close (every 15 min).
  - Risk per trade: 1.0%. Expiration: 12 hours.
- **Engine 2 (M5 Scalper)**:
  - Magic Number: `778855`
  - Order Comment Tag: `M5_SCALP_XX`
  - Evaluated on M5 candle close (every 5 min).
  - Risk per trade: 0.5%. Expiration: 90 minutes.

### 3. Broker UTC+3 Expiration & IPC Resilience
- **Root Cause**: OctaFX broker ticks operate at UTC+3 (10,800-second offset relative to system UTC).
- **Fix**: Implemented `get_broker_time()` using live tick epoch times for pending order expiration requests. Eliminates MT5 `10022 (Invalid expiration)` error.
- **Auto-Reconnection**: Added `ensure_connection()` heartbeat in main loop to reconnect seamlessly if broker connection drops.

### 4. Removal of Single Pending Order Restriction
- **Previous Behavior**: Script canceled any existing pending order before placing a new one, or when `NO_ORDER` was emitted.
- **New Behavior**: 
  - Allows stacking multiple concurrent pending orders across candles.
  - `NO_ORDER` no longer deletes live pending orders.
  - Added `has_duplicate_pending_order()` with a **$0.25 tolerance window** to prevent placing redundant duplicate orders at the exact same penny/price.

### 5. Removal of Trailing Stop / Breakeven
- **Previous Behavior**: `manage_trailing_breakeven()` dynamically shifted Stop Loss to `Entry + $0.15` once a position reached `+1.5R` profit.
- **New Behavior**:
  - Removed trailing breakeven completely.
  - Orders execute with **pure static Stop Loss and Take Profit** targets as predicted by the AI at entry time.
  - Trades run freely to full target or full stop loss without modification.

---

## [v2.0.0] - 2026-10-18: Multi-Timeframe Dual-Branch & Liquidity Anchors
- Added M15 + H4 dual-branch architecture (`train_dual_branch.py`).
- Added Liquidity Anchor labeling engine with Focal Loss (`build_dataset_v3.py`, `train_v3.py`).
- Added formal trading profiles matrix (`PROFILES.md`).

---

## [v1.0.0] - 2026-10-11: Initial M15 1D ResNet Engine
- 99,000 M15 bar ingestion.
- 1D Temporal Dilated ResNet (`best_order_net.pt`).
- Baseline pending order live trader daemon deployed to Session 0.
