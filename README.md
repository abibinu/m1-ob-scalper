# M1 Order Block Scalper

An advanced, event-driven algorithmic trading system designed to scalp 1-Minute (M1) Order Blocks on Forex pairs like EURUSD and GBPUSD via MetaTrader 5 (MT5).

This system incorporates institutional concepts (Displacement, Order Blocks, Fair Value Gaps) with strict execution quality filters (Spread, Latency, Trend Confluence) and features a built-in Bayesian Optimizer to dynamically tune parameters for evolving market conditions.

---

## Key Features

### 1. Institutional Strategy Engine
- **Order Block Detection:** Automatically detects bullish/bearish displacement based on volume-weighted average true range calculations.
- **Rejection Confirmation Entries:** Avoids M1 "liquidity traps" by waiting for the institutional algos to finish sweeping the order block, only entering on a confirmed close back inside/above the zone.
- **Fair Value Gap (FVG) Confluence:** Scans for 3-bar M1 imbalances. Order Blocks co-located with FVGs receive a higher `volume_score` and `quality_score`.
- **Higher Timeframe Trend Filter:** Calculates a rolling 200 EMA. Filters out counter-trend Order Blocks at creation time, ensuring trades are only taken in the direction of the macro trend.

### 2. Advanced Exit Management
- **R-Multiple Take Profit:** Sets precise Take Profit targets based on the initial risk (e.g., 3.1 R).
- **Realistic Spread Buffers:** Calculates stop loss placement based on the true OB boundaries plus a wide spread multiplier (e.g., 15x avg spread) to survive natural M1 noise.
- **Partial Take Profit:** Closes a fraction of the position at an intermediate R-multiple to secure early profit, while leaving the rest to run.
- **Trailing Stops:** Locks in profits automatically once a trade moves in your favor.
- **Breakeven Logic:** Moves Stop Loss to the entry price once a certain R threshold is hit.
- **Time-Based Exits:** Closes trades automatically at market price if they don't reach their target within `MAX_HOLD_BARS`.

### 3. Execution Quality Pipeline
- **Session Window Filter:** Restricts trading to high-volume hours (e.g., NY / London Overlap) using strict UTC time limits.
- **Session Volume Strength:** Ensures the current hourly session meets a minimum volume threshold before allowing entries.
- **Spread Guard:** Dynamically rejects trades if the current live spread is too high compared to the 20-bar average.

### 4. Lightning Fast Backtesting & Optimization
- **CSV Caching:** Downloads MT5 ticks once and caches them locally for instantaneous repeated backtests.
- **Optuna AI Optimization:** Uses Bayesian search (`run_optimization.py`) to discover the most profitable parameter combinations (targeting metrics like the Sortino Ratio).
- **Walk-Forward Validation:** Validates results on Out-Of-Sample data to prevent curve fitting.

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.9+** (Tested on Python 3.10+)
- **MetaTrader 5** terminal installed and logged into a demo/live account (Windows only).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
The entire system is controlled via a `.env` file. A sample `.env` configuration:

```ini
# MT5 Credentials
MT5_LOGIN=YOUR_LOGIN_ID
MT5_PASSWORD=YOUR_PASSWORD
MT5_SERVER=Your-Broker-Server

# Strategy Parameters
SYMBOLS=EURUSD
MAX_OB_AGE_BARS=80
MAX_OB_PER_SYMBOL=5
OB_STACK_TOLERANCE=1.5
DISPLACEMENT_THRESHOLD=0.8

# Trend Filter
TREND_FILTER_ENABLED=true
TREND_EMA_PERIOD=200

# FVG Quality
FVG_QUALITY_THRESHOLD=0.05

# Exits & Risk
R_MULTIPLE_TP=3.1
SL_SPREAD_BUFFER=15.0
BREAKEVEN_AT_R=0.2
MAX_HOLD_BARS=115
PARTIAL_TP_ENABLED=false
```

---

## How to Run the System

### 1. Backtesting
Test the strategy on historical data.

```bash
# Run a standard backtest (Downloads data from MT5)
python run_backtest.py --symbol EURUSD --bars 40000

# Run a backtest using local Cache for faster speeds
python run_backtest.py --cache --symbol EURUSD --bars 40000

# Print every individual trade
python run_backtest.py --cache --trades

# Run Walk-Forward Validation (70% In-Sample / 30% Out-Of-Sample)
python run_backtest.py --cache --walk-forward
```

### 2. Optimization (Optuna)
Have the AI search for the most profitable parameters. The optimizer tests thousands of combinations and outputs the best values to put into your `.env`.

```bash
# Run 100 trials across 20,000 bars
python run_optimization.py --symbol EURUSD --bars 20000 --trials 100
```

*The optimizer will automatically output the best parameter block at the end of the run.*

### 3. Live Execution (Coming Soon)
*(The core logic, processors, and execution quality gates are fully built, optimized, and proven robust. The `run_live.py` MT5 event-loop script is scheduled for future deployment.)*

---

## Project Structure

```text
m1-ob-scalper/
├── run_backtest.py            # CLI entry point for Backtesting
├── run_optimization.py        # CLI entry point for AI Optimization
├── requirements.txt           # Python dependencies
├── .env                       # Master Configuration File
├── data/                      # Historical CSV Cache (Auto-generated)
└── src/
    ├── backtest/
    │   └── engine.py          # Event-driven Bar-by-Bar Replay Engine
    ├── core/
    │   ├── config.py          # Environment Variable Loader
    │   └── logger.py          # System Logging
    ├── data/
    │   └── market_data.py     # MT5 Interface & ATR Math
    ├── execution/
    │   ├── exit_manager.py    # TP, SL, Trailing Stop, Time-Exits, Partial TP
    │   └── order_executor.py  # Spread & Session Guards
    ├── strategy/
    │   ├── order_block.py     # Displacement Scanner, FVG Scanner, & OB Register
    │   └── signal.py          # Trade Signal Dataclasses
    └── validation/
        └── walk_forward.py    # IS/OOS Validation Engine
```
