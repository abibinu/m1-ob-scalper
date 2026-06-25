# M1 Order Block Scalper

An advanced, event-driven algorithmic trading system designed to scalp 1-Minute (M1) Order Blocks on Forex pairs like EURUSD and GBPUSD via MetaTrader 5 (MT5).

This system incorporates institutional concepts (Displacement, Order Blocks) with strict execution quality filters (Spread, Latency, News Blackouts, Session Overlaps) and features a built-in Bayesian Optimizer to dynamically tune parameters for evolving market conditions.

---

## Key Features

1. **Order Block Strategy Engine**
   - Automatically detects bullish/bearish displacement based on dynamic average true range calculations.
   - Registers, tracks, and merges overlapping Order Blocks (`OB_STACK_TOLERANCE`).
   - Automatically expires stale Order Blocks after a configurable number of bars.

2. **Advanced Exit Management**
   - **R-Multiple Take Profit:** Sets precise Take Profit targets based on the initial risk (e.g., 3.0 R).
   - **Trailing Stops:** Locks in profits automatically once a trade moves in your favor.
   - **Breakeven Logic:** Moves Stop Loss to the entry price once a certain R threshold is hit.
   - **Time-Based Exits:** Closes trades automatically if they don't reach their target within `MAX_HOLD_BARS`.

3. **Execution Quality Pipeline**
   - **Session Window Filter:** Restricts trading to high-volume hours (e.g., NY / London Overlap) using strict UTC time limits.
   - **Spread Guard:** Dynamically rejects trades if the current spread is too high compared to the 20-bar average.

4. **Lightning Fast Backtesting & Optimization**
   - **CSV Caching:** Downloads MT5 ticks once and caches them locally for instantaneous repeated backtests.
   - **Optuna AI Optimization:** Uses Bayesian search (`run_optimization.py`) to discover the most profitable parameter combinations.
   - **Walk-Forward Validation:** Validates results on Out-Of-Sample data to prevent curve fitting.

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.9+** (Tested on Python 3.10+)
- **MetaTrader 5** terminal installed and logged into a demo/live account.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
The entire system is controlled via a `.env` file. A sample `.env` file:

```ini
# MT5 Credentials
MT5_LOGIN=YOUR_LOGIN_ID
MT5_PASSWORD=YOUR_PASSWORD
MT5_SERVER=Your-Broker-Server

# Strategy Parameters
SYMBOLS=EURUSD,GBPUSD
MAX_OB_AGE_BARS=145
DISPLACEMENT_THRESHOLD=1.3

# Exits & Risk
R_MULTIPLE_TP=3.0
BREAKEVEN_AT_R=0.7
MAX_HOLD_BARS=65
SL_SPREAD_BUFFER=1.7
TRAILING_STOP_ACTIVATION_R=0.0
TRAILING_STOP_DISTANCE_PIPS=0.0

# Session Window (UTC Time) - Example: NY / London Overlap
SESSION_START_UTC=13:00
SESSION_END_UTC=16:30
```

> **Note on Timezones:** `SESSION_START_UTC` and `SESSION_END_UTC` strictly evaluate against Universal Coordinated Time (UTC). You must convert your local time to UTC when configuring the session window.

---

## How to Run the System

### 1. Backtesting
Test the strategy on historical data.

```bash
# Run a standard backtest (Downloads data from MT5)
python run_backtest.py --symbol EURUSD --bars 5000

# Run a backtest using local Cache for faster speeds
python run_backtest.py --cache

# Print every individual trade
python run_backtest.py --cache --trades

# Run Walk-Forward Validation (70% In-Sample / 30% Out-Of-Sample)
python run_backtest.py --cache --walk-forward
```

### 2. Optimization (Optuna)
Have the AI search for the most profitable parameters. The optimizer tests thousands of combinations and outputs the best values to put into your `.env`.

```bash
# Run 50 trials across 20,000 bars
python run_optimization.py --symbol EURUSD --bars 20000 --trials 50
```

*The optimizer will automatically output the best parameter block at the end of the run.*

### 3. Live Execution (Coming Soon)
*(The core logic, processors, and execution quality gates are fully built, but the `run_live.py` MT5 event-loop script is currently in development.)*

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
    │   ├── exit_manager.py    # TP, SL, Trailing Stop, Time-Exits
    │   └── order_executor.py  # Spread & Session Guards
    ├── strategy/
    │   ├── order_block.py     # Displacement Scanner & OB Register
    │   └── signal.py          # Trade Signal Dataclasses
    └── validation/
        └── walk_forward.py    # IS/OOS Validation Engine
```
