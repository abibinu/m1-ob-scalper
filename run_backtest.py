"""
Backtest Runner
===============
Connects to your MT5 account, pulls historical M1 bars, runs the full
backtesting engine, and prints a detailed performance report.

Usage:
    python run_backtest.py [--symbol EURUSD] [--bars 5000] [--slippage 0.5]

Defaults use whatever is configured in .env
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

# ── Load .env before any project imports ──────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5

from src.backtest.engine import BacktestEngine, BacktestReport
from src.core.logger import get_logger
from src.validation.walk_forward import (
    WalkForwardValidator,
    check_kill_criteria,
    evaluate_gate,
)

log = get_logger("backtest_runner")


# ─────────────────────────────────────────────────────────────────────────────
# MT5 data fetch (standalone, no dependency on MT5Client singleton)
# ─────────────────────────────────────────────────────────────────────────────

def connect_mt5() -> bool:
    """Initialise MT5 using .env credentials."""
    login = int(os.environ["MT5_LOGIN"])
    password = os.environ["MT5_PASSWORD"]
    server = os.environ["MT5_SERVER"]

    if not mt5.initialize():
        print(f"[ERROR] MT5 initialize() failed: {mt5.last_error()}")
        return False

    if not mt5.login(login=login, password=password, server=server):
        print(f"[ERROR] MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    print(f"\n[OK] Connected to MT5")
    print(f"  Account  : {info.login}")
    print(f"  Server   : {info.server}")
    print(f"  Balance  : {info.balance:.2f} {info.currency}")
    print(f"  Equity   : {info.equity:.2f} {info.currency}")
    return True


def fetch_m1_bars(symbol: str, count: int, use_cache: bool = False) -> pd.DataFrame:
    """Pull ``count`` M1 bars from MT5 and return as a clean DataFrame. Uses local CSV cache if use_cache is True."""
    import os
    cache_dir = "data"
    cache_file = os.path.join(cache_dir, f"historical_bars_{symbol}_{count}.csv")

    if use_cache and os.path.exists(cache_file):
        print(f"  Loading  : {count:,} M1 bars from cache: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    raw = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
    if raw is None or len(raw) == 0:
        print(f"[ERROR] No data returned for {symbol}: {mt5.last_error()}")
        sys.exit(1)

    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Rename tick_volume if present
    if "tick_volume" not in df.columns and "real_volume" in df.columns:
        df["tick_volume"] = df["real_volume"]

    print(f"  Fetched  : {len(df):,} M1 bars for {symbol}")
    print(f"  From     : {df['time'].iloc[0].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  To       : {df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}")

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"  Saved    : {cache_file}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

def _bar(label: str, value: str, width: int = 50) -> str:
    return f"  {label:<30} {value}"


def print_report(report: BacktestReport, symbol: str, bars_count: int) -> None:
    """Pretty-print the backtest results."""
    s = report.summary()
    wins = len(report.wins)
    losses = len(report.losses)

    # Breakdown by exit reason
    from src.execution.exit_manager import ExitReason
    by_reason = {}
    for t in report.trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS - {symbol}")
    print("=" * 60)
    print(_bar("Bars analysed", f"{bars_count:,}"))
    print(_bar("Total trades", str(s["total_trades"])))
    print(_bar("Wins / Losses", f"{wins} / {losses}"))
    print()
    print(_bar("Win rate", f"{s['win_rate']:.1%}"))
    print(_bar("Profit factor", f"{s['profit_factor']:.2f}"))
    print(_bar("Expectancy (R/trade)", f"{s['expectancy_r']:+.3f} R"))
    print(_bar("Total P&L (pips)", f"{s['total_pnl_pips']:+.1f}"))
    print(_bar("Total P&L (R)", f"{s['total_pnl_r']:+.3f}"))
    print()
    print(_bar("Max drawdown (pips)", f"{s['max_drawdown_pips']:.1f}"))
    print(_bar("Max consecutive losses", str(s["max_consecutive_losses"])))
    print()

    if by_reason:
        print("  Exit breakdown:")
        for reason, cnt in sorted(by_reason.items(), key=lambda x: x[1], reverse=True):
            print(f"    {reason.name:<20} {cnt} trade(s)")

    if report.trades:
        avg_risk = sum(t.risk_pips for t in report.trades) / len(report.trades)
        avg_hold = sum(t.exit_bar - t.entry_bar for t in report.trades) / len(report.trades)
        print()
        print(_bar("Avg SL distance (pips)", f"{avg_risk:.1f}"))
        print(_bar("Avg hold (bars)", f"{avg_hold:.1f}"))

    # Gate check
    print()
    if s["total_trades"] >= 5:
        pf = s["profit_factor"]
        wr = s["win_rate"]
        gate_ok = pf >= 1.2 and wr >= 0.40
        gate_str = "[PASS]" if gate_ok else "[FAIL]"
        print(f"  Deployment gate (PF>=1.2, WR>=40%): {gate_str}")
    else:
        print("  Deployment gate: N/A (not enough trades)")
    print("=" * 60)


def print_trade_list(report: BacktestReport, max_rows: int = 20) -> None:
    """Print individual trade rows (most recent first)."""
    if not report.trades:
        return
    trades = list(reversed(report.trades))[:max_rows]
    print(f"\n  Last {len(trades)} trades (most recent first):")
    print(f"  {'#':>4}  {'Dir':<8}  {'Entry':>10}  {'Exit':>10}  {'P&L pips':>10}  {'R':>6}  Reason")
    print("  " + "-" * 65)
    for i, t in enumerate(trades):
        pnl_str = f"{t.pnl_pips:+.1f}"
        r_str = f"{t.pnl_r:+.2f}"
        print(f"  {i+1:>4}  {t.direction.name:<8}  {t.entry_price:>10.5f}  "
              f"{t.exit_price:>10.5f}  {pnl_str:>10}  {r_str:>6}  {t.exit_reason.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward report
# ─────────────────────────────────────────────────────────────────────────────

def print_walk_forward(results) -> None:
    if not results:
        print("\n  Walk-forward: not enough data for windows.")
        return
    print(f"\n{'-'*60}")
    print(f"  WALK-FORWARD VALIDATION ({len(results)} window(s))")
    print(f"{'-'*60}")
    for r in results:
        status = "[PASS]" if r.gate_passed else "[FAIL]"
        oos = r.oos_report
        print(f"\n  Window {r.window_index + 1}: {status}")
        print(f"    IS bars: {r.is_sample_bars:,}  |  OOS bars: {r.oos_bars:,}")
        print(f"    OOS trades  : {oos.total_trades}")
        if oos.total_trades > 0:
            pf = oos.profit_factor
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            print(f"    OOS PF      : {pf_str}")
            print(f"    OOS win rate: {oos.win_rate:.1%}")
        if r.gate_failures:
            for f in r.gate_failures:
                print(f"    [!] {f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M1 Order Block Backtest Runner")
    parser.add_argument("--symbol", default=None,
                        help="Symbol to backtest (default: first in SYMBOLS env)")
    parser.add_argument("--bars", type=int, default=5000,
                        help="Number of M1 bars to fetch (default: 5000 ≈ 3 weeks)")
    parser.add_argument("--slippage", type=float, default=0.5,
                        help="Simulated slippage in pips (default: 0.5)")
    parser.add_argument("--r-multiple", type=float, default=float(os.environ.get("R_MULTIPLE_TP", "2.0")),
                        help="Take-profit R multiple")
    parser.add_argument("--breakeven", type=float, default=float(os.environ.get("BREAKEVEN_AT_R", "1.0")),
                        help="Move SL to BE at N*R")
    parser.add_argument("--displacement", type=float, default=float(os.environ.get("DISPLACEMENT_THRESHOLD", "1.5")),
                        help="Displacement threshold multiplier")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation (70/30 IS/OOS split)")
    parser.add_argument("--trades", action="store_true",
                        help="Print individual trade list")
    parser.add_argument("--cache", action="store_true",
                        help="Cache downloaded data to CSV for faster re-runs")
    # Rev 3
    parser.add_argument("--use-atr-sl", type=lambda x: x.lower() == 'true', default=os.environ.get("USE_ATR_SL", "false").lower() == 'true',
                        help="Use adaptive ATR-based SL")
    parser.add_argument("--atr-multiplier", type=float, default=float(os.environ.get("ATR_SL_MULTIPLIER", "1.0")),
                        help="ATR SL multiplier")
    parser.add_argument("--partial-tp", type=lambda x: x.lower() == 'true', default=os.environ.get("PARTIAL_TP_ENABLED", "false").lower() == 'true',
                        help="Enable partial TP")
    parser.add_argument("--fvg-threshold", type=float, default=float(os.environ.get("FVG_QUALITY_THRESHOLD", "0.0")),
                        help="Min signal quality score for entry [0-1]")
    args = parser.parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    if not connect_mt5():
        sys.exit(1)

    # ── Pick symbol ───────────────────────────────────────────────────────────
    symbol = args.symbol
    if not symbol:
        symbols_env = os.environ.get("SYMBOLS", "EURUSD")
        symbol = symbols_env.split(",")[0].strip()

    print(f"\n  Symbol     : {symbol}")
    print(f"  Bars       : {args.bars:,}")
    print(f"  Slippage   : {args.slippage} pips")
    print(f"  R-multiple : {args.r_multiple}")
    print(f"  Breakeven  : {args.breakeven}R" if args.breakeven else "  Breakeven  : disabled")
    print(f"  Displacement: >= {args.displacement}x avg range")

    # ── Fetch data ────────────────────────────────────────────────────────────
    bars = fetch_m1_bars(symbol, args.bars, use_cache=args.cache)

    # ── Run backtest ──────────────────────────────────────────────────────────
    print(f"\n  Running backtest…")
    engine = BacktestEngine(
        symbol=symbol,
        bars=bars,
        slippage_pips=args.slippage,
        r_multiple=args.r_multiple,
        sl_spread_buffer=float(os.environ.get("SL_SPREAD_BUFFER", "1.5")),
        breakeven_at_r=args.breakeven,
        max_hold_bars=int(os.environ.get("MAX_HOLD_BARS", "40")),
        max_ob_age_bars=int(os.environ.get("MAX_OB_AGE_BARS", "75")),
        max_ob_per_symbol=int(os.environ.get("MAX_OB_PER_SYMBOL", "5")),
        ob_stack_tolerance=float(os.environ.get("OB_STACK_TOLERANCE", "1.5")),
        displacement_threshold=args.displacement,
        trailing_stop_activation_r=float(os.environ.get("TRAILING_STOP_ACTIVATION_R", "0.0")),
        trailing_stop_distance_pips=float(os.environ.get("TRAILING_STOP_DISTANCE_PIPS", "0.0")),
        session_start_utc=os.environ.get("SESSION_START_UTC", "00:00"),
        session_end_utc=os.environ.get("SESSION_END_UTC", "23:59"),
        # Rev 3
        use_atr_sl=args.use_atr_sl,
        atr_sl_multiplier=args.atr_multiplier,
        atr_period=int(os.environ.get("ATR_PERIOD", "14")),
        partial_tp_enabled=args.partial_tp,
        partial_tp_r=float(os.environ.get("PARTIAL_TP_R", "1.0")),
        partial_tp_fraction=float(os.environ.get("PARTIAL_TP_FRACTION", "0.5")),
        fvg_quality_threshold=args.fvg_threshold,
    )
    report = engine.run()

    # ── Print results ─────────────────────────────────────────────────────────
    print_report(report, symbol, len(bars))

    if args.trades:
        print_trade_list(report)

    # ── Walk-forward ──────────────────────────────────────────────────────────
    if args.walk_forward and len(bars) >= 100:
        print(f"\n  Running walk-forward validation…")

        def factory(sym, b):
            return BacktestEngine(
                symbol=sym, bars=b,
                slippage_pips=args.slippage,
                r_multiple=args.r_multiple,
                sl_spread_buffer=float(os.environ.get("SL_SPREAD_BUFFER", "1.5")),
                breakeven_at_r=args.breakeven,
                max_hold_bars=int(os.environ.get("MAX_HOLD_BARS", "40")),
                displacement_threshold=args.displacement,
                trailing_stop_activation_r=float(os.environ.get("TRAILING_STOP_ACTIVATION_R", "0.0")),
                trailing_stop_distance_pips=float(os.environ.get("TRAILING_STOP_DISTANCE_PIPS", "0.0")),
                session_start_utc=os.environ.get("SESSION_START_UTC", "00:00"),
                session_end_utc=os.environ.get("SESSION_END_UTC", "23:59"),
            )

        n_windows = max(1, len(bars) // 2000)
        validator = WalkForwardValidator(factory, split=0.7, min_oos_trades=3)
        wf_results = validator.run(bars, n_windows=n_windows, symbol=symbol)
        print_walk_forward(wf_results)

    mt5.shutdown()
    print("  Done.\n")


if __name__ == "__main__":
    main()
