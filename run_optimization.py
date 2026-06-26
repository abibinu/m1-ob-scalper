import argparse
import math
import os
import sys
import pandas as pd
import numpy as np

import optuna
from dotenv import load_dotenv

from run_backtest import connect_mt5, fetch_m1_bars
from src.backtest.engine import BacktestEngine
import MetaTrader5 as mt5

load_dotenv()


# ── Rev 3: Sortino-based scoring ──────────────────────────────────────────────

def compute_sortino(trades, target_r: float = 0.0) -> float:
    """
    Compute Sortino ratio from a list of BacktestTrade objects.

    Sortino = (mean_r - target_r) / downside_deviation

    Unlike Sharpe, Sortino only penalizes downside (losing) trades.
    This is ideal for scalping systems where the upside tail (TP hits)
    should not be penalized.

    Args:
        trades:   List of BacktestTrade with .pnl_r attribute.
        target_r: Minimum acceptable return per trade (default 0 = breakeven).

    Returns:
        Sortino ratio. Returns 0.0 if insufficient data or no downside.
    """
    if len(trades) < 5:
        return 0.0
    r_values = [t.pnl_r for t in trades]
    mean_r = np.mean(r_values)
    downside = [min(r - target_r, 0) ** 2 for r in r_values]
    downside_dev = math.sqrt(np.mean(downside))
    if downside_dev == 0:
        return mean_r * 10.0  # all wins — reward heavily
    return (mean_r - target_r) / downside_dev


def objective(trial, symbol: str, bars: pd.DataFrame) -> float:
    # ── Core parameters ───────────────────────────────────────────────────────
    r_multiple = trial.suggest_float("r_multiple", 1.0, 4.0, step=0.1)
    sl_spread_buffer = trial.suggest_float("sl_spread_buffer", 1.0, 3.0, step=0.1)
    breakeven_at_r = trial.suggest_float("breakeven_at_r", 0.0, 2.0, step=0.1)
    max_hold_bars = trial.suggest_int("max_hold_bars", 15, 120, step=5)
    max_ob_age_bars = trial.suggest_int("max_ob_age_bars", 30, 150, step=5)
    displacement_threshold = trial.suggest_float("displacement_threshold", 0.8, 2.5, step=0.1)

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    use_trailing = trial.suggest_categorical("use_trailing", [True, False])
    if use_trailing:
        trailing_stop_activation_r = trial.suggest_float("trailing_stop_activation_r", 0.5, 2.5, step=0.1)
        trailing_stop_distance_pips = trial.suggest_float("trailing_stop_distance_pips", 5.0, 25.0, step=1.0)
    else:
        trailing_stop_activation_r = 0.0
        trailing_stop_distance_pips = 0.0

    # ── Rev 3: ATR-based SL ───────────────────────────────────────────────────
    use_atr_sl = trial.suggest_categorical("use_atr_sl", [True, False])
    if use_atr_sl:
        atr_sl_multiplier = trial.suggest_float("atr_sl_multiplier", 0.5, 3.0, step=0.1)
        atr_period = trial.suggest_int("atr_period", 7, 21, step=7)
    else:
        atr_sl_multiplier = 1.0
        atr_period = 14

    # ── Rev 3: Partial TP ─────────────────────────────────────────────────────
    partial_tp_enabled = trial.suggest_categorical("partial_tp_enabled", [True, False])
    if partial_tp_enabled:
        partial_tp_r = trial.suggest_float("partial_tp_r", 0.5, 1.5, step=0.1)
        partial_tp_fraction = trial.suggest_float("partial_tp_fraction", 0.3, 0.7, step=0.1)
    else:
        partial_tp_r = 1.0
        partial_tp_fraction = 0.5

    # ── Rev 3: FVG quality threshold ──────────────────────────────────────────
    fvg_quality_threshold = trial.suggest_float("fvg_quality_threshold", 0.0, 0.5, step=0.05)

    engine = BacktestEngine(
        symbol=symbol,
        bars=bars,
        slippage_pips=0.5,
        r_multiple=r_multiple,
        sl_spread_buffer=sl_spread_buffer,
        breakeven_at_r=breakeven_at_r,
        max_hold_bars=max_hold_bars,
        max_ob_age_bars=max_ob_age_bars,
        displacement_threshold=displacement_threshold,
        trailing_stop_activation_r=trailing_stop_activation_r,
        trailing_stop_distance_pips=trailing_stop_distance_pips,
        use_atr_sl=use_atr_sl,
        atr_sl_multiplier=atr_sl_multiplier,
        atr_period=atr_period,
        partial_tp_enabled=partial_tp_enabled,
        partial_tp_r=partial_tp_r,
        partial_tp_fraction=partial_tp_fraction,
        fvg_quality_threshold=fvg_quality_threshold,
        session_start_utc=os.environ.get("SESSION_START_UTC", "00:00"),
        session_end_utc=os.environ.get("SESSION_END_UTC", "23:59"),
    )

    report = engine.run()

    # Penalize if very few trades are taken
    if report.total_trades < 10:
        return -100.0  # Heavy penalty for low trade count

    # ── Rev 3: Composite Sortino-based objective ───────────────────────────────
    # sortino: penalizes downside volatility, not raw P&L
    # log10(trades): rewards trade frequency (critical for HFT scalping)
    # profit_factor: rewards gross profit over gross loss
    sortino = compute_sortino(report.trades)
    if sortino <= 0:
        score = sortino  # direct penalty
    else:
        freq_modifier = math.log10(max(report.total_trades, 1))
        pf = min(report.profit_factor, 10.0)  # cap to prevent outlier domination
        score = sortino * freq_modifier * pf

    trial.set_user_attr("profit_factor", report.profit_factor)
    trial.set_user_attr("win_rate", report.win_rate)
    trial.set_user_attr("trades", report.total_trades)
    trial.set_user_attr("sortino", sortino)
    trial.set_user_attr("expectancy_r", report.expectancy_r)

    return score


def main():
    parser = argparse.ArgumentParser(description="M1 OB Scalper \u2014 Optuna Optimizer (Rev 3 Sortino)")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to optimize")
    parser.add_argument("--bars", type=int, default=10000, help="Number of bars to use for optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (n_jobs for Optuna)")
    args = parser.parse_args()

    sys.stdout = open(os.devnull, 'w')
    if not connect_mt5():
        sys.stdout = sys.__stdout__
        print("Failed to connect to MT5.")
        sys.exit(1)

    bars = fetch_m1_bars(args.symbol, args.bars, use_cache=True)
    mt5.shutdown()

    sys.stdout = sys.__stdout__
    print(f"Loaded {len(bars)} bars for optimization.")
    print(f"Running {args.trials} trials (Sortino objective). This may take some time...\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")

    def wrapped_objective(trial):
        return objective(trial, args.symbol, bars)

    study.optimize(wrapped_objective, n_trials=args.trials, show_progress_bar=True,
                   n_jobs=args.jobs)

    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE (Rev 3 — Sortino Objective)")
    print("=" * 60)

    best = study.best_trial
    print(f"Best Score (Sortino \u00d7 logTrades \u00d7 PF): {best.value:.3f}")
    if best.value == -100.0 or not best.user_attrs:
        print("  Trades        : None (All trials failed the minimum trade limit)")
    else:
        print(f"  Trades        : {best.user_attrs.get('trades')}")
        print(f"  Win Rate      : {best.user_attrs.get('win_rate'):.1%}")
        print(f"  Profit Factor : {best.user_attrs.get('profit_factor'):.2f}")
        print(f"  Sortino Ratio : {best.user_attrs.get('sortino', 0):.3f}")
        print(f"  Expectancy    : {best.user_attrs.get('expectancy_r', 0):+.3f} R/trade")

    print("\nBest Parameters to put in your .env file:")
    env_key_map = {
        "r_multiple":                "R_MULTIPLE_TP",
        "sl_spread_buffer":          "SL_SPREAD_BUFFER",
        "breakeven_at_r":            "BREAKEVEN_AT_R",
        "max_hold_bars":             "MAX_HOLD_BARS",
        "max_ob_age_bars":           "MAX_OB_AGE_BARS",
        "displacement_threshold":    "DISPLACEMENT_THRESHOLD",
        "trailing_stop_activation_r":"TRAILING_STOP_ACTIVATION_R",
        "trailing_stop_distance_pips":"TRAILING_STOP_DISTANCE_PIPS",
        "atr_sl_multiplier":         "ATR_SL_MULTIPLIER",
        "atr_period":                "ATR_PERIOD",
        "partial_tp_r":              "PARTIAL_TP_R",
        "partial_tp_fraction":       "PARTIAL_TP_FRACTION",
        "fvg_quality_threshold":     "FVG_QUALITY_THRESHOLD",
    }
    skip_keys = {"use_trailing", "use_atr_sl", "partial_tp_enabled"}

    for k, v in best.params.items():
        if k in skip_keys:
            # Write bool as enabled flag
            if k == "use_atr_sl":
                print(f"  USE_ATR_SL={'true' if v else 'false'}")
            elif k == "partial_tp_enabled":
                print(f"  PARTIAL_TP_ENABLED={'true' if v else 'false'}")
            continue
        env_key = env_key_map.get(k, k.upper())
        print(f"  {env_key}={v}")


if __name__ == "__main__":
    main()
