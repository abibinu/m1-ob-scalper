import argparse
import os
import sys

import optuna
from dotenv import load_dotenv

from run_backtest import connect_mt5, fetch_m1_bars
from src.backtest.engine import BacktestEngine
import MetaTrader5 as mt5

load_dotenv()

def objective(trial, symbol, bars):
    # Hyperparameter search space
    r_multiple = trial.suggest_float("r_multiple", 1.0, 4.0, step=0.1)
    sl_spread_buffer = trial.suggest_float("sl_spread_buffer", 1.0, 3.0, step=0.1)
    breakeven_at_r = trial.suggest_float("breakeven_at_r", 0.0, 2.0, step=0.1)
    max_hold_bars = trial.suggest_int("max_hold_bars", 15, 120, step=5)
    max_ob_age_bars = trial.suggest_int("max_ob_age_bars", 30, 150, step=5)
    displacement_threshold = trial.suggest_float("displacement_threshold", 1.0, 2.5, step=0.1)
    
    # Trailing Stop
    use_trailing = trial.suggest_categorical("use_trailing", [True, False])
    if use_trailing:
        trailing_stop_activation_r = trial.suggest_float("trailing_stop_activation_r", 0.5, 2.5, step=0.1)
        trailing_stop_distance_pips = trial.suggest_float("trailing_stop_distance_pips", 5.0, 25.0, step=1.0)
    else:
        trailing_stop_activation_r = 0.0
        trailing_stop_distance_pips = 0.0

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
        session_start_utc=os.environ.get("SESSION_START_UTC", "00:00"),
        session_end_utc=os.environ.get("SESSION_END_UTC", "23:59"),
    )
    
    report = engine.run()
    
    # Penalize if very few trades are taken
    if report.total_trades < 20:
        return -100.0  # Heavy penalty for low trade count
        
    # Custom score: Total R multiplied by a logarithmic trade frequency modifier.
    # This forces Optuna to seek out parameters that trade more often, while still demanding profitability.
    import math
    if report.total_pnl_r <= 0:
        score = report.total_pnl_r
    else:
        score = report.total_pnl_r * math.log10(report.total_trades)
    
    trial.set_user_attr("profit_factor", report.profit_factor)
    trial.set_user_attr("win_rate", report.win_rate)
    trial.set_user_attr("trades", report.total_trades)
    
    return score

def main():
    parser = argparse.ArgumentParser(description="Optimize parameters using Optuna")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to optimize")
    parser.add_argument("--bars", type=int, default=10000, help="Number of bars to use for optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
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
    print(f"Running {args.trials} trials. This may take some time...\n")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    
    def wrapped_objective(trial):
        return objective(trial, args.symbol, bars)
        
    study.optimize(wrapped_objective, n_trials=args.trials, show_progress_bar=True)
    
    print("\n=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    best = study.best_trial
    print(f"Best Score (Total R): {best.value:.2f}")
    print(f"  Trades        : {best.user_attrs.get('trades')}")
    print(f"  Win Rate      : {best.user_attrs.get('win_rate'):.1%}")
    print(f"  Profit Factor : {best.user_attrs.get('profit_factor'):.2f}")
    
    print("\nBest Parameters to put in your .env file:")
    for k, v in best.params.items():
        if k == "use_trailing":
            continue
        env_key = k.upper()
        if k == "max_ob_age_bars":
            env_key = "MAX_OB_AGE_BARS"
        elif k == "displacement_threshold":
            env_key = "DISPLACEMENT_THRESHOLD"
        elif k == "r_multiple":
            env_key = "R_MULTIPLE_TP"
            
        print(f"  {env_key}={v}")

if __name__ == "__main__":
    main()
