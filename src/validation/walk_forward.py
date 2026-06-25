"""
Module 6: Validation & Deployment Gate
=======================================
Per SDD Rev 2, Sections 7.1, 7.2, 7.3:

  7.1 Walk-Forward Validation
      - Split data into sequential in-sample / out-of-sample windows
      - Gate: profit_factor >= min_profit_factor on OOS data
      - Gate: win rate drop from IS→OOS < 10 percentage points
      - If gate fails → BLOCKED (overfitting evidence)

  7.3 Kill Criteria (live deployment)
      - Drawdown > 1.5× backtest max_drawdown
      - 5 consecutive confirmed losses
      - Unhandled exception in live path → hard-stop

This module does NOT execute trades. It evaluates BacktestReport objects
produced by BacktestEngine and returns structured pass/fail results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.backtest.engine import BacktestReport
from src.core.logger import get_logger

log = get_logger(__name__)


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """
    Result of a single walk-forward window evaluation.

    Attributes:
        window_index:   0-based index of this window.
        is_sample_bars: Number of in-sample bars.
        oos_bars:       Number of out-of-sample bars.
        is_report:      Backtest report for in-sample period.
        oos_report:     Backtest report for out-of-sample period.
        gate_passed:    True if OOS results meet the gate criteria.
        gate_failures:  Human-readable list of failed criteria.
    """
    window_index: int
    is_sample_bars: int
    oos_bars: int
    is_report: BacktestReport
    oos_report: BacktestReport
    gate_passed: bool = True
    gate_failures: List[str] = field(default_factory=list)


# ── Gate criteria ─────────────────────────────────────────────────────────────

def evaluate_gate(
    is_report: BacktestReport,
    oos_report: BacktestReport,
    min_profit_factor: float = 1.2,
    max_win_rate_drop: float = 0.10,
    min_oos_trades: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Apply walk-forward gate criteria per SDD Section 7.1.

    Gates:
      1. OOS profit factor >= min_profit_factor
      2. OOS win rate drop from IS win rate <= max_win_rate_drop
      3. Minimum number of OOS trades (statistical significance)

    Returns:
        (passed, list_of_failure_reasons)
    """
    failures = []

    # Gate 1: Minimum OOS trades
    if oos_report.total_trades < min_oos_trades:
        failures.append(
            f"Insufficient OOS trades: {oos_report.total_trades} < {min_oos_trades} minimum"
        )

    # Gate 2: OOS profit factor
    if oos_report.profit_factor < min_profit_factor:
        failures.append(
            f"OOS profit factor {oos_report.profit_factor:.2f} < "
            f"minimum {min_profit_factor:.2f} — possible overfit"
        )

    # Gate 3: Win rate degradation
    is_wr = is_report.win_rate
    oos_wr = oos_report.win_rate
    drop = is_wr - oos_wr
    if drop > max_win_rate_drop:
        failures.append(
            f"Win rate drop {drop:.1%} exceeds {max_win_rate_drop:.1%} threshold "
            f"(IS={is_wr:.1%}, OOS={oos_wr:.1%}) — overfitting evidence"
        )

    passed = len(failures) == 0
    if passed:
        pf_str = "inf" if oos_report.profit_factor == float("inf") else f"{oos_report.profit_factor:.2f}"
        log.info("Walk-forward gate PASSED: PF=%s WR=%.1f%%", pf_str, oos_wr * 100)
    else:
        log.warning("Walk-forward gate FAILED: %s", "; ".join(failures))

    return passed, failures


# ── Kill criteria (live) ──────────────────────────────────────────────────────

@dataclass
class KillCriteriaResult:
    """Outcome of a live kill-criteria check."""
    kill_triggered: bool
    reasons: List[str] = field(default_factory=list)


def check_kill_criteria(
    live_report: BacktestReport,
    backtest_max_drawdown_pips: float,
    drawdown_multiplier: float = 1.5,
    max_consecutive_losses: int = 5,
) -> KillCriteriaResult:
    """
    SDD Section 7.3 kill criteria.

    Triggers a full strategy review and trading pause if:
      1. Live drawdown > drawdown_multiplier × backtest_max_drawdown
      2. max_consecutive_losses consecutive confirmed losing trades
      3. (Unhandled exceptions are handled at the entry point — not here)

    Args:
        live_report:                  BacktestReport of live/demo trades so far.
        backtest_max_drawdown_pips:   Max drawdown observed in historical backtest.
        drawdown_multiplier:          Default 1.5× (SDD 7.3).
        max_consecutive_losses:       Default 5 (SDD 7.3).

    Returns:
        KillCriteriaResult with kill_triggered flag and reason list.
    """
    reasons = []

    # Criterion 1: Drawdown threshold
    dd_ceiling = drawdown_multiplier * backtest_max_drawdown_pips
    if live_report.max_drawdown_pips > dd_ceiling:
        reasons.append(
            f"Drawdown {live_report.max_drawdown_pips:.1f} pips exceeds "
            f"{drawdown_multiplier:.1f}× backtest DD ({dd_ceiling:.1f} pips)"
        )

    # Criterion 2: Consecutive losses
    if live_report.max_consecutive_losses >= max_consecutive_losses:
        reasons.append(
            f"{live_report.max_consecutive_losses} consecutive confirmed losses "
            f"(threshold={max_consecutive_losses}) — possible regime change"
        )

    kill = len(reasons) > 0
    if kill:
        log.critical("KILL CRITERIA TRIGGERED: %s", "; ".join(reasons))
    return KillCriteriaResult(kill_triggered=kill, reasons=reasons)


# ── WalkForwardValidator ──────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Orchestrates walk-forward validation across sequential windows.

    Usage::

        validator = WalkForwardValidator(engine_factory, split=0.7)
        results = validator.run(bars, n_windows=3)
        all_passed = all(r.gate_passed for r in results)
    """

    def __init__(
        self,
        engine_factory,   # callable(symbol, bars) → BacktestEngine
        split: float = 0.7,
        min_profit_factor: float = 1.2,
        max_win_rate_drop: float = 0.10,
        min_oos_trades: int = 5,
    ) -> None:
        self.engine_factory = engine_factory
        self.split = split
        self.min_profit_factor = min_profit_factor
        self.max_win_rate_drop = max_win_rate_drop
        self.min_oos_trades = min_oos_trades

    def run(
        self,
        bars,             # pd.DataFrame
        n_windows: int = 1,
        symbol: str = "EURUSD",
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward validation with ``n_windows`` sequential windows.

        Each window covers total_bars / n_windows bars.
        Within each window, split% is in-sample and (1-split)% is out-of-sample.
        """
        import pandas as pd
        import numpy as np

        total = len(bars)
        window_size = total // n_windows
        results = []

        for w in range(n_windows):
            start = w * window_size
            end = start + window_size if w < n_windows - 1 else total
            window_bars = bars.iloc[start:end].reset_index(drop=True)

            is_size = int(len(window_bars) * self.split)
            is_bars = window_bars.iloc[:is_size]
            oos_bars = window_bars.iloc[is_size:]

            if len(is_bars) < 30 or len(oos_bars) < 10:
                log.warning("Window %d too small — skipping (IS=%d, OOS=%d)",
                            w, len(is_bars), len(oos_bars))
                continue

            is_engine = self.engine_factory(symbol, is_bars)
            oos_engine = self.engine_factory(symbol, oos_bars)

            is_report = is_engine.run()
            oos_report = oos_engine.run()

            passed, failures = evaluate_gate(
                is_report, oos_report,
                self.min_profit_factor,
                self.max_win_rate_drop,
                self.min_oos_trades,
            )

            results.append(WalkForwardResult(
                window_index=w,
                is_sample_bars=len(is_bars),
                oos_bars=len(oos_bars),
                is_report=is_report,
                oos_report=oos_report,
                gate_passed=passed,
                gate_failures=failures,
            ))

        return results
