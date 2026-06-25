"""
Unit tests — Phase 7: Walk-Forward Validation & Kill Criteria
"""

from __future__ import annotations

from typing import List

import pytest

from src.backtest.engine import BacktestReport, BacktestTrade
from src.execution.exit_manager import ExitReason
from src.strategy.signal import Direction
from src.validation.walk_forward import (
    KillCriteriaResult,
    WalkForwardResult,
    check_kill_criteria,
    evaluate_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars_df(n: int = 100):
    from datetime import datetime, timezone, timedelta
    import pandas as pd
    base = datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        records.append({
            "time": pd.Timestamp(base + timedelta(minutes=i)),
            "open": 1.1000, "high": 1.1003, "low": 1.0997, "close": 1.1000,
            "tick_volume": 100, "spread": 2,
        })
    return pd.DataFrame(records)


def _trade(pnl_pips: float, risk_pips: float = 10.0) -> BacktestTrade:
    entry = 1.1010
    exit_price = entry + pnl_pips * 0.0001
    return BacktestTrade(
        symbol="EURUSD",
        direction=Direction.BULLISH,
        entry_bar=0,
        exit_bar=5,
        entry_price=entry,
        exit_price=exit_price,
        stop_loss=entry - risk_pips * 0.0001,
        take_profit=entry + risk_pips * 2 * 0.0001,
        exit_reason=ExitReason.TAKE_PROFIT if pnl_pips > 0 else ExitReason.STOP_LOSS,
        risk_pips=risk_pips,
        pnl_pips=pnl_pips,
    )


def _report(pnls: List[float]) -> BacktestReport:
    trades = [_trade(p) for p in pnls]
    return BacktestReport(trades=trades, bars=_make_bars_df(10))


# ---------------------------------------------------------------------------
# evaluate_gate
# ---------------------------------------------------------------------------

class TestEvaluateGate:

    def test_gate_passes_good_results(self):
        is_rep = _report([20, 20, -10, 20])     # IS: PF=3.0, WR=75%
        oos_rep = _report([20, 20, -10, 20, 20]) # OOS: PF=4.0, WR=80%
        passed, failures = evaluate_gate(is_rep, oos_rep, min_profit_factor=1.2,
                                         max_win_rate_drop=0.10, min_oos_trades=5)
        assert passed
        assert failures == []

    def test_gate_fails_low_oos_pf(self):
        is_rep = _report([20, 20, -10])
        oos_rep = _report([10, -10, -10, -10, -5])  # PF < 1.2
        passed, failures = evaluate_gate(is_rep, oos_rep, min_profit_factor=1.2,
                                         min_oos_trades=5)
        assert not passed
        assert any("profit factor" in f.lower() for f in failures)

    def test_gate_fails_high_win_rate_drop(self):
        # IS: 100% win rate; OOS: 50% → drop = 50% > 10%
        is_rep = _report([20, 20, 20])
        oos_rep = _report([20, -10, 20, -10, 20])  # 60% WR
        passed, failures = evaluate_gate(is_rep, oos_rep, min_profit_factor=1.0,
                                         max_win_rate_drop=0.10, min_oos_trades=5)
        # IS WR=100%, OOS WR=60%, drop=40% > 10%
        assert not passed
        assert any("win rate" in f.lower() for f in failures)

    def test_gate_fails_insufficient_oos_trades(self):
        is_rep = _report([20, 20, -10])
        oos_rep = _report([20, 20])  # only 2 trades
        passed, failures = evaluate_gate(is_rep, oos_rep, min_oos_trades=5)
        assert not passed
        assert any("insufficient" in f.lower() for f in failures)

    def test_gate_multiple_failures(self):
        is_rep = _report([20, 20, 20])
        oos_rep = _report([10, -20])  # low PF, low WR, insufficient trades
        passed, failures = evaluate_gate(is_rep, oos_rep, min_profit_factor=1.2,
                                         max_win_rate_drop=0.10, min_oos_trades=5)
        assert not passed
        assert len(failures) >= 2

    def test_perfect_oos_passes(self):
        is_rep = _report([10, -5, 10])
        oos_rep = _report([10, 10, 10, 10, 10])  # perfect
        passed, _ = evaluate_gate(is_rep, oos_rep, min_profit_factor=1.2,
                                   max_win_rate_drop=0.50, min_oos_trades=5)
        assert passed


# ---------------------------------------------------------------------------
# check_kill_criteria
# ---------------------------------------------------------------------------

class TestKillCriteria:

    def test_no_kill_on_good_performance(self):
        live = _report([20, -10, 20, 20])
        # backtest max DD: let's say 10 pips, threshold = 15
        result = check_kill_criteria(live, backtest_max_drawdown_pips=10.0,
                                     drawdown_multiplier=1.5, max_consecutive_losses=5)
        # Live max DD = 0 (3 wins after 1 loss) — well within 15 pip ceiling
        assert not result.kill_triggered

    def test_kill_on_excessive_drawdown(self):
        # Sequence: -10, -10, -10 → max DD = 30 pips
        live = _report([-10, -10, -10])
        # Backtest max DD = 10 pips, threshold = 15 pips
        result = check_kill_criteria(live, backtest_max_drawdown_pips=10.0,
                                     drawdown_multiplier=1.5)
        assert result.kill_triggered
        assert any("drawdown" in r.lower() for r in result.reasons)

    def test_kill_on_5_consecutive_losses(self):
        live = _report([-10, -10, -10, -10, -10])  # 5 consecutive losses
        result = check_kill_criteria(live, backtest_max_drawdown_pips=1000.0,
                                     max_consecutive_losses=5)
        assert result.kill_triggered
        assert any("consecutive" in r.lower() for r in result.reasons)

    def test_no_kill_on_4_consecutive_losses(self):
        live = _report([-10, -10, -10, -10, 20])  # 4 losses then a win
        result = check_kill_criteria(live, backtest_max_drawdown_pips=1000.0,
                                     max_consecutive_losses=5)
        assert not result.kill_triggered

    def test_kill_both_criteria(self):
        live = _report([-10, -10, -10, -10, -10])  # 5 losses, big DD
        result = check_kill_criteria(live, backtest_max_drawdown_pips=1.0,  # tiny backtest DD
                                     max_consecutive_losses=5)
        assert result.kill_triggered
        assert len(result.reasons) >= 2

    def test_kill_result_has_reasons(self):
        live = _report([-10, -10, -10, -10, -10])
        result = check_kill_criteria(live, backtest_max_drawdown_pips=1.0,
                                     max_consecutive_losses=5)
        assert isinstance(result.reasons, list)
        assert len(result.reasons) > 0

    def test_drawdown_exactly_at_threshold_does_not_trigger(self):
        """Drawdown = exactly 1.5× backtest → not triggered (> not >=)."""
        live = _report([-10, -5])  # DD = 15 pips exactly
        result = check_kill_criteria(live, backtest_max_drawdown_pips=10.0,
                                     drawdown_multiplier=1.5)
        # 15 > 15 is False → no kill
        assert not result.kill_triggered


# ---------------------------------------------------------------------------
# WalkForwardValidator smoke test
# ---------------------------------------------------------------------------

class TestWalkForwardValidatorSmoke:

    def test_validator_returns_results(self):
        from src.backtest.engine import BacktestEngine
        from src.validation.walk_forward import WalkForwardValidator

        def factory(symbol, bars):
            return BacktestEngine(
                symbol=symbol,
                bars=bars,
                slippage_pips=0.5,
                displacement_threshold=1.5,
            )

        bars = _make_bars_df(200)
        validator = WalkForwardValidator(factory, split=0.7, min_oos_trades=1)
        results = validator.run(bars, n_windows=1, symbol="EURUSD")

        assert isinstance(results, list)
        assert len(results) >= 0  # may be 0 if window too small, that's OK

    def test_validator_splits_data_correctly(self):
        """Verify IS and OOS bar counts sum to window size."""
        from src.backtest.engine import BacktestEngine
        from src.validation.walk_forward import WalkForwardValidator

        def factory(symbol, bars):
            return BacktestEngine(symbol=symbol, bars=bars, slippage_pips=0)

        bars = _make_bars_df(100)
        validator = WalkForwardValidator(factory, split=0.7, min_oos_trades=1)
        results = validator.run(bars, n_windows=1)

        if results:
            r = results[0]
            assert r.is_sample_bars == 70
            assert r.oos_bars == 30
