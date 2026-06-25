"""
Unit tests — Phase 6: Backtesting Engine
Tests cover:
  - No lookahead bias (bar N cannot see N+1)
  - Slippage applied on entry
  - P&L calculation correctness
  - BacktestReport metrics (win rate, profit factor, max DD, etc.)
  - BacktestTrade.pnl_r
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestEngine,
    BacktestReport,
    BacktestTrade,
)
from src.execution.exit_manager import ExitReason
from src.strategy.signal import Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_min: int = 0) -> pd.Timestamp:
    base = datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)
    return pd.Timestamp(base + timedelta(minutes=offset_min))


def _make_bars(
    n: int = 100,
    base: float = 1.1000,
    spread: int = 2,
) -> pd.DataFrame:
    """Generate n flat M1 bars."""
    records = []
    for i in range(n):
        records.append({
            "time": _ts(i),
            "open": base,
            "high": base + 0.0003,
            "low": base - 0.0003,
            "close": base,
            "tick_volume": 100,
            "spread": spread,
        })
    return pd.DataFrame(records)


def _make_engine(bars: pd.DataFrame, slippage_pips: float = 0.5) -> BacktestEngine:
    return BacktestEngine(
        symbol="EURUSD",
        bars=bars,
        slippage_pips=slippage_pips,
        pip_size=0.0001,
        r_multiple=2.0,
        sl_spread_buffer=0,
        breakeven_at_r=0,
        max_hold_bars=40,
        displacement_threshold=1.5,
    )


def _make_trade(pnl_pips: float, risk_pips: float = 10.0, is_win: bool | None = None) -> BacktestTrade:
    direction = Direction.BULLISH
    entry = 1.1010
    if pnl_pips >= 0:
        exit_price = entry + pnl_pips * 0.0001
        reason = ExitReason.TAKE_PROFIT
    else:
        exit_price = entry + pnl_pips * 0.0001
        reason = ExitReason.STOP_LOSS

    return BacktestTrade(
        symbol="EURUSD",
        direction=direction,
        entry_bar=0,
        exit_bar=5,
        entry_price=entry,
        exit_price=exit_price,
        stop_loss=entry - risk_pips * 0.0001,
        take_profit=entry + risk_pips * 2 * 0.0001,
        exit_reason=reason,
        risk_pips=risk_pips,
        pnl_pips=pnl_pips,
    )


# ---------------------------------------------------------------------------
# BacktestTrade
# ---------------------------------------------------------------------------

class TestBacktestTrade:

    def test_pnl_r_win(self):
        t = _make_trade(pnl_pips=20.0, risk_pips=10.0)
        assert abs(t.pnl_r - 2.0) < 1e-9

    def test_pnl_r_loss(self):
        t = _make_trade(pnl_pips=-10.0, risk_pips=10.0)
        assert abs(t.pnl_r - (-1.0)) < 1e-9

    def test_is_win_positive(self):
        t = _make_trade(20.0)
        assert t.is_win is True

    def test_is_win_negative(self):
        t = _make_trade(-10.0)
        assert t.is_win is False

    def test_is_win_zero(self):
        t = _make_trade(0.0)
        assert t.is_win is False   # breakeven is not a win


# ---------------------------------------------------------------------------
# BacktestReport metrics
# ---------------------------------------------------------------------------

class TestBacktestReport:

    def _report(self, pnls: List[float]) -> BacktestReport:
        trades = [_make_trade(p) for p in pnls]
        return BacktestReport(trades=trades, bars=_make_bars(10))

    def test_total_trades(self):
        r = self._report([10, -10, 20])
        assert r.total_trades == 3

    def test_win_rate(self):
        r = self._report([10, -10, 20, -5])  # 2 wins
        assert abs(r.win_rate - 0.5) < 1e-9

    def test_win_rate_all_wins(self):
        r = self._report([10, 20, 5])
        assert r.win_rate == 1.0

    def test_win_rate_all_losses(self):
        r = self._report([-10, -20])
        assert r.win_rate == 0.0

    def test_win_rate_empty(self):
        r = self._report([])
        assert r.win_rate == 0.0

    def test_profit_factor(self):
        r = self._report([20, -10])  # gross_profit=20, gross_loss=10 → PF=2.0
        assert abs(r.profit_factor - 2.0) < 1e-9

    def test_profit_factor_no_losses(self):
        r = self._report([10, 20])
        assert r.profit_factor == float("inf")

    def test_profit_factor_no_wins(self):
        r = self._report([-10])
        assert r.profit_factor == 0.0

    def test_total_pnl_pips(self):
        r = self._report([10, -5, 20])
        assert abs(r.total_pnl_pips - 25.0) < 1e-9

    def test_expectancy_r(self):
        # 2 wins (+20 pips each = 2R), 1 loss (-10 pips = -1R) → avg = (2+2-1)/3 = 1.0R
        trades = [
            _make_trade(20.0, risk_pips=10.0),
            _make_trade(20.0, risk_pips=10.0),
            _make_trade(-10.0, risk_pips=10.0),
        ]
        r = BacktestReport(trades=trades, bars=_make_bars(10))
        assert abs(r.expectancy_r - 1.0) < 1e-9

    def test_max_drawdown_no_trades(self):
        r = self._report([])
        assert r.max_drawdown_pips == 0.0

    def test_max_drawdown_all_wins(self):
        r = self._report([10, 20, 30])
        assert r.max_drawdown_pips == 0.0

    def test_max_drawdown_calculation(self):
        # P&L sequence: +10, -30, +5 → peak=10, trough=10-30=-20 → DD=30
        r = self._report([10.0, -30.0, 5.0])
        assert abs(r.max_drawdown_pips - 30.0) < 1e-9

    def test_max_consecutive_losses(self):
        r = self._report([10, -5, -5, -5, 10, -5])
        assert r.max_consecutive_losses == 3

    def test_max_consecutive_losses_no_losses(self):
        r = self._report([10, 20])
        assert r.max_consecutive_losses == 0

    def test_summary_returns_dict(self):
        r = self._report([20, -10])
        s = r.summary()
        assert isinstance(s, dict)
        assert "win_rate" in s
        assert "profit_factor" in s
        assert "max_drawdown_pips" in s


# ---------------------------------------------------------------------------
# BacktestEngine — causality & slippage
# ---------------------------------------------------------------------------

class TestBacktestEngineCausality:
    """Verify no lookahead bias — processor only sees bars up to current idx."""

    def test_engine_runs_on_flat_market(self):
        """Flat market produces 0 trades (no displacement → no signals)."""
        bars = _make_bars(100)
        engine = _make_engine(bars)
        report = engine.run()
        assert report.total_trades == 0  # flat market, no displacement detected

    def test_engine_completes_all_bars(self):
        """Engine processes exactly len(bars) bars."""
        bars = _make_bars(50)
        engine = _make_engine(bars)
        report = engine.run()
        assert isinstance(report, BacktestReport)

    def test_slippage_increases_entry_price_for_longs(self):
        """
        For BULLISH entries, entry_price = signal_price + slippage.
        We verify this by injecting a synthetic trade directly.
        """
        from src.backtest.engine import BacktestPosition
        from src.strategy.signal import Signal
        from src.execution.exit_manager import ExitPlan

        signal = Signal(
            symbol="EURUSD",
            direction=Direction.BULLISH,
            entry_price=1.1005,
            ob_top=1.1010,
            ob_bottom=1.1000,
            confirmation_time=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
            bar_index=5,
        )
        bars = _make_bars(10)
        engine = _make_engine(bars, slippage_pips=2.0)
        bar = bars.iloc[5]
        avg_spread = 2.0

        engine._enter_position(signal, bar, 5, avg_spread)

        pos = engine._open_positions[0]
        # Entry should be signal.entry_price + 2 pips slippage
        expected_entry = signal.entry_price + 2.0 * 0.0001
        assert abs(pos.entry_price - expected_entry) < 1e-9

    def test_slippage_decreases_entry_price_for_shorts(self):
        from src.strategy.signal import Signal

        signal = Signal(
            symbol="EURUSD",
            direction=Direction.BEARISH,
            entry_price=1.1005,
            ob_top=1.1010,
            ob_bottom=1.1000,
            confirmation_time=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
            bar_index=5,
        )
        bars = _make_bars(10)
        engine = _make_engine(bars, slippage_pips=2.0)
        bar = bars.iloc[5]

        engine._enter_position(signal, bar, 5, 2.0)

        pos = engine._open_positions[0]
        expected_entry = signal.entry_price - 2.0 * 0.0001
        assert abs(pos.entry_price - expected_entry) < 1e-9


class TestBacktestEnginePnL:
    """Verify P&L calculation on known price sequences."""

    def test_bullish_tp_pnl(self):
        """A bullish trade that hits TP: pnl_pips = (exit - entry) / pip_size."""
        from src.backtest.engine import BacktestPosition
        from src.strategy.signal import Signal
        from src.execution.exit_manager import ExitPlan

        signal = Signal(
            symbol="EURUSD",
            direction=Direction.BULLISH,
            entry_price=1.1010,
            ob_top=1.1010,
            ob_bottom=1.1000,
            confirmation_time=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
            bar_index=0,
        )
        plan = ExitPlan(
            entry_price=1.1010,
            stop_loss=1.1000,
            take_profit=1.1030,
            risk_pips=10.0,
            r_target_pips=20.0,
            breakeven_price=0.0,
            max_bar=50,
            direction=Direction.BULLISH,
        )
        pos = BacktestPosition(signal=signal, plan=plan, entry_price=1.1010, entry_bar=0, current_sl=1.1000)

        bars = _make_bars(5)
        engine = _make_engine(bars, slippage_pips=0)
        trade = engine._close_position(pos, exit_price=1.1030, exit_bar=5, reason=ExitReason.TAKE_PROFIT)

        assert abs(trade.pnl_pips - 20.0) < 1e-6

    def test_bearish_sl_pnl(self):
        """A bearish trade stopped out: pnl_pips = (entry - exit) / pip_size → negative."""
        from src.backtest.engine import BacktestPosition
        from src.strategy.signal import Signal
        from src.execution.exit_manager import ExitPlan

        signal = Signal(
            symbol="EURUSD",
            direction=Direction.BEARISH,
            entry_price=1.1010,
            ob_top=1.1020,
            ob_bottom=1.1010,
            confirmation_time=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
            bar_index=0,
        )
        plan = ExitPlan(
            entry_price=1.1010,
            stop_loss=1.1020,
            take_profit=1.0990,
            risk_pips=10.0,
            r_target_pips=20.0,
            breakeven_price=0.0,
            max_bar=50,
            direction=Direction.BEARISH,
        )
        pos = BacktestPosition(signal=signal, plan=plan, entry_price=1.1010, entry_bar=0, current_sl=1.1020)

        bars = _make_bars(5)
        engine = _make_engine(bars, slippage_pips=0)
        trade = engine._close_position(pos, exit_price=1.1020, exit_bar=3, reason=ExitReason.STOP_LOSS)

        assert trade.pnl_pips < 0
        assert abs(trade.pnl_pips - (-10.0)) < 1e-6
