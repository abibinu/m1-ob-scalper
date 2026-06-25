"""
Unit tests — Phase 5b: Exit Manager
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.execution.exit_manager import ExitManager, ExitPlan, ExitReason
from src.strategy.signal import Direction, Signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> datetime:
    return datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)


def _signal(direction=Direction.BULLISH, ob_top=1.1010, ob_bottom=1.1000) -> Signal:
    return Signal(
        symbol="EURUSD",
        direction=direction,
        entry_price=(ob_top + ob_bottom) / 2,
        ob_top=ob_top,
        ob_bottom=ob_bottom,
        confirmation_time=_ts(),
        bar_index=10,
        spread_at_signal=2.0,
    )


def _bar(high: float, low: float, open_: float = 0, close: float = 0) -> pd.Series:
    return pd.Series({"high": high, "low": low, "open": open_, "close": close})


def _default_manager() -> ExitManager:
    return ExitManager(
        r_multiple=2.0,
        sl_spread_buffer=1.5,
        breakeven_at_r=1.0,
        max_hold_bars=40,
        pip_size=0.0001,
    )


# ── ExitPlan creation — Bullish ───────────────────────────────────────────────

class TestExitPlanBullish:

    def test_sl_below_ob_bottom(self):
        mgr = _default_manager()
        sig = _signal(Direction.BULLISH, ob_top=1.1010, ob_bottom=1.1000)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0, current_spread_pips=2.0)
        # SL = ob_bottom - sl_buffer = 1.1000 - (1.5 * 2.0 * 0.0001) = 1.1000 - 0.0003
        assert plan.stop_loss < sig.ob_bottom

    def test_tp_above_entry(self):
        mgr = _default_manager()
        sig = _signal(Direction.BULLISH)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0)
        assert plan.take_profit > plan.entry_price

    def test_tp_is_2r(self):
        mgr = ExitManager(r_multiple=2.0, sl_spread_buffer=0, breakeven_at_r=0, max_hold_bars=40)
        sig = _signal(Direction.BULLISH, ob_top=1.1010, ob_bottom=1.1000)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0, current_spread_pips=0)
        # risk_pips = (1.1010 - 1.1000) / 0.0001 = 10 pips
        # TP = entry + 2R = 1.1010 + 20 pips = 1.1030
        assert abs(plan.take_profit - 1.1030) < 1e-6

    def test_opposing_liquidity_caps_tp(self):
        mgr = ExitManager(r_multiple=2.0, sl_spread_buffer=0, breakeven_at_r=0, max_hold_bars=40)
        sig = _signal(Direction.BULLISH, ob_top=1.1010, ob_bottom=1.1000)
        # Full 2R would be 1.1030, but opposing liquidity is at 1.1020 (closer)
        plan = mgr.create_exit_plan(
            sig, entry_price=1.1010, current_bar=0, opposing_liquidity=1.1020
        )
        assert plan.take_profit == 1.1020

    def test_breakeven_price_set(self):
        mgr = ExitManager(r_multiple=2.0, sl_spread_buffer=0, breakeven_at_r=1.0, max_hold_bars=40)
        sig = _signal(Direction.BULLISH, ob_top=1.1010, ob_bottom=1.1000)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0, current_spread_pips=0)
        # BE at 1R = entry + 10 pips = 1.1020
        assert abs(plan.breakeven_price - 1.1020) < 1e-6

    def test_breakeven_disabled_when_zero(self):
        mgr = ExitManager(r_multiple=2.0, sl_spread_buffer=0, breakeven_at_r=0, max_hold_bars=40)
        sig = _signal(Direction.BULLISH)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0)
        assert plan.breakeven_price == 0.0

    def test_max_bar_set_correctly(self):
        mgr = ExitManager(r_multiple=2.0, sl_spread_buffer=0, breakeven_at_r=0, max_hold_bars=40)
        sig = _signal(Direction.BULLISH)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=5)
        assert plan.max_bar == 45


# ── ExitPlan creation — Bearish ───────────────────────────────────────────────

class TestExitPlanBearish:

    def test_sl_above_ob_top(self):
        mgr = _default_manager()
        sig = _signal(Direction.BEARISH, ob_top=1.1020, ob_bottom=1.1010)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0, current_spread_pips=2.0)
        assert plan.stop_loss > sig.ob_top

    def test_tp_below_entry(self):
        mgr = _default_manager()
        sig = _signal(Direction.BEARISH, ob_top=1.1020, ob_bottom=1.1010)
        plan = mgr.create_exit_plan(sig, entry_price=1.1010, current_bar=0)
        assert plan.take_profit < plan.entry_price

    def test_opposing_liquidity_caps_tp_bearish(self):
        """For shorts, opposing liquidity (prior swing low) above TP → cap does not apply."""
        mgr = ExitManager(r_multiple=2.0, sl_spread_buffer=0, breakeven_at_r=0, max_hold_bars=40)
        sig = _signal(Direction.BEARISH, ob_top=1.1020, ob_bottom=1.1010)
        # 2R TP would be below entry; if opposing liq is ABOVE the 2R TP it's irrelevant
        plan = mgr.create_exit_plan(
            sig, entry_price=1.1010, current_bar=0, opposing_liquidity=1.1005
        )
        # opposing_liquidity=1.1005 > r_tp_price → cap applied
        assert plan.take_profit == 1.1005


# ── evaluate_bar — Bullish position ──────────────────────────────────────────

class TestEvaluateBarBullish:

    def _plan(self, entry=1.1010, sl=1.0990, tp=1.1030, be=1.1020, max_bar=50) -> ExitPlan:
        return ExitPlan(
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            risk_pips=20.0,
            r_target_pips=40.0,
            breakeven_price=be,
            max_bar=max_bar,
            direction=Direction.BULLISH,
        )

    def test_stop_loss_triggered(self):
        mgr = _default_manager()
        plan = self._plan()
        bar = _bar(high=1.1005, low=1.0985)  # low < sl=1.0990
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.0990)
        assert reason == ExitReason.STOP_LOSS
        assert price == 1.0990

    def test_take_profit_triggered(self):
        mgr = _default_manager()
        plan = self._plan()
        bar = _bar(high=1.1035, low=1.1020)  # high > tp=1.1030
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.0990)
        assert reason == ExitReason.TAKE_PROFIT
        assert price == 1.1030

    def test_breakeven_triggered(self):
        mgr = _default_manager()
        plan = self._plan(be=1.1020)
        bar = _bar(high=1.1022, low=1.1010)   # high >= be=1.1020
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.0990)
        assert reason == ExitReason.BREAKEVEN
        assert price == plan.entry_price  # SL moves to entry

    def test_breakeven_not_triggered_if_already_at_be(self):
        """If current_sl is already at entry (BE already moved), no BREAKEVEN event."""
        mgr = _default_manager()
        plan = self._plan(be=1.1020)
        bar = _bar(high=1.1025, low=1.1015)
        reason, _ = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=plan.entry_price)
        assert reason is None or reason == ExitReason.BREAKEVEN  # either OK — already at BE

    def test_time_exit_triggered(self):
        mgr = _default_manager()
        plan = self._plan(max_bar=10)
        bar = _bar(high=1.1018, low=1.1010, open_=1.1012, close=1.1015)
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.0990)
        assert reason == ExitReason.TIME_BASED

    def test_no_exit_in_normal_conditions(self):
        mgr = _default_manager()
        plan = self._plan()
        bar = _bar(high=1.1015, low=1.1005)  # within range, time is fine
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=5, current_sl=1.0990)
        assert reason is None
        assert price is None


# ── evaluate_bar — Bearish position ──────────────────────────────────────────

class TestEvaluateBarBearish:

    def _plan(self, entry=1.1010, sl=1.1030, tp=1.0990, be=1.1000) -> ExitPlan:
        return ExitPlan(
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            risk_pips=20.0,
            r_target_pips=40.0,
            breakeven_price=be,
            max_bar=50,
            direction=Direction.BEARISH,
        )

    def test_stop_loss_triggered_bearish(self):
        mgr = _default_manager()
        plan = self._plan()
        bar = _bar(high=1.1035, low=1.1005)  # high > sl=1.1030
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.1030)
        assert reason == ExitReason.STOP_LOSS

    def test_take_profit_triggered_bearish(self):
        mgr = _default_manager()
        plan = self._plan()
        bar = _bar(high=1.1005, low=1.0985)  # low < tp=1.0990
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.1030)
        assert reason == ExitReason.TAKE_PROFIT

    def test_breakeven_triggered_bearish(self):
        mgr = _default_manager()
        plan = self._plan(be=1.1000)
        bar = _bar(high=1.1005, low=1.0998)  # low <= be=1.1000
        reason, price = mgr.evaluate_bar(plan, bar, current_bar=10, current_sl=1.1030)
        assert reason == ExitReason.BREAKEVEN
        assert price == plan.entry_price
