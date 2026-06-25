"""
Unit tests — Phase 5a: Risk Manager
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.risk_manager import (
    DailyLossTracker,
    SymbolInfo,
    calculate_lot_size,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

EURUSD = SymbolInfo(
    symbol="EURUSD",
    contract_size=100_000,
    tick_size=0.00001,
    tick_value=1.0,       # $1 per pip per standard lot
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
)


# ── calculate_lot_size ────────────────────────────────────────────────────────

class TestCalculateLotSize:

    def test_basic_calculation(self):
        """$10,000 account, 0.5% risk, 10-pip SL → 0.5 lots."""
        equity = 10_000.0
        risk_pct = 0.5      # $50 risk
        sl_pips = 10.0
        # pip_value_per_lot = tick_value / tick_size = 1.0 / 0.00001 = 100,000
        # But that's in USD per lot per pip (for EURUSD with $1 per 0.00001)
        # Actually: pip_value = tick_value / tick_size = 1.0/0.00001 = 100_000 $/lot per pip
        # raw_lots = 50 / (10 * 100_000) = 0.00005 → clamped to 0.01 (volume_min)
        # The result should be clamped to minimum
        lots = calculate_lot_size(equity, risk_pct, sl_pips, EURUSD)
        assert lots >= EURUSD.volume_min

    def test_risk_scales_with_equity(self):
        """Larger equity → larger lot size, using a symbol with realistic pip values."""
        # Use a symbol where pip_value = tick_value/tick_size = 10/0.0001 = $100,000/pip/lot
        # At 10 pips SL, pip_risk = 10 * 100,000 = $1,000,000/lot
        # For $10k equity at 0.5% risk = $50 → 0.00005 lots (clamped to 0.01)
        # So use a very small pip_value to stay above minimum
        low_pip_sym = SymbolInfo(
            symbol="TEST",
            contract_size=1_000,
            tick_size=0.01,
            tick_value=0.01,      # pip_value = 0.01/0.01 = 1 $/pip/lot
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        # $10k * 0.5% = $50 risk, 10-pip SL: raw = 50/(10*1) = 5.0 lots
        # $100k * 0.5% = $500 risk, 10-pip SL: raw = 500/(10*1) = 50.0 lots
        small = calculate_lot_size(10_000, 0.5, 10, low_pip_sym)
        large = calculate_lot_size(100_000, 0.5, 10, low_pip_sym)
        assert large > small

    def test_wider_sl_reduces_lot_size(self):
        """Wider stop-loss → fewer lots (same risk amount)."""
        tight = calculate_lot_size(10_000, 0.5, 5, EURUSD)
        wide = calculate_lot_size(10_000, 0.5, 20, EURUSD)
        assert tight >= wide

    def test_result_multiple_of_volume_step(self):
        """Lot size must be a multiple of volume_step."""
        lots = calculate_lot_size(50_000, 0.5, 15, EURUSD)
        step = EURUSD.volume_step
        remainder = round(lots % step, 8)
        assert remainder == 0.0 or abs(remainder - step) < 1e-9

    def test_result_within_broker_limits(self):
        lots = calculate_lot_size(10_000, 0.5, 10, EURUSD)
        assert EURUSD.volume_min <= lots <= EURUSD.volume_max

    def test_raises_on_zero_equity(self):
        with pytest.raises(ValueError, match="equity"):
            calculate_lot_size(0, 0.5, 10, EURUSD)

    def test_raises_on_negative_equity(self):
        with pytest.raises(ValueError, match="equity"):
            calculate_lot_size(-1000, 0.5, 10, EURUSD)

    def test_raises_on_zero_sl(self):
        with pytest.raises(ValueError, match="sl_pips"):
            calculate_lot_size(10_000, 0.5, 0, EURUSD)

    def test_raises_on_invalid_risk_pct(self):
        with pytest.raises(ValueError, match="risk_pct"):
            calculate_lot_size(10_000, 0, 10, EURUSD)

    def test_raises_on_risk_pct_over_100(self):
        with pytest.raises(ValueError, match="risk_pct"):
            calculate_lot_size(10_000, 101, 10, EURUSD)

    def test_large_sl_uses_custom_pip_value(self):
        """With higher tick_value, pip_value is proportionally larger."""
        high_value_sym = SymbolInfo(
            symbol="USDJPY",
            contract_size=100_000,
            tick_size=0.001,
            tick_value=0.01,      # per tick per lot
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        lots = calculate_lot_size(10_000, 0.5, 10, high_value_sym)
        assert lots >= high_value_sym.volume_min


# ── DailyLossTracker ──────────────────────────────────────────────────────────

class TestDailyLossTracker:

    def _now(self, **kwargs) -> datetime:
        return datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc).replace(**kwargs)

    def test_initial_pnl_is_zero(self):
        tracker = DailyLossTracker(starting_equity=10_000, max_daily_loss_pct=2.0)
        assert tracker.daily_pnl == 0.0

    def test_initial_not_halted(self):
        tracker = DailyLossTracker(10_000)
        assert tracker.is_halted is False

    def test_record_profit(self):
        tracker = DailyLossTracker(10_000)
        tracker.record_trade(50.0, _now=self._now())
        assert tracker.daily_pnl == 50.0
        assert not tracker.is_halted

    def test_record_loss(self):
        tracker = DailyLossTracker(10_000)
        tracker.record_trade(-50.0, _now=self._now())
        assert tracker.daily_pnl == -50.0

    def test_ceiling_not_breached_below_2pct(self):
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        tracker.record_trade(-150.0, _now=self._now())  # 1.5% loss
        assert not tracker.is_halted

    def test_ceiling_breached_at_2pct(self):
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        tracker.record_trade(-200.0, _now=self._now())  # exactly 2%
        assert tracker.is_halted

    def test_ceiling_breached_beyond_2pct(self):
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        tracker.record_trade(-250.0, _now=self._now())  # 2.5%
        assert tracker.is_halted

    def test_halt_persists_after_subsequent_profit(self):
        """Halt is a hard stop — a winning trade cannot clear it."""
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        tracker.record_trade(-200.0, _now=self._now())  # breach
        tracker.record_trade(500.0, _now=self._now())   # profit after
        assert tracker.is_halted

    def test_multiple_losses_accumulate(self):
        tracker = DailyLossTracker(10_000)
        tracker.record_trade(-80.0, _now=self._now())
        tracker.record_trade(-80.0, _now=self._now())
        assert tracker.daily_pnl == -160.0

    def test_ceiling_triggered_across_trades(self):
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        tracker.record_trade(-100.0, _now=self._now())  # 1%
        tracker.record_trade(-100.0, _now=self._now())  # 2% total → breach
        assert tracker.is_halted

    def test_daily_reset_at_midnight(self):
        tracker = DailyLossTracker(10_000)
        day1 = datetime(2024, 1, 2, 23, 59, tzinfo=timezone.utc)
        day2 = datetime(2024, 1, 3, 0, 1, tzinfo=timezone.utc)
        tracker.record_trade(-100.0, _now=day1)
        assert tracker.daily_pnl == -100.0
        tracker.record_trade(-10.0, _now=day2)  # next day → reset before recording
        assert tracker.daily_pnl == -10.0  # only the new day's trade

    def test_halt_not_cleared_on_reset(self):
        """Daily reset does NOT clear a halt — requires manual restart."""
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        day1 = datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc)
        day2 = datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc)
        tracker.record_trade(-200.0, _now=day1)  # breach
        assert tracker.is_halted
        tracker.record_trade(-10.0, _now=day2)   # new day, still halted
        assert tracker.is_halted

    def test_loss_pct_property(self):
        tracker = DailyLossTracker(10_000)
        tracker.record_trade(-100.0, _now=self._now())
        assert abs(tracker.loss_pct - (-1.0)) < 1e-9

    def test_remaining_risk_property(self):
        tracker = DailyLossTracker(10_000, max_daily_loss_pct=2.0)
        # $200 ceiling, $0 spent → $200 remaining
        assert abs(tracker.remaining_risk - 200.0) < 1e-9
        tracker.record_trade(-50.0, _now=self._now())
        assert abs(tracker.remaining_risk - 150.0) < 1e-9

    def test_raises_on_zero_equity(self):
        with pytest.raises(ValueError):
            DailyLossTracker(0)
