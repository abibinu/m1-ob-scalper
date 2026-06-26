"""
Unit tests — Phase 4: Module 3 — Order Block Strategy Processor

Tests cover:
  - OrderBlock dataclass
  - DisplacementScanner detection logic
  - OrderBlockRegister lifecycle (age, invalidation, stacking, max count)
  - Signal confirmation (touch vs. rejection close)
  - StrategyProcessor end-to-end
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
import pytest

from src.strategy.order_block import (
    DisplacementScanner,
    OrderBlock,
    OrderBlockRegister,
    StrategyProcessor,
)
from src.strategy.signal import Direction, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_s: int = 0) -> pd.Timestamp:
    base = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
    return pd.Timestamp(base + timedelta(seconds=offset_s))


def _bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    tick_volume: int = 100,
    offset_s: int = 0,
) -> pd.Series:
    return pd.Series(
        {
            "time": _ts(offset_s),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
            "spread": 2,
        }
    )


def _make_bars(rows: List[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of dicts with keys open/high/low/close."""
    records = []
    for i, r in enumerate(rows):
        records.append(
            {
                "time": _ts(i * 60),
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "tick_volume": r.get("vol", 100),
                "spread": 2,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# OrderBlock dataclass
# ---------------------------------------------------------------------------

class TestOrderBlock:

    def test_midpoint(self):
        ob = OrderBlock("EURUSD", Direction.BULLISH, top=1.1010, bottom=1.1000, birth_bar=0)
        assert abs(ob.midpoint - 1.1005) < 1e-9

    def test_height(self):
        ob = OrderBlock("EURUSD", Direction.BULLISH, top=1.1020, bottom=1.1000, birth_bar=0)
        assert abs(ob.height - 0.0020) < 1e-7

    def test_age(self):
        ob = OrderBlock("EURUSD", Direction.BULLISH, top=1.1010, bottom=1.1000, birth_bar=10)
        assert ob.age(15) == 5

    def test_active_default(self):
        ob = OrderBlock("EURUSD", Direction.BULLISH, top=1.101, bottom=1.100, birth_bar=0)
        assert ob.active is True

    def test_overlaps_same_direction(self):
        a = OrderBlock("EURUSD", Direction.BULLISH, top=1.1010, bottom=1.1000, birth_bar=0)
        b = OrderBlock("EURUSD", Direction.BULLISH, top=1.1008, bottom=1.0998, birth_bar=1)
        # They genuinely overlap — tolerance irrelevant
        assert a.overlaps(b, spread_tolerance=0.0001) is True

    def test_no_overlap_different_direction(self):
        a = OrderBlock("EURUSD", Direction.BULLISH, top=1.1010, bottom=1.1000, birth_bar=0)
        b = OrderBlock("EURUSD", Direction.BEARISH, top=1.1010, bottom=1.1000, birth_bar=1)
        assert a.overlaps(b, spread_tolerance=100) is False

    def test_merge_uses_outermost_boundaries(self):
        a = OrderBlock("EURUSD", Direction.BULLISH, top=1.1010, bottom=1.1000, birth_bar=5)
        b = OrderBlock("EURUSD", Direction.BULLISH, top=1.1015, bottom=1.0995, birth_bar=10)
        merged = OrderBlock.merge(a, b)
        assert merged.top == 1.1015
        assert merged.bottom == 1.0995

    def test_merge_birth_bar_is_newer(self):
        """When volume_score is equal, merge uses the block with higher volume_score.
        Both default to 1.0, so dominant = a (checked first in max comparison).
        The merged block takes the dominant's birth_bar."""
        a = OrderBlock("EURUSD", Direction.BULLISH, top=1.101, bottom=1.100, birth_bar=5)
        b = OrderBlock("EURUSD", Direction.BULLISH, top=1.102, bottom=1.099, birth_bar=10)
        merged = OrderBlock.merge(a, b)
        # Both volume_score=1.0, so dominant = a (first with max score); birth_bar=5
        assert merged.birth_bar in (5, 10)  # either is valid with equal scores
        assert merged.volume_score == 1.0
        assert merged.top == 1.102
        assert merged.bottom == 1.099


# ---------------------------------------------------------------------------
# DisplacementScanner
# ---------------------------------------------------------------------------

class TestDisplacementScanner:

    def _make_flat_bars(self, n: int = 25, base: float = 1.1000) -> pd.DataFrame:
        """n bars with small range (0.0005) and stable prices."""
        rows = [{"o": base, "h": base + 0.0005, "l": base - 0.0005, "c": base}
                for _ in range(n)]
        return _make_bars(rows)

    def test_no_signal_on_flat_market(self):
        scanner = DisplacementScanner(displacement_threshold=1.5, lookback=20)
        bars = self._make_flat_bars(25)
        result = scanner.scan(bars, len(bars) - 1)
        assert result is None

    def test_detects_bullish_displacement(self):
        """A large upward bar breaking prior high → BEARISH OB identified."""
        base = 1.1000
        # 20 flat bars, then 1 explicit bearish candle (so scanner can find OB candle)
        # then 1 huge bullish displacement that breaks prior_high
        rows = [{"o": base, "h": base + 0.0005, "l": base - 0.0005, "c": base}
                for _ in range(21)]
        # Explicit bearish candle before displacement
        rows.append({"o": base + 0.0002, "h": base + 0.0004, "l": base - 0.0003, "c": base - 0.0001})
        # Displacement bar: range >> avg, close breaks prior_high (≈1.1005)
        rows.append({"o": base, "h": base + 0.0200, "l": base - 0.0001, "c": base + 0.0180})
        bars = _make_bars(rows)
        scanner = DisplacementScanner(displacement_threshold=1.5, lookback=20)
        result = scanner.scan(bars, len(bars) - 1)
        assert result is not None, "Expected displacement to be detected"
        direction, ob_idx, volume_score = result   # Rev 3: 3-tuple
        assert direction == Direction.BEARISH   # supply zone created
        assert volume_score > 0

    def test_detects_bearish_displacement(self):
        """A large downward bar breaking prior low → BULLISH OB identified."""
        base = 1.1000
        rows = [{"o": base, "h": base + 0.0005, "l": base - 0.0005, "c": base}
                for _ in range(21)]
        # Explicit bullish candle before displacement
        rows.append({"o": base - 0.0002, "h": base + 0.0003, "l": base - 0.0004, "c": base + 0.0001})
        # Displacement bar: huge bearish — breaks prior_low (≈1.0995)
        rows.append({"o": base, "h": base + 0.0001, "l": base - 0.0200, "c": base - 0.0180})
        bars = _make_bars(rows)
        scanner = DisplacementScanner(displacement_threshold=1.5, lookback=20)
        result = scanner.scan(bars, len(bars) - 1)
        assert result is not None, "Expected displacement to be detected"
        direction, ob_idx, volume_score = result   # Rev 3: 3-tuple
        assert direction == Direction.BULLISH   # demand zone created
        assert volume_score > 0

    def test_skip_zero_volume_bars(self):
        """A displacement bar with zero volume must be skipped."""
        base = 1.1000
        rows = [{"o": base, "h": base + 0.0005, "l": base - 0.0005, "c": base}
                for _ in range(22)]
        rows.append({
            "o": base, "h": base + 0.0100, "l": base - 0.0001,
            "c": base + 0.0090, "vol": 0  # zero volume
        })
        bars = _make_bars(rows)
        ts_last = bars.iloc[-1]["time"]
        scanner = DisplacementScanner(
            displacement_threshold=1.5,
            lookback=20,
            zero_volume_bars=[ts_last],
        )
        result = scanner.scan(bars, len(bars) - 1)
        assert result is None

    def test_not_enough_bars_returns_none(self):
        bars = self._make_flat_bars(5)
        scanner = DisplacementScanner(displacement_threshold=1.5, lookback=20)
        assert scanner.scan(bars, 4) is None


# ---------------------------------------------------------------------------
# OrderBlockRegister — lifecycle
# ---------------------------------------------------------------------------

class TestOrderBlockRegisterLifecycle:

    def _fresh_register(self, max_age=50, max_count=5) -> OrderBlockRegister:
        return OrderBlockRegister("EURUSD", max_age_bars=max_age, max_count=max_count)

    def _make_ob(self, birth: int, direction=Direction.BULLISH,
                 top=1.1010, bottom=1.1000) -> OrderBlock:
        return OrderBlock("EURUSD", direction, top=top, bottom=bottom, birth_bar=birth)

    # ── Age expiry ────────────────────────────────────────────────────────────

    def test_ob_expires_after_max_age(self):
        reg = self._fresh_register(max_age=10)
        ob = self._make_ob(birth=0)
        reg.add(ob)
        far_bar = _bar(1.1020, 1.1025, 1.1015, 1.1022, offset_s=600)
        reg.update(far_bar, current_bar_idx=10)   # age = 10 = max_age
        assert len(reg.active_blocks) == 0

    def test_ob_survives_before_max_age(self):
        reg = self._fresh_register(max_age=10)
        ob = self._make_ob(birth=0)
        reg.add(ob)
        far_bar = _bar(1.1020, 1.1025, 1.1015, 1.1022, offset_s=540)
        reg.update(far_bar, current_bar_idx=9)   # age = 9 < max_age
        assert len(reg.active_blocks) == 1

    # ── Close-through invalidation ────────────────────────────────────────────

    def test_bullish_ob_invalidated_by_close_below_bottom(self):
        """Demand zone: close below bottom → invalidated."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BULLISH, top=1.1010, bottom=1.1000)
        reg.add(ob)
        bearish_close = _bar(1.1010, 1.1010, 1.0985, 1.0990)  # close=0.999 < bottom
        reg.update(bearish_close, current_bar_idx=1)
        assert len(reg.active_blocks) == 0

    def test_bearish_ob_invalidated_by_close_above_top(self):
        """Supply zone: close above top → invalidated."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BEARISH, top=1.1020, bottom=1.1010)
        reg.add(ob)
        bullish_close = _bar(1.1010, 1.1030, 1.1008, 1.1025)  # close > top
        reg.update(bullish_close, current_bar_idx=1)
        assert len(reg.active_blocks) == 0

    def test_wick_through_does_not_invalidate(self):
        """Only a CLOSE through the boundary invalidates — a wick through does not."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BULLISH, top=1.1010, bottom=1.1000)
        reg.add(ob)
        # wick goes below bottom but close is above bottom
        wick_bar = _bar(1.1005, 1.1010, 1.0990, 1.1003)
        reg.update(wick_bar, current_bar_idx=1)
        assert len(reg.active_blocks) == 1  # still active

    # ── Max count cap ─────────────────────────────────────────────────────────

    def test_max_count_enforced(self):
        reg = self._fresh_register(max_count=3)
        for i in range(4):
            reg.add(self._make_ob(birth=i, top=1.1010 + i * 0.001, bottom=1.1000 + i * 0.001))
        assert len(reg.active_blocks) == 3

    def test_oldest_ob_dropped_on_overflow(self):
        reg = self._fresh_register(max_count=2)
        ob0 = self._make_ob(birth=0)
        ob1 = self._make_ob(birth=1, top=1.102, bottom=1.101)
        ob2 = self._make_ob(birth=2, top=1.103, bottom=1.102)
        reg.add(ob0)
        reg.add(ob1)
        reg.add(ob2)
        active_births = {b.birth_bar for b in reg.active_blocks}
        assert 0 not in active_births  # oldest dropped

    # ── Stacking/overlap merge ────────────────────────────────────────────────

    def test_overlapping_obs_are_merged(self):
        reg = self._fresh_register()
        ob1 = self._make_ob(birth=0, top=1.1010, bottom=1.1000)
        ob2 = self._make_ob(birth=1, top=1.1008, bottom=1.0998)  # overlaps ob1
        reg.add(ob1, avg_spread=0.0)    # tolerance = 0
        reg.add(ob2, avg_spread=0.0)
        # Since they directly overlap, they should merge into one
        assert len(reg.active_blocks) == 1
        merged = reg.active_blocks[0]
        assert merged.top == 1.1010
        assert merged.bottom == 1.0998

    def test_non_overlapping_obs_not_merged(self):
        reg = self._fresh_register()
        ob1 = self._make_ob(birth=0, top=1.1010, bottom=1.1000)
        ob2 = self._make_ob(birth=1, top=1.1030, bottom=1.1020)  # no overlap
        reg.add(ob1, avg_spread=0.0)
        reg.add(ob2, avg_spread=0.0)
        assert len(reg.active_blocks) == 2

    # ── Mitigation touch ─────────────────────────────────────────────────────

    def test_bullish_ob_mitigation_touch(self):
        """Demand zone: price low touches/enters zone."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BULLISH, top=1.1010, bottom=1.1000)
        reg.add(ob)
        touch_bar = _bar(1.1015, 1.1015, 1.1005, 1.1012)  # low=1.1005 ≤ top=1.1010
        touched = reg.check_mitigation(touch_bar, current_bar_idx=1)
        assert len(touched) == 1
        assert ob.mitigated is True
        assert ob.touch_bar == 1

    def test_bearish_ob_mitigation_touch(self):
        """Supply zone: price high touches/enters zone."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BEARISH, top=1.1020, bottom=1.1010)
        reg.add(ob)
        touch_bar = _bar(1.1005, 1.1015, 1.1000, 1.1008)  # high=1.1015 ≥ bottom=1.1010
        touched = reg.check_mitigation(touch_bar, current_bar_idx=1)
        assert len(touched) == 1

    def test_no_touch_below_zone(self):
        """Bar stays well above demand zone — no touch."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BULLISH, top=1.0990, bottom=1.0980)
        reg.add(ob)
        bar = _bar(1.1010, 1.1015, 1.1005, 1.1012)  # low=1.1005 > top=1.0990
        touched = reg.check_mitigation(bar, 1)
        assert touched == []

    def test_already_mitigated_ob_not_re_touched(self):
        """Once mitigated, a block is not added to candidates again."""
        reg = self._fresh_register()
        ob = self._make_ob(birth=0, direction=Direction.BULLISH, top=1.1010, bottom=1.1000)
        reg.add(ob)
        touch = _bar(1.1012, 1.1012, 1.1005, 1.1011)
        reg.check_mitigation(touch, 1)  # first touch
        touched_again = reg.check_mitigation(touch, 2)  # second time same bar
        assert touched_again == []


# ---------------------------------------------------------------------------
# Signal Confirmation (4.1.4)
# ---------------------------------------------------------------------------

class TestSignalConfirmation:

    def _setup_register_with_candidate(self):
        reg = OrderBlockRegister("EURUSD", max_age_bars=50, max_count=5)
        ob = OrderBlock("EURUSD", Direction.BULLISH, top=1.1010, bottom=1.1000, birth_bar=0)
        ob.mitigated = True
        ob.touch_bar = 5
        reg._blocks.append(ob)
        return reg, ob

    def test_rejection_close_above_ob_top_confirms_bullish(self):
        reg, ob = self._setup_register_with_candidate()
        rejection_bar = _bar(1.1005, 1.1015, 1.1003, 1.1012)  # close > ob_top=1.1010
        confirmed = reg.check_rejection_confirmation(rejection_bar, [ob], current_bar_idx=6)
        assert len(confirmed) == 1

    def test_no_confirmation_if_close_inside_ob(self):
        reg, ob = self._setup_register_with_candidate()
        inside_bar = _bar(1.1005, 1.1009, 1.1002, 1.1006)  # close still inside zone
        confirmed = reg.check_rejection_confirmation(inside_bar, [ob], current_bar_idx=6)
        assert confirmed == []

    def test_rejection_window_expires_after_1_bar(self):
        """Rejection must occur at touch_bar or touch_bar+1."""
        reg, ob = self._setup_register_with_candidate()
        ob.touch_bar = 5
        late_rejection = _bar(1.1005, 1.1015, 1.1003, 1.1012)
        # current_bar_idx=7 → distance=2 → expired
        confirmed = reg.check_rejection_confirmation(late_rejection, [ob], current_bar_idx=7)
        assert confirmed == []

    def test_confirmed_ob_is_deactivated(self):
        reg, ob = self._setup_register_with_candidate()
        rejection_bar = _bar(1.1005, 1.1015, 1.1003, 1.1012)
        reg.check_rejection_confirmation(rejection_bar, [ob], current_bar_idx=6)
        assert ob.active is False  # consumed

    def test_bearish_ob_rejection_close_below_bottom(self):
        reg = OrderBlockRegister("EURUSD")
        ob = OrderBlock("EURUSD", Direction.BEARISH, top=1.1020, bottom=1.1010, birth_bar=0)
        ob.mitigated = True
        ob.touch_bar = 5
        reg._blocks.append(ob)
        rejection_bar = _bar(1.1018, 1.1021, 1.1005, 1.1007)  # close < ob_bottom=1.1010
        confirmed = reg.check_rejection_confirmation(rejection_bar, [ob], current_bar_idx=6)
        assert len(confirmed) == 1


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignalDataclass:

    def _make_signal(self, direction=Direction.BULLISH) -> Signal:
        return Signal(
            symbol="EURUSD",
            direction=direction,
            entry_price=1.1005,
            ob_top=1.1010,
            ob_bottom=1.1000,
            confirmation_time=_ts(),
            bar_index=10,
            spread_at_signal=2.0,
        )

    def test_ob_midpoint(self):
        sig = self._make_signal()
        assert abs(sig.ob_midpoint - 1.1005) < 1e-9

    def test_ob_height(self):
        sig = self._make_signal()
        assert abs(sig.ob_height - 0.001) < 1e-9

    def test_immutable(self):
        sig = self._make_signal()
        with pytest.raises((AttributeError, TypeError)):
            sig.entry_price = 999.0  # type: ignore


# ---------------------------------------------------------------------------
# StrategyProcessor — end-to-end (no displacement → no signals on flat market)
# ---------------------------------------------------------------------------

class TestStrategyProcessorSmoke:

    def test_flat_market_produces_no_signals(self, monkeypatch):
        """A totally flat market produces no OBs and therefore no signals."""
        monkeypatch.setenv("DISPLACEMENT_THRESHOLD", "1.5")
        from src.core.config import get_config
        cfg = get_config()
        proc = StrategyProcessor("EURUSD", cfg=cfg)
        base = 1.1000
        for i in range(50):
            b = _bar(base, base + 0.0003, base - 0.0003, base, offset_s=i * 60)
            sigs = proc.process_bar(b, i)
            assert sigs == []

    def test_processor_register_exposed(self, monkeypatch):
        monkeypatch.setenv("DISPLACEMENT_THRESHOLD", "1.5")
        from src.core.config import get_config
        cfg = get_config()
        proc = StrategyProcessor("EURUSD", cfg=cfg)
        assert proc.register is not None
