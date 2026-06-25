"""
Unit tests — Phase 5c: Order Executor (Execution Quality Filters)
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import List

import pytest

from src.execution.order_executor import (
    ExecutionConfig,
    FilterResult,
    OrderExecutor,
    check_latency_filter,
    check_news_blackout,
    check_session_filter,
    check_spread_filter,
)
from src.strategy.signal import Direction, Signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 2, hour, minute, 0, tzinfo=timezone.utc)


def _signal(signal_time: datetime | None = None) -> Signal:
    return Signal(
        symbol="EURUSD",
        direction=Direction.BULLISH,
        entry_price=1.1005,
        ob_top=1.1010,
        ob_bottom=1.1000,
        confirmation_time=signal_time or _utc(9, 0),
        bar_index=10,
        spread_at_signal=2.0,
    )


def _default_executor() -> OrderExecutor:
    cfg = ExecutionConfig(
        session_start_utc=time(7, 0),
        session_end_utc=time(12, 0),
        spread_filter_multiplier=1.5,
        signal_latency_budget_s=1.5,
        news_blackout_minutes=15,
    )
    return OrderExecutor(cfg)


# ── check_spread_filter ───────────────────────────────────────────────────────

class TestSpreadFilter:

    def test_passes_below_ceiling(self):
        result = check_spread_filter(current_spread_pips=2.0, avg_spread_pips=2.0, multiplier=1.5)
        assert result.passed  # 2.0 <= 3.0

    def test_passes_at_ceiling(self):
        result = check_spread_filter(current_spread_pips=3.0, avg_spread_pips=2.0, multiplier=1.5)
        assert result.passed  # 3.0 == ceiling

    def test_fails_above_ceiling(self):
        result = check_spread_filter(current_spread_pips=3.1, avg_spread_pips=2.0, multiplier=1.5)
        assert not result.passed
        assert "Spread filter" in result.reason

    def test_passes_when_no_avg_spread(self):
        """If no avg spread data, filter is skipped."""
        result = check_spread_filter(current_spread_pips=10.0, avg_spread_pips=0.0)
        assert result.passed

    def test_details_populated_on_failure(self):
        result = check_spread_filter(5.0, 2.0, 1.5)
        assert "current" in result.details
        assert "ceiling" in result.details


# ── check_session_filter ──────────────────────────────────────────────────────

class TestSessionFilter:

    def test_passes_within_session(self):
        result = check_session_filter(
            now_utc=_utc(9, 0), session_start=time(7, 0), session_end=time(12, 0)
        )
        assert result.passed

    def test_passes_at_session_start(self):
        result = check_session_filter(_utc(7, 0), time(7, 0), time(12, 0))
        assert result.passed

    def test_passes_at_session_end(self):
        result = check_session_filter(_utc(12, 0), time(7, 0), time(12, 0))
        assert result.passed

    def test_fails_before_session_start(self):
        result = check_session_filter(_utc(6, 59), time(7, 0), time(12, 0))
        assert not result.passed

    def test_fails_after_session_end(self):
        result = check_session_filter(_utc(12, 1), time(7, 0), time(12, 0))
        assert not result.passed

    def test_reason_populated_on_failure(self):
        result = check_session_filter(_utc(5, 0), time(7, 0), time(12, 0))
        assert "Session filter" in result.reason


# ── check_latency_filter ──────────────────────────────────────────────────────

class TestLatencyFilter:

    def test_passes_under_budget(self):
        signal_time = _utc(9, 0)
        now = signal_time + timedelta(seconds=1.0)
        result = check_latency_filter(signal_time, now_utc=now, budget_s=1.5)
        assert result.passed

    def test_passes_at_budget(self):
        signal_time = _utc(9, 0)
        now = signal_time + timedelta(seconds=1.5)
        result = check_latency_filter(signal_time, now_utc=now, budget_s=1.5)
        assert result.passed

    def test_fails_over_budget(self):
        signal_time = _utc(9, 0)
        now = signal_time + timedelta(seconds=2.0)
        result = check_latency_filter(signal_time, now_utc=now, budget_s=1.5)
        assert not result.passed
        assert "Latency" in result.reason

    def test_zero_elapsed_passes(self):
        signal_time = _utc(9, 0)
        result = check_latency_filter(signal_time, now_utc=signal_time, budget_s=1.5)
        assert result.passed

    def test_details_populated(self):
        signal_time = _utc(9, 0)
        now = signal_time + timedelta(seconds=3.0)
        result = check_latency_filter(signal_time, now, budget_s=1.5)
        assert "elapsed_s" in result.details


# ── check_news_blackout ───────────────────────────────────────────────────────

class TestNewsBlackout:

    def test_passes_no_news(self):
        result = check_news_blackout(_utc(9, 0), news_events=[], blackout_minutes=15)
        assert result.passed

    def test_passes_news_far_away(self):
        news = [_utc(11, 0)]  # 2 hours away
        result = check_news_blackout(_utc(9, 0), news_events=news, blackout_minutes=15)
        assert result.passed

    def test_fails_before_news(self):
        news = [_utc(9, 10)]  # 10 min away
        result = check_news_blackout(_utc(9, 0), news_events=news, blackout_minutes=15)
        assert not result.passed
        assert "blackout" in result.reason.lower()

    def test_fails_after_news(self):
        news = [_utc(8, 55)]  # 5 min ago
        result = check_news_blackout(_utc(9, 0), news_events=news, blackout_minutes=15)
        assert not result.passed

    def test_fails_at_news_time(self):
        news = [_utc(9, 0)]
        result = check_news_blackout(_utc(9, 0), news_events=news, blackout_minutes=15)
        assert not result.passed

    def test_multiple_events_checked(self):
        news = [_utc(7, 0), _utc(8, 0), _utc(9, 5)]  # only last one is close
        result = check_news_blackout(_utc(9, 0), news_events=news, blackout_minutes=15)
        assert not result.passed


# ── OrderExecutor.run_filters ─────────────────────────────────────────────────

class TestOrderExecutorRunFilters:

    def test_all_filters_pass(self):
        executor = _default_executor()
        sig = _signal(signal_time=_utc(9, 0))
        result = executor.run_filters(
            signal=sig,
            current_spread_pips=2.0,
            avg_spread_pips=2.0,
            now_utc=_utc(9, 0),
        )
        assert result.passed

    def test_session_filter_blocks_early(self):
        executor = _default_executor()
        sig = _signal(_utc(5, 0))
        result = executor.run_filters(sig, 2.0, 2.0, now_utc=_utc(5, 0))
        assert not result.passed
        assert "Session" in result.reason

    def test_spread_filter_blocks_wide_spread(self):
        executor = _default_executor()
        sig = _signal(_utc(9, 0))
        result = executor.run_filters(
            sig, current_spread_pips=10.0, avg_spread_pips=2.0, now_utc=_utc(9, 0)
        )
        assert not result.passed
        assert "Spread" in result.reason

    def test_latency_filter_blocks_stale_signal(self):
        executor = _default_executor()
        signal_time = _utc(9, 0)
        now = signal_time + timedelta(seconds=5)  # 5s old
        sig = _signal(signal_time)
        result = executor.run_filters(sig, 2.0, 2.0, now_utc=now)
        assert not result.passed
        assert "Latency" in result.reason

    def test_news_blackout_blocks_entry(self):
        executor = _default_executor()
        executor.set_news_events([_utc(9, 5)])  # 5 min away
        sig = _signal(_utc(9, 0))
        result = executor.run_filters(sig, 2.0, 2.0, now_utc=_utc(9, 0))
        assert not result.passed


# ── OrderExecutor.should_retry_requote ────────────────────────────────────────

class TestRequoteRetry:

    def test_first_retry_allowed_with_acceptable_spread(self):
        executor = _default_executor()
        assert executor.should_retry_requote(2.0, 2.0, retry_count=0) is True

    def test_second_retry_blocked(self):
        executor = _default_executor()
        assert executor.should_retry_requote(2.0, 2.0, retry_count=1) is False

    def test_retry_blocked_if_spread_widened(self):
        executor = _default_executor()
        # current=10 > 1.5 * avg=2 → spread filter fails
        assert executor.should_retry_requote(10.0, 2.0, retry_count=0) is False
