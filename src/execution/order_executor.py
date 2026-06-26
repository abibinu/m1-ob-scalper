"""
Module 4 — Order Executor (Execution Quality Filters)
=====================================================
Applies all execution-quality gates before submitting an order:

  Quality Gate:     Skip if signal quality_score < fvg_quality_threshold (Rev 3)
  Spread Filter:    Skip if live spread > 1.5× 20-bar average
  Session Filter:   Dormant outside London/NY overlap window (UTC)
  Strength Filter:  Skip if hourly volatility score < session_strength_min (Rev 3)
  News Blackout:    No entries within N minutes of high-impact events
  Latency Budget:   Discard signal if >1.5s elapsed since signal generation
  Requote handler:  Re-evaluate spread/price once on requote; never retry twice

SDD Rev 2, Section 5.2 + Rev 3 enhancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import List, Optional

import MetaTrader5 as mt5

from src.core.logger import get_logger
from src.strategy.signal import Direction, Signal

log = get_logger(__name__)


# ── Filter result ─────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """Outcome of the execution quality filter pipeline."""
    passed: bool
    reason: str = ""     # Human-readable rejection reason (empty if passed)
    details: dict = field(default_factory=dict)


# ── Individual filter functions ───────────────────────────────────────────────

def check_signal_quality(
    signal: "Signal",
    quality_threshold: float = 0.0,
) -> FilterResult:
    """
    Rev 3: Reject signals whose quality_score falls below the threshold.

    quality_score combines:
      - volume_score: displacement candle volume vs lookback average
      - FVG confluence bonus (+0.25)

    A quality_threshold of 0.0 disables this filter (all signals pass).

    Args:
        signal:            The confirmed Signal.
        quality_threshold: Minimum quality_score to allow entry [0.0–1.0].
    """
    if quality_threshold <= 0.0:
        return FilterResult(passed=True)
    if signal.quality_score < quality_threshold:
        return FilterResult(
            passed=False,
            reason=(
                f"Signal quality too low: {signal.quality_score:.2f} < "
                f"threshold {quality_threshold:.2f} "
                f"(FVG={'yes' if signal.fvg_confluence else 'no'})"
            ),
            details={"quality_score": signal.quality_score, "threshold": quality_threshold},
        )
    return FilterResult(passed=True)


def check_spread_filter(
    current_spread_pips: float,
    avg_spread_pips: float,
    multiplier: float = 1.5,
) -> FilterResult:
    """
    Reject entry if current spread exceeds multiplier × 20-bar average spread.

    Args:
        current_spread_pips: Live spread in pips.
        avg_spread_pips:     Rolling 20-bar average spread in pips.
        multiplier:          Ceiling multiplier (default 1.5).
    """
    if avg_spread_pips <= 0:
        return FilterResult(passed=True, reason="no avg spread data — skipping filter")

    ceiling = multiplier * avg_spread_pips
    if current_spread_pips > ceiling:
        return FilterResult(
            passed=False,
            reason=f"Spread filter: {current_spread_pips:.2f} > ceiling {ceiling:.2f} pips",
            details={"current": current_spread_pips, "ceiling": ceiling},
        )
    return FilterResult(passed=True)


def check_session_filter(
    now_utc: datetime,
    session_start: time,
    session_end: time,
) -> FilterResult:
    """
    Reject entry outside the configured session window (UTC).

    Args:
        now_utc:       Current UTC time.
        session_start: Session open time (UTC).
        session_end:   Session close time (UTC).
    """
    current_time = now_utc.time().replace(second=0, microsecond=0)
    if session_start <= current_time <= session_end:
        return FilterResult(passed=True)
    return FilterResult(
        passed=False,
        reason=(
            f"Session filter: {current_time} not in "
            f"[{session_start}, {session_end}] UTC"
        ),
        details={"current_time": str(current_time)},
    )


def check_session_strength(
    hour_utc: int,
    strength_score: float,
    min_strength: float = 0.0,
) -> FilterResult:
    """
    Rev 3: Reject entry if the current UTC hour has a low historical volatility score.

    Args:
        hour_utc:      Current UTC hour (0–23).
        strength_score: Pre-computed score from SessionScorer.get_hour_strength().
        min_strength:   Minimum score to allow entry (0.0 = disabled).
    """
    if min_strength <= 0.0:
        return FilterResult(passed=True)
    if strength_score < min_strength:
        return FilterResult(
            passed=False,
            reason=(
                f"Session strength too low: hour {hour_utc:02d}UTC "
                f"score={strength_score:.2f} < min={min_strength:.2f}"
            ),
            details={"hour": hour_utc, "score": strength_score, "min": min_strength},
        )
    return FilterResult(passed=True)


def check_latency_filter(
    signal_time: datetime,
    now_utc: Optional[datetime] = None,
    budget_s: float = 1.5,
) -> FilterResult:
    """
    Discard signal if time since signal generation exceeds budget_s seconds.

    Args:
        signal_time: UTC datetime when the signal was generated.
        now_utc:     Current UTC time (injectable for testing).
        budget_s:    Maximum allowed latency in seconds.
    """
    now = now_utc or datetime.now(timezone.utc)
    elapsed = (now - signal_time).total_seconds()
    if elapsed > budget_s:
        return FilterResult(
            passed=False,
            reason=f"Latency budget: signal is {elapsed:.2f}s old (budget={budget_s}s)",
            details={"elapsed_s": elapsed, "budget_s": budget_s},
        )
    return FilterResult(passed=True)


def check_news_blackout(
    now_utc: datetime,
    news_events: List[datetime],
    blackout_minutes: int = 15,
) -> FilterResult:
    """
    News blackout filter. Rejects entry if ``now_utc`` is within
    ``blackout_minutes`` of any scheduled high-impact event.

    In production, ``news_events`` is populated by ``src.data.news_loader``
    which scrapes investing.com and caches events locally.

    Args:
        now_utc:           Current UTC datetime.
        news_events:       List of UTC datetimes of high-impact events.
        blackout_minutes:  Window in minutes before/after each event.
    """
    from datetime import timedelta
    window = timedelta(minutes=blackout_minutes)
    for event_time in news_events:
        if abs((now_utc - event_time).total_seconds()) <= window.total_seconds():
            return FilterResult(
                passed=False,
                reason=f"News blackout: event at {event_time} within {blackout_minutes}min",
                details={"event_time": str(event_time)},
            )
    return FilterResult(passed=True)


# ── Execution quality gate (pipeline) ────────────────────────────────────────

@dataclass
class ExecutionConfig:
    """All parameters needed to run the execution quality filter pipeline."""
    session_start_utc: time       # e.g. time(7, 0)
    session_end_utc: time         # e.g. time(12, 0)
    spread_filter_multiplier: float = 1.5
    signal_latency_budget_s: float = 1.5
    news_blackout_minutes: int = 15
    # Rev 3
    fvg_quality_threshold: float = 0.0    # 0 = disabled
    session_strength_min: float = 0.0     # 0 = disabled


class OrderExecutor:
    """
    Applies all execution quality filters before an order is submitted.
    Acts as the final firewall between a confirmed signal and the broker.
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self._cfg = config
        self._news_events: List[datetime] = []

    def set_news_events(self, events: List[datetime]) -> None:
        """Provide the current session's high-impact news schedule."""
        self._news_events = events

    def run_filters(
        self,
        signal: Signal,
        current_spread_pips: float,
        avg_spread_pips: float,
        now_utc: Optional[datetime] = None,
        session_strength_score: float = 1.0,
    ) -> FilterResult:
        """
        Run the full quality filter pipeline. Returns on first failure.

        Filters applied in order:
          0. Signal quality score (FVG + volume — Rev 3)
          1. Session window
          2. Session hour strength (Rev 3)
          3. Spread ceiling
          4. Signal latency budget
          5. News blackout
        """
        now = now_utc or datetime.now(timezone.utc)

        # 0. Signal quality (Rev 3)
        quality_result = check_signal_quality(signal, self._cfg.fvg_quality_threshold)
        if not quality_result.passed:
            return quality_result

        # 1. Session
        session_result = check_session_filter(
            now, self._cfg.session_start_utc, self._cfg.session_end_utc
        )
        if not session_result.passed:
            return session_result

        # 2. Session strength (Rev 3)
        if self._cfg.session_strength_min > 0.0:
            strength_result = check_session_strength(
                now.hour, session_strength_score, self._cfg.session_strength_min
            )
            if not strength_result.passed:
                return strength_result

        # 3. Spread
        spread_result = check_spread_filter(
            current_spread_pips,
            avg_spread_pips,
            self._cfg.spread_filter_multiplier,
        )
        if not spread_result.passed:
            return spread_result

        # 4. Latency
        latency_result = check_latency_filter(
            signal.confirmation_time,
            now,
            self._cfg.signal_latency_budget_s,
        )
        if not latency_result.passed:
            return latency_result

        # 5. News blackout
        news_result = check_news_blackout(
            now, self._news_events, self._cfg.news_blackout_minutes
        )
        if not news_result.passed:
            return news_result

        return FilterResult(passed=True)

    def should_retry_requote(
        self,
        current_spread_pips: float,
        avg_spread_pips: float,
        retry_count: int,
        multiplier: float = 1.5,
    ) -> bool:
        """
        On a requote/partial fill response, decide whether to resubmit once.

        Policy:
          - Retry at most ONCE (retry_count must be 0).
          - Only retry if spread is still acceptable.
        """
        if retry_count >= 1:
            log.warning("Requote: already retried once — abandoning order.")
            return False
        spread_ok = check_spread_filter(current_spread_pips, avg_spread_pips, multiplier)
        if not spread_ok.passed:
            log.warning("Requote: spread widened too much — abandoning order.")
            return False
        log.info("Requote: spread acceptable — resubmitting once.")
        return True
