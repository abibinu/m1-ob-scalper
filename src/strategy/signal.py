"""
Signal dataclass used by the Order Block strategy processor.

A Signal is only produced after BOTH:
  1. Mitigation touch    — price enters the OB zone
  2. Rejection close     — a candle closes BACK outside the OB zone
                          (same or next bar after the touch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class Direction(Enum):
    BULLISH = auto()   # Long signal — order block is below current price
    BEARISH = auto()   # Short signal — order block is above current price


@dataclass(frozen=True)
class Signal:
    """
    A confirmed trading signal produced by the strategy processor.

    Attributes:
        symbol:           Instrument (e.g. "EURUSD").
        direction:        BULLISH or BEARISH.
        entry_price:      Suggested entry (mid of OB zone at touch time).
        ob_top:           Order block zone upper boundary.
        ob_bottom:        Order block zone lower boundary.
        confirmation_time: UTC time of the rejection-close bar.
        bar_index:        Sequential bar number at time of confirmation.
        spread_at_signal: Live spread (in points) at signal time.
    """
    symbol: str
    direction: Direction
    entry_price: float
    ob_top: float
    ob_bottom: float
    confirmation_time: datetime
    bar_index: int
    spread_at_signal: float = 0.0

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def ob_midpoint(self) -> float:
        return (self.ob_top + self.ob_bottom) / 2

    @property
    def ob_height(self) -> float:
        return abs(self.ob_top - self.ob_bottom)
