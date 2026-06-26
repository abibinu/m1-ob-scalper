"""
Signal dataclass used by the Order Block strategy processor.

A Signal is only produced after BOTH:
  1. Mitigation touch    — price enters the OB zone
  2. Rejection close     — a candle closes BACK outside the OB zone
                          (same or next bar after the touch)

Enhancement fields (Rev 3):
  fvg_confluence  — True if a Fair Value Gap is co-located with the OB zone
  quality_score   — Composite 0.0–1.0 signal quality (volume + FVG confluence)
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
        fvg_confluence:   True if a Fair Value Gap overlaps this OB zone (M1).
        quality_score:    Composite signal quality score 0.0–1.0.
                          Combines volume_score (displacement strength) and FVG bonus.
    """
    symbol: str
    direction: Direction
    entry_price: float
    ob_top: float
    ob_bottom: float
    confirmation_time: datetime
    bar_index: int
    spread_at_signal: float = 0.0
    fvg_confluence: bool = False
    quality_score: float = 0.5      # default mid-quality

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def ob_midpoint(self) -> float:
        return (self.ob_top + self.ob_bottom) / 2

    @property
    def ob_height(self) -> float:
        return abs(self.ob_top - self.ob_bottom)
