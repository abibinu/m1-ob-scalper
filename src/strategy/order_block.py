"""
Module 3: Algorithmic Strategy Processor — Order Block Detection & Lifecycle
=============================================================================
Implements the complete order block state machine per SDD Rev 2, Sections 4 & 4.1,
extended with Rev 3 enhancements:

Key components:
  OrderBlock          — dataclass representing a single cached zone.
  OrderBlockRegister  — manages the active set of OBs with lifecycle rules.
  DisplacementScanner — detects qualifying displacement moves (volume-weighted).
  FairValueGapScanner — detects M1 3-bar Fair Value Gaps (imbalances).
  StrategyProcessor   — top-level per-bar processor wiring all components.

Lifecycle rules enforced:
  4.1.1 Validity window      — OB expires after max_age_bars
  4.1.2 Invalidation         — close through far boundary, or newer displacement
  4.1.3 Stacking & overlap   — merge OBs within 1.5× avg spread tolerance
  4.1.4 Signal confirmation  — rejection close required (touch alone = candidate)

Rev 3 additions:
  - volume_score on OrderBlock (displacement candle volume vs lookback avg)
  - FairValueGapScanner: M1 3-bar imbalance detection
  - Signal carries fvg_confluence + quality_score for downstream filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from src.core.config import get_config
from src.core.logger import get_logger
from src.strategy.signal import Direction, Signal

log = get_logger(__name__)


# ── OrderBlock dataclass ──────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    """
    A single cached order block zone.

    Attributes:
        symbol:       Instrument.
        direction:    BULLISH (demand zone below price) or BEARISH (supply zone above).
        top:          Upper price boundary.
        bottom:       Lower price boundary.
        birth_bar:    Bar index when the OB was created.
        volume_score: Displacement candle volume as a ratio of lookback average (Rev 3).
                      1.0 = average; >1.0 = above-average institutional footprint.
        mitigated:    True once price has entered the zone (candidate state).
        touch_bar:    Bar index of the most recent mitigation touch.
        active:       False once expired/invalidated.
    """
    symbol: str
    direction: Direction
    top: float
    bottom: float
    birth_bar: int
    volume_score: float = 1.0       # Rev 3: displacement volume ratio
    mitigated: bool = False
    touch_bar: Optional[int] = None
    active: bool = True

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)

    def age(self, current_bar: int) -> int:
        return current_bar - self.birth_bar

    def overlaps(self, other: "OrderBlock", spread_tolerance: float) -> bool:
        """Return True if this OB overlaps/stacks with another within price tolerance."""
        if self.direction != other.direction:
            return False
        gap = max(self.bottom, other.bottom) - min(self.top, other.top)
        return gap < spread_tolerance   # negative gap = actual overlap; within tolerance = merge

    @classmethod
    def merge(cls, a: "OrderBlock", b: "OrderBlock") -> "OrderBlock":
        """Merge two overlapping OBs into a single zone using outermost boundaries.
        Rev 3: the block with the higher volume_score takes precedence for birth_bar
        and dominates the merged score.
        """
        assert a.direction == b.direction
        assert a.symbol == b.symbol
        # Higher volume_score is the stronger institutional block
        dominant = a if a.volume_score >= b.volume_score else b
        return cls(
            symbol=a.symbol,
            direction=a.direction,
            top=max(a.top, b.top),
            bottom=min(a.bottom, b.bottom),
            birth_bar=dominant.birth_bar,
            volume_score=max(a.volume_score, b.volume_score),
        )


# ── FairValueGap dataclass ────────────────────────────────────────────────────

@dataclass
class FairValueGap:
    """
    A 3-bar M1 Fair Value Gap (imbalance zone).

    A bullish FVG:  bar[i-2].high < bar[i].low  (upward gap, demand imbalance)
    A bearish FVG:  bar[i-2].low  > bar[i].high (downward gap, supply imbalance)

    Attributes:
        direction: BULLISH = price gapped up (demand); BEARISH = price gapped down (supply).
        top:       Upper boundary of the gap.
        bottom:    Lower boundary of the gap.
        bar_index: Index of the third bar (i) that completed the FVG.
    """
    direction: Direction
    top: float
    bottom: float
    bar_index: int

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    def overlaps_ob(self, ob: OrderBlock) -> bool:
        """True if this FVG zone overlaps with the given order block zone."""
        if self.direction != ob.direction:
            return False
        # Check if the two zones overlap at all
        return self.top >= ob.bottom and self.bottom <= ob.top


# ── FairValueGapScanner ───────────────────────────────────────────────────────

class FairValueGapScanner:
    """
    Detects M1 3-bar Fair Value Gaps (imbalances) using the standard ICT definition:

    Bullish FVG (demand imbalance, price gapped up):
        bar[i-2].high < bar[i].low
        Gap zone: [bar[i-2].high, bar[i].low]

    Bearish FVG (supply imbalance, price gapped down):
        bar[i-2].low > bar[i].high
        Gap zone: [bar[i].high, bar[i-2].low]

    Used to confirm order blocks — an OB co-located with an FVG is a
    high-confidence institutional zone.
    """

    def scan_recent(
        self,
        bars: pd.DataFrame,
        current_idx: int,
        lookback: int = 10,
    ) -> List[FairValueGap]:
        """
        Scan the last ``lookback`` bars for any active FVG patterns.

        Args:
            bars:        Full bar DataFrame.
            current_idx: Index of the most recent bar.
            lookback:    How many bars back to search for FVGs.

        Returns:
            List of detected FairValueGap objects (may be empty).
        """
        fvgs: List[FairValueGap] = []
        start = max(2, current_idx - lookback)

        for i in range(start, current_idx + 1):
            if i < 2 or i >= len(bars):
                continue
            b0 = bars.iloc[i - 2]   # two bars ago
            b2 = bars.iloc[i]       # current bar

            # Bullish FVG: b0.high < b2.low (gap up — demand imbalance)
            if b0["high"] < b2["low"]:
                fvgs.append(FairValueGap(
                    direction=Direction.BULLISH,
                    top=b2["low"],
                    bottom=b0["high"],
                    bar_index=i,
                ))

            # Bearish FVG: b0.low > b2.high (gap down — supply imbalance)
            elif b0["low"] > b2["high"]:
                fvgs.append(FairValueGap(
                    direction=Direction.BEARISH,
                    top=b0["low"],
                    bottom=b2["high"],
                    bar_index=i,
                ))

        return fvgs

    def check_confluence(
        self,
        ob: OrderBlock,
        bars: pd.DataFrame,
        current_idx: int,
        lookback: int = 20,
    ) -> bool:
        """
        Return True if any recently-detected FVG overlaps with ``ob``.
        Used to mark a signal as having FVG confluence.
        """
        fvgs = self.scan_recent(bars, current_idx, lookback=lookback)
        return any(fvg.overlaps_ob(ob) for fvg in fvgs)


# ── Displacement Scanner ──────────────────────────────────────────────────────

class DisplacementScanner:
    """
    Detects displacement candles: a bar whose range is >= displacement_threshold
    times the average range of the lookback window AND whose close breaks a
    prior structural high/low.

    Displacement direction:
      BULLISH displacement → close breaks above prior high → creates BEARISH OB
                            (the last bearish candle before the upward move)
      BEARISH displacement → close breaks below prior low  → creates BULLISH OB
                            (the last bullish candle before the downward move)

    Rev 3: also returns the volume_score (displacement vol / avg lookback vol)
    so order blocks can be ranked by institutional footprint strength.
    """

    def __init__(
        self,
        displacement_threshold: float = 1.5,
        lookback: int = 20,
        zero_volume_bars: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        self.threshold = displacement_threshold
        self.lookback = lookback
        self.zero_volume_bars: set = set(zero_volume_bars or [])

    def scan(
        self,
        bars: pd.DataFrame,
        current_idx: int,
    ) -> Optional[Tuple[Direction, int, float]]:
        """
        Analyse bar at ``current_idx``.

        Returns (direction, ob_candle_idx, volume_score) if a displacement is detected,
        where ob_candle_idx is the index of the order-block candle
        (last opposite candle immediately before the displacement bar),
        and volume_score is the displacement candle's volume / avg lookback volume.

        Returns None if no displacement.
        """
        if current_idx < self.lookback + 1:
            return None

        bar = bars.iloc[current_idx]

        # Skip zero-volume bars
        if bar["time"] in self.zero_volume_bars:
            return None

        # Compute average range over lookback window (excluding zero-vol)
        window = bars.iloc[current_idx - self.lookback: current_idx]
        valid = window[window["tick_volume"] > 0]
        if len(valid) < 3:
            return None
        avg_range = (valid["high"] - valid["low"]).mean()
        if avg_range <= 0:
            return None

        bar_range = bar["high"] - bar["low"]
        if bar_range < self.threshold * avg_range:
            return None  # not a displacement candle

        # Rev 3: compute volume score (displacement vol vs avg lookback vol)
        avg_volume = valid["tick_volume"].mean()
        volume_score = float(bar["tick_volume"]) / avg_volume if avg_volume > 0 else 1.0

        prior_high = window["high"].max()
        prior_low = window["low"].min()

        # Bullish displacement: close > prior high → supply OB (last bearish before move)
        if bar["close"] > prior_high:
            ob_idx = self._find_last_opposite_candle(bars, current_idx, bullish_move=True)
            if ob_idx is not None:
                return (Direction.BEARISH, ob_idx, volume_score)   # bearish OB (supply zone)

        # Bearish displacement: close < prior low → demand OB (last bullish before move)
        elif bar["close"] < prior_low:
            ob_idx = self._find_last_opposite_candle(bars, current_idx, bullish_move=False)
            if ob_idx is not None:
                return (Direction.BULLISH, ob_idx, volume_score)   # bullish OB (demand zone)

        return None

    @staticmethod
    def _find_last_opposite_candle(
        bars: pd.DataFrame,
        displacement_idx: int,
        bullish_move: bool,
    ) -> Optional[int]:
        """
        Walk backwards from the displacement bar to find the last candle that moved
        OPPOSITE to the displacement direction.
          bullish_move=True  → find last bearish candle (close < open)
          bullish_move=False → find last bullish candle (close > open)
        """
        for i in range(displacement_idx - 1, max(displacement_idx - 10, 0) - 1, -1):
            row = bars.iloc[i]
            if bullish_move and row["close"] < row["open"]:   # bearish candle
                return i
            if not bullish_move and row["close"] > row["open"]:  # bullish candle
                return i
        return None


# ── OrderBlockRegister ────────────────────────────────────────────────────────

class OrderBlockRegister:
    """
    Maintains the active set of order blocks for a single symbol.

    Enforces:
      - Max count per symbol (default 5)
      - Age-based expiry
      - Close-through invalidation
      - Stacking/overlap merge (Rev 3: dominant by volume_score)
    """

    def __init__(
        self,
        symbol: str,
        max_age_bars: int = 75,
        max_count: int = 5,
        stack_tolerance_multiplier: float = 1.5,
    ) -> None:
        self.symbol = symbol
        self.max_age_bars = max_age_bars
        self.max_count = max_count
        self.stack_tolerance_multiplier = stack_tolerance_multiplier
        self._blocks: List[OrderBlock] = []

    @property
    def active_blocks(self) -> List[OrderBlock]:
        return [b for b in self._blocks if b.active]

    def add(self, block: OrderBlock, avg_spread: float = 0.0) -> None:
        """
        Add a new order block. Merges with any overlapping block (keeping the
        highest volume_score), then enforces the max-count cap (oldest dropped).
        """
        tolerance = self.stack_tolerance_multiplier * avg_spread

        # Check for overlap with existing active blocks in same direction
        for existing in self.active_blocks:
            if existing.direction == block.direction and existing.overlaps(block, tolerance):
                existing_idx = self._blocks.index(existing)
                merged = OrderBlock.merge(existing, block)
                self._blocks[existing_idx] = merged
                log.debug("Merged OBs into zone [%.5f, %.5f] for %s (vol_score=%.2f)",
                          merged.bottom, merged.top, self.symbol, merged.volume_score)
                return

        self._blocks.append(block)
        log.debug("Registered new OB: %s %s [%.5f, %.5f] bar=%d vol_score=%.2f",
                  self.symbol, block.direction.name, block.bottom, block.top,
                  block.birth_bar, block.volume_score)

        # Enforce max count: remove oldest if over limit
        active = self.active_blocks
        if len(active) > self.max_count:
            oldest = min(active, key=lambda b: b.birth_bar)
            oldest.active = False
            log.debug("Register full — dropped oldest OB (bar %d)", oldest.birth_bar)

    def update(self, bar: pd.Series, current_bar_idx: int, avg_spread: float = 0.0) -> None:
        """
        Process one incoming bar against all active blocks:
          1. Expire aged blocks.
          2. Invalidate blocks where price closed through the far boundary.
        """
        for block in self.active_blocks:
            # 4.1.1 Age expiry
            if block.age(current_bar_idx) >= self.max_age_bars:
                block.active = False
                log.debug("OB expired at bar %d (age %d)", current_bar_idx,
                          block.age(current_bar_idx))
                continue

            # 4.1.2 Close-through invalidation
            if self._is_closed_through(block, bar):
                block.active = False
                log.debug("OB invalidated: close-through at bar %d", current_bar_idx)

    def check_mitigation(self, bar: pd.Series, current_bar_idx: int) -> List[OrderBlock]:
        """
        Return list of active, non-yet-mitigated blocks that price entered this bar.
        Sets block.mitigated = True and records touch_bar.
        """
        touched = []
        for block in self.active_blocks:
            if block.mitigated:
                continue
            if self._price_entered_zone(block, bar):
                block.mitigated = True
                block.touch_bar = current_bar_idx
                touched.append(block)
                log.debug("OB mitigation touch: %s [%.5f-%.5f] at bar %d",
                          block.direction.name, block.bottom, block.top, current_bar_idx)
        return touched

    def check_rejection_confirmation(
        self,
        bar: pd.Series,
        candidate_blocks: List[OrderBlock],
        current_bar_idx: int,
    ) -> List[OrderBlock]:
        """
        Check mitigated blocks to see if the current bar represents a valid
        rejection close (4.1.4).
        """
        confirmed = []
        for block in candidate_blocks:
            if block.touch_bar is not None and current_bar_idx > block.touch_bar:
                if self._is_rejection_close(block, bar):
                    log.debug("OB confirmation (rejection close): %s at bar %d",
                              block.direction.name, current_bar_idx)
                    confirmed.append(block)
        return confirmed

    def invalidate_direction(self, direction: Direction, from_bar: int) -> None:
        """
        4.1.2: When a stronger, newer displacement occurs in the same direction,
        invalidate all older OBs in that direction.
        """
        for block in self.active_blocks:
            if block.direction == direction and block.birth_bar < from_bar:
                block.active = False
                log.debug("OB superseded by newer displacement: bar %d", block.birth_bar)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _price_entered_zone(block: OrderBlock, bar: pd.Series) -> bool:
        """True if the bar's high/low range touched the OB zone."""
        if block.direction == Direction.BULLISH:
            # Demand zone — price enters from above (low touches/enters the zone)
            return bar["low"] <= block.top
        else:
            # Supply zone — price enters from below (high touches/enters the zone)
            return bar["high"] >= block.bottom

    @staticmethod
    def _is_closed_through(block: OrderBlock, bar: pd.Series) -> bool:
        """True if bar CLOSED through the far boundary (not just a wick)."""
        if block.direction == Direction.BULLISH:
            # Far boundary for demand zone is the bottom
            return bar["close"] < block.bottom
        else:
            # Far boundary for supply zone is the top
            return bar["close"] > block.top

    @staticmethod
    def _is_rejection_close(block: OrderBlock, bar: pd.Series) -> bool:
        """True if bar closed BACK OUTSIDE the OB zone (rejection)."""
        if block.direction == Direction.BULLISH:
            # Price came down into demand zone then closed back above the zone top
            return bar["close"] > block.top
        else:
            # Price came up into supply zone then closed back below the zone bottom
            return bar["close"] < block.bottom


# ── StrategyProcessor ─────────────────────────────────────────────────────────

class StrategyProcessor:
    """
    Top-level per-symbol strategy processor.

    Call ``process_bar(bar, bar_idx, avg_spread)`` on each incoming M1 bar.
    Returns a list of confirmed Signal objects (usually empty, occasionally one).

    Rev 3 enhancements:
      - Integrates FairValueGapScanner for FVG confluence detection
      - Attaches volume_score and fvg_confluence to all emitted Signals
      - Supports optional FVG filter: discard signals with no FVG confluence
        when FVG_REQUIRE_CONFLUENCE=True in config
    """

    def __init__(
        self,
        symbol: str,
        cfg=None,
        zero_volume_bars: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        cfg = cfg or get_config()
        self.symbol = symbol
        self._cfg = cfg
        self._register = OrderBlockRegister(
            symbol=symbol,
            max_age_bars=cfg.max_ob_age_bars,
            max_count=cfg.max_ob_per_symbol,
            stack_tolerance_multiplier=cfg.ob_stack_tolerance,
        )
        self._scanner = DisplacementScanner(
            displacement_threshold=cfg.displacement_threshold,
            zero_volume_bars=zero_volume_bars,
        )
        self._fvg_scanner = FairValueGapScanner()

        # Rolling window — keep only what the scanner needs (lookback + 2 bars)
        from collections import deque
        _lookback = getattr(cfg, "displacement_lookback", 20)
        self._window_size = _lookback + 5
        self._bar_deque: deque = deque(maxlen=self._window_size)

        # Rev 4: Trend Filter State
        self._trend_filter_enabled = getattr(self._cfg, "trend_filter_enabled", False)
        self._ema_period = getattr(self._cfg, "trend_ema_period", 200)
        self._ema_alpha = 2 / (self._ema_period + 1)
        self._current_ema = None

    def process_bar(
        self,
        bar: pd.Series,
        bar_idx: int,
        avg_spread: float = 0.0,
    ) -> List[Signal]:
        """
        Process one M1 bar through the full strategy pipeline.

        Returns list of confirmed Signal objects produced this bar.
        """
        self._bar_deque.append(bar)
        signals: List[Signal] = []

        # ── 0. Update Trend Filter EMA ───────────────────────────────────────
        if self._trend_filter_enabled:
            if self._current_ema is None:
                self._current_ema = bar["close"]
            else:
                self._current_ema = (bar["close"] * self._ema_alpha) + (self._current_ema * (1 - self._ema_alpha))

        # ── 1. Update register (age / invalidation) ──────────────────────────
        self._register.update(bar, bar_idx, avg_spread)

        # ── 2. Check mitigation touches on active OBs ───────────────────────
        self._register.check_mitigation(bar, bar_idx)

        # ── 3. Check for rejection confirmations on mitigated OBs ────────────
        mitigated = [b for b in self._register.active_blocks if b.mitigated]
        if mitigated:
            confirmed = self._register.check_rejection_confirmation(bar, mitigated, bar_idx)
            for block in confirmed:
                sig = self._build_signal(block, bar, bar_idx, avg_spread)
                signals.append(sig)
                block.active = False  # OB consumed — deactivate

        # ── 4. Scan for new displacement → register new OB ───────────────────
        if len(self._bar_deque) > 1:
            # Build DataFrame from rolling window only (O(1) amortised)
            bars_df = pd.DataFrame(list(self._bar_deque)).reset_index(drop=True)
            result = self._scanner.scan(bars_df, len(bars_df) - 1)
            if result is not None:
                direction, ob_idx, volume_score = result
                
                # Rev 4: Trend Filter
                if self._trend_filter_enabled and self._current_ema is not None:
                    if direction == Direction.BULLISH and bar["close"] < self._current_ema:
                        log.debug("Filtered BULLISH OB at bar %d due to price below 200 EMA", bar_idx)
                        return signals
                    if direction == Direction.BEARISH and bar["close"] > self._current_ema:
                        log.debug("Filtered BEARISH OB at bar %d due to price above 200 EMA", bar_idx)
                        return signals

                ob_bar = bars_df.iloc[ob_idx]
                block = OrderBlock(
                    symbol=self.symbol,
                    direction=direction,
                    top=ob_bar["high"],
                    bottom=ob_bar["low"],
                    birth_bar=bar_idx,
                    volume_score=volume_score,
                )
                # Invalidate older OBs in same direction (SDD 4.1.2)
                self._register.invalidate_direction(direction, from_bar=bar_idx)
                self._register.add(block, avg_spread)

        return signals

    @property
    def register(self) -> OrderBlockRegister:
        return self._register

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        block: OrderBlock,
        bar: pd.Series,
        bar_idx: int,
        spread: float,
    ) -> Signal:
        """
        Build a Signal from a confirmed OB. Computes FVG confluence and
        quality score (volume_score boosted by FVG presence).
        """
        # Market entry at the rejection close price
        entry = bar["close"]

        # Rev 3: FVG confluence check (M1 only)
        bars_df = pd.DataFrame(list(self._bar_deque)).reset_index(drop=True)
        fvg_confluence = self._fvg_scanner.check_confluence(
            block, bars_df, len(bars_df) - 1, lookback=20
        )

        # Quality score: normalise volume_score to 0–1, add FVG bonus
        # volume_score is a ratio; cap at 3.0 to avoid outlier dominance
        vol_component = min(block.volume_score / 3.0, 1.0)
        fvg_bonus = 0.25 if fvg_confluence else 0.0
        quality_score = min(vol_component * 0.75 + fvg_bonus, 1.0)

        # Optional: require FVG confluence (controlled by config)
        fvg_required = getattr(self._cfg, "fvg_require_confluence", False)
        if fvg_required and not fvg_confluence:
            log.debug(
                "Signal rejected: FVG confluence required but absent [%.5f-%.5f]",
                block.bottom, block.top,
            )
            # Return a placeholder signal that will be filtered by quality_score == 0
            # We still return it so the backtest can optionally log it; the
            # BacktestEngine/live bot should check quality_score > 0 before entering.
            quality_score = 0.0

        return Signal(
            symbol=block.symbol,
            direction=block.direction,
            entry_price=entry,
            ob_top=block.top,
            ob_bottom=block.bottom,
            confirmation_time=bar["time"],
            bar_index=bar_idx,
            spread_at_signal=spread,
            fvg_confluence=fvg_confluence,
            quality_score=quality_score,
        )
