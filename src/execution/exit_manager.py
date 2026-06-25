"""
Module 4 — Exit Manager
========================
Handles all exit logic per SDD Rev 2, Section 5.1:

  5.1.1 Stop-Loss Placement  : far OB boundary + 1-2x spread buffer
  5.1.2 Take-Profit Placement: fixed R-multiple (1.5R-2R) OR nearest opposing liquidity
  5.1.3 Trade Management     : optional breakeven move at 1R in favour
  5.1.4 Time-Based Exit      : close at market after max_hold_bars M1 bars

An ``ExitPlan`` dataclass captures the full exit parameters for a trade.
A separate ``ExitMonitor`` tracks open positions against their plan and
emits exit actions when conditions are met.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import pandas as pd

from src.core.logger import get_logger
from src.strategy.signal import Direction, Signal

log = get_logger(__name__)


# ── Exit action enum ──────────────────────────────────────────────────────────

class ExitReason(Enum):
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TIME_BASED = auto()
    BREAKEVEN = auto()    # SL moved to breakeven (position not closed)


# ── ExitPlan ──────────────────────────────────────────────────────────────────

@dataclass
class ExitPlan:
    """
    Computed exit parameters for a single trade.

    Attributes:
        entry_price:      Fill price (may differ from signal entry_price by spread).
        stop_loss:        Absolute SL price.
        take_profit:      Absolute TP price.
        risk_pips:        SL distance in pips (|entry - sl|).
        r_target_pips:    TP distance in R-pips.
        breakeven_price:  Price at which SL should be moved to entry (0 = disabled).
        max_bar:          Bar index at which the time-based exit fires.
        direction:        BULLISH (long) or BEARISH (short).
    """
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pips: float
    r_target_pips: float
    breakeven_price: float      # 0.0 = breakeven disabled
    trailing_activation_price: float # 0.0 = trailing stop disabled
    trailing_distance_pips: float
    max_bar: int
    direction: Direction


# ── ExitManager ───────────────────────────────────────────────────────────────

class ExitManager:
    """
    Computes exit parameters from a signal and monitors open positions
    for exit conditions on each incoming bar.
    """

    def __init__(
        self,
        r_multiple: float = 2.0,
        sl_spread_buffer: float = 1.5,
        breakeven_at_r: float = 1.0,
        max_hold_bars: int = 40,
        pip_size: float = 0.0001,   # 1 pip = 0.0001 for most FX pairs
        trailing_stop_activation_r: float = 0.0,
        trailing_stop_distance_pips: float = 0.0,
    ) -> None:
        self.r_multiple = r_multiple
        self.sl_spread_buffer = sl_spread_buffer
        self.breakeven_at_r = breakeven_at_r     # 0 = disabled
        self.max_hold_bars = max_hold_bars
        self.pip_size = pip_size
        self.trailing_stop_activation_r = trailing_stop_activation_r
        self.trailing_stop_distance_pips = trailing_stop_distance_pips

    # ── Plan creation ─────────────────────────────────────────────────────────

    def create_exit_plan(
        self,
        signal: Signal,
        entry_price: float,
        current_bar: int,
        current_spread_pips: float = 0.0,
        opposing_liquidity: Optional[float] = None,
    ) -> ExitPlan:
        """
        Build a complete ExitPlan for a just-entered trade.

        Args:
            signal:              The confirmed signal.
            entry_price:         Actual fill price.
            current_bar:         Bar index at entry.
            current_spread_pips: Live spread in pips (for SL buffer).
            opposing_liquidity:  Nearest opposing swing high/low (optional TP cap).

        Returns:
            ExitPlan with SL, TP, breakeven trigger, and time limit.
        """
        direction = signal.direction
        sl_buffer = self.sl_spread_buffer * current_spread_pips * self.pip_size

        if direction == Direction.BULLISH:
            # SL: below ob_bottom minus buffer
            stop_loss = signal.ob_bottom - sl_buffer
            risk_pips = (entry_price - stop_loss) / self.pip_size
            r_tp_price = entry_price + (risk_pips * self.r_multiple * self.pip_size)
            # Cap TP at opposing liquidity if it's closer
            if opposing_liquidity and opposing_liquidity < r_tp_price:
                take_profit = opposing_liquidity
            else:
                take_profit = r_tp_price
            # Breakeven trigger: entry + breakeven_at_r × risk_pips
            if self.breakeven_at_r > 0:
                breakeven_price = entry_price + (self.breakeven_at_r * risk_pips * self.pip_size)
            else:
                breakeven_price = 0.0
            # Trailing stop activation
            if self.trailing_stop_activation_r > 0:
                trailing_activation_price = entry_price + (self.trailing_stop_activation_r * risk_pips * self.pip_size)
            else:
                trailing_activation_price = 0.0

        else:  # BEARISH
            stop_loss = signal.ob_top + sl_buffer
            risk_pips = (stop_loss - entry_price) / self.pip_size
            r_tp_price = entry_price - (risk_pips * self.r_multiple * self.pip_size)
            if opposing_liquidity and opposing_liquidity > r_tp_price:
                take_profit = opposing_liquidity
            else:
                take_profit = r_tp_price
            if self.breakeven_at_r > 0:
                breakeven_price = entry_price - (self.breakeven_at_r * risk_pips * self.pip_size)
            else:
                breakeven_price = 0.0
            # Trailing stop activation
            if self.trailing_stop_activation_r > 0:
                trailing_activation_price = entry_price - (self.trailing_stop_activation_r * risk_pips * self.pip_size)
            else:
                trailing_activation_price = 0.0

        plan = ExitPlan(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pips=abs(risk_pips),
            r_target_pips=abs(risk_pips) * self.r_multiple,
            breakeven_price=breakeven_price,
            trailing_activation_price=trailing_activation_price,
            trailing_distance_pips=self.trailing_stop_distance_pips,
            max_bar=current_bar + self.max_hold_bars,
            direction=direction,
        )
        log.debug(
            "ExitPlan: entry=%.5f SL=%.5f TP=%.5f BE=%.5f max_bar=%d",
            plan.entry_price, plan.stop_loss, plan.take_profit,
            plan.breakeven_price, plan.max_bar,
        )
        return plan

    # ── Per-bar monitoring ────────────────────────────────────────────────────

    def evaluate_bar(
        self,
        plan: ExitPlan,
        bar: pd.Series,
        current_bar: int,
        current_sl: float,          # may differ from plan.stop_loss if BE was triggered
    ) -> Tuple[Optional[ExitReason], Optional[float]]:
        """
        Check a single bar against an ExitPlan.

        Returns:
            (ExitReason, exit_price) if an exit condition is triggered.
            (None, None) if the position should remain open.
            (ExitReason.BREAKEVEN, new_sl_price) for a SL adjustment (no exit).
        """
        high = bar["high"]
        low = bar["low"]

        if plan.direction == Direction.BULLISH:
            # Stop hit?
            if low <= current_sl:
                return ExitReason.STOP_LOSS, current_sl
            # TP hit?
            if high >= plan.take_profit:
                return ExitReason.TAKE_PROFIT, plan.take_profit
            # Trailing stop logic
            if plan.trailing_activation_price > 0 and high >= plan.trailing_activation_price:
                new_sl = high - (plan.trailing_distance_pips * self.pip_size)
                if new_sl > current_sl:
                    return ExitReason.BREAKEVEN, new_sl
            # Breakeven trigger?
            if plan.breakeven_price > 0 and high >= plan.breakeven_price:
                if current_sl < plan.entry_price:  # not yet at BE
                    return ExitReason.BREAKEVEN, plan.entry_price

        else:  # BEARISH
            if high >= current_sl:
                return ExitReason.STOP_LOSS, current_sl
            if low <= plan.take_profit:
                return ExitReason.TAKE_PROFIT, plan.take_profit
            # Trailing stop logic
            if plan.trailing_activation_price > 0 and low <= plan.trailing_activation_price:
                new_sl = low + (plan.trailing_distance_pips * self.pip_size)
                if new_sl < current_sl:
                    return ExitReason.BREAKEVEN, new_sl
            # Breakeven trigger?
            if plan.breakeven_price > 0 and low <= plan.breakeven_price:
                if current_sl > plan.entry_price:
                    return ExitReason.BREAKEVEN, plan.entry_price

        # Time-based exit
        if current_bar >= plan.max_bar:
            mid = (bar["open"] + bar["close"]) / 2
            log.info("Time-based exit at bar %d, mid=%.5f", current_bar, mid)
            return ExitReason.TIME_BASED, mid

        return None, None
