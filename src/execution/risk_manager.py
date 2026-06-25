"""
Module 4 — Risk Manager
========================
Handles:
  - Mathematical lot sizing (0.25% or 0.5% risk per trade)
  - Daily loss tracking and 2% hard-ceiling enforcement
  - Account equity polling wrapper

SDD Rev 2, Section 5:
  "Mathematical Lot Sizer: Computes the specific volume parameters automatically.
   Total financial risk is restricted to exactly 0.25% or 0.5% of net equity per transaction."
  "Daily Loss Interceptor: Keeps track of net daily closed metrics. If daily loss
   parameters approach or breach the 2% hard risk ceiling, all active order pipelines
   are immediately shut down."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.core.logger import get_logger

log = get_logger(__name__)

# ── Symbol tick info ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SymbolInfo:
    """Minimal symbol metadata needed for lot-size calculation."""
    symbol: str
    contract_size: float      # e.g. 100_000 for a standard FX lot
    tick_size: float          # e.g. 0.00001 for EURUSD (5-decimal)
    tick_value: float         # value of one tick per lot in account currency
    volume_min: float         # minimum lot size
    volume_max: float         # maximum lot size
    volume_step: float        # lot size increment


# ── Lot sizing ────────────────────────────────────────────────────────────────

def calculate_lot_size(
    equity: float,
    risk_pct: float,
    sl_pips: float,
    symbol_info: SymbolInfo,
) -> float:
    """
    Compute the lot size that risks exactly ``risk_pct`` % of ``equity``
    given a stop-loss distance of ``sl_pips`` pips.

    Formula:
        risk_amount  = equity × (risk_pct / 100)
        pip_value    = tick_value / tick_size          [value per pip per lot]
        raw_lots     = risk_amount / (sl_pips × pip_value)
        lots         = clamp(round_down_to_step(raw_lots), volume_min, volume_max)

    Args:
        equity:       Account equity in base currency.
        risk_pct:     Risk percentage per trade (e.g. 0.5 for 0.5%).
        sl_pips:      Stop-loss distance in pips (e.g. 10.0).
        symbol_info:  Symbol metadata.

    Returns:
        Computed lot size, rounded down to the nearest volume_step,
        clamped between volume_min and volume_max.

    Raises:
        ValueError: on non-positive inputs or impossibly tight SL.
    """
    if equity <= 0:
        raise ValueError(f"equity must be > 0, got {equity}")
    if risk_pct <= 0 or risk_pct > 100:
        raise ValueError(f"risk_pct must be in (0, 100], got {risk_pct}")
    if sl_pips <= 0:
        raise ValueError(f"sl_pips must be > 0, got {sl_pips}")
    if symbol_info.tick_size <= 0:
        raise ValueError("tick_size must be > 0")

    risk_amount = equity * (risk_pct / 100.0)
    pip_value_per_lot = symbol_info.tick_value / symbol_info.tick_size
    raw_lots = risk_amount / (sl_pips * pip_value_per_lot)

    # Round down to nearest volume_step
    step = symbol_info.volume_step
    lots = (raw_lots // step) * step

    # Clamp to broker limits
    lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

    log.debug(
        "Lot size: equity=%.2f risk=%.2f%% sl=%.1f pips → raw=%.4f → lots=%.2f",
        equity, risk_pct, sl_pips, raw_lots, lots,
    )
    return round(lots, 8)   # avoid floating-point display noise


# ── Daily loss tracker ────────────────────────────────────────────────────────

class DailyLossTracker:
    """
    Tracks cumulative realised P&L for the current UTC trading day.
    Automatically resets at midnight UTC.

    The 2% hard ceiling (``max_daily_loss_pct``) is configurable.
    Once breached, ``is_halted`` returns True and must not be cleared
    automatically — requires manual restart.
    """

    def __init__(self, starting_equity: float, max_daily_loss_pct: float = 2.0) -> None:
        if starting_equity <= 0:
            raise ValueError("starting_equity must be > 0")
        self._starting_equity = starting_equity
        self._max_loss_pct = max_daily_loss_pct
        self._daily_pnl: float = 0.0
        self._trade_day: date = datetime.now(timezone.utc).date()
        self._halted: bool = False

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def loss_pct(self) -> float:
        """Current daily loss as a % of starting equity (negative = loss)."""
        return (self._daily_pnl / self._starting_equity) * 100.0

    @property
    def remaining_risk(self) -> float:
        """
        Remaining dollar risk budget before the daily ceiling is breached.
        A negative value means the ceiling is already exceeded.
        """
        max_loss = self._starting_equity * (self._max_loss_pct / 100.0)
        return max_loss + self._daily_pnl  # pnl is negative for losses

    def record_trade(self, pnl: float, *, _now: Optional[datetime] = None) -> None:
        """
        Record a closed trade P&L (positive = profit, negative = loss).
        Automatically resets daily counter at UTC midnight.
        Triggers halt if the daily ceiling is breached.
        """
        now = _now or datetime.now(timezone.utc)
        today = now.date()

        if today != self._trade_day:
            self._reset(now)

        self._daily_pnl += pnl

        max_loss = self._starting_equity * (self._max_loss_pct / 100.0)
        if self._daily_pnl <= -max_loss and not self._halted:
            self._halted = True
            log.critical(
                "DAILY LOSS CEILING BREACHED: daily_pnl=%.2f (%.2f%% of equity=%.2f). "
                "All pipelines HALTED.",
                self._daily_pnl,
                abs(self.loss_pct),
                self._starting_equity,
            )

    def update_equity(self, new_equity: float) -> None:
        """Update the reference equity (call at session start or after resets)."""
        if new_equity <= 0:
            raise ValueError("equity must be > 0")
        self._starting_equity = new_equity

    # ── private ───────────────────────────────────────────────────────────────

    def _reset(self, now: datetime) -> None:
        log.info("Daily reset: previous P&L=%.2f, new day=%s", self._daily_pnl, now.date())
        self._daily_pnl = 0.0
        self._trade_day = now.date()
        # Note: halt state is NOT cleared on reset — requires manual restart
