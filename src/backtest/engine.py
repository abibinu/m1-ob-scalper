"""
Module 5: Event-Driven Backtesting Engine
==========================================
Per SDD Rev 2, Section 6 & 6.1:

  - Bar-by-bar replay (strict causal order — no lookahead)
  - Same OrderBlockRegister + ExitManager logic used live
  - Fills at bid/ask midpoint + configurable slippage model
  - Vectorized pandas ops allowed ONLY within a single bar's context

Architecture:
    BacktestEngine
      └── StrategyProcessor  (same as live)
      └── ExitManager        (same as live)
      └── BacktestPosition   (simulated open position)
      └── BacktestReport     (collected trade results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.core.logger import get_logger
from src.data.market_data import compute_average_spread
from src.execution.exit_manager import ExitManager, ExitPlan, ExitReason
from src.strategy.order_block import StrategyProcessor
from src.strategy.signal import Direction, Signal

log = get_logger(__name__)


# ── BacktestPosition ──────────────────────────────────────────────────────────

@dataclass
class BacktestPosition:
    """A simulated open position tracked during the replay."""
    signal: Signal
    plan: ExitPlan
    entry_price: float
    entry_bar: int
    current_sl: float    # may be updated by breakeven logic


# ── BacktestTrade ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """
    A completed simulated trade with full metadata.
    ``pnl`` is in R-multiples (positive = win, negative = loss).
    """
    symbol: str
    direction: Direction
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: ExitReason
    risk_pips: float
    pnl_pips: float

    @property
    def pnl_r(self) -> float:
        """P&L expressed as R-multiples."""
        if self.risk_pips <= 0:
            return 0.0
        return self.pnl_pips / self.risk_pips

    @property
    def is_win(self) -> bool:
        return self.pnl_pips > 0


# ── BacktestEngine ────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven, bar-by-bar backtesting engine.

    Usage::

        engine = BacktestEngine(symbol="EURUSD", bars=df, slippage_pips=0.5)
        report = engine.run()
    """

    def __init__(
        self,
        symbol: str,
        bars: pd.DataFrame,
        slippage_pips: float = 0.5,
        pip_size: float = 0.0001,
        r_multiple: float = 2.0,
        sl_spread_buffer: float = 1.5,
        breakeven_at_r: float = 1.0,
        max_hold_bars: int = 40,
        max_ob_age_bars: int = 75,
        max_ob_per_symbol: int = 5,
        ob_stack_tolerance: float = 1.5,
        displacement_threshold: float = 1.5,
        avg_spread_window: int = 20,
        trailing_stop_activation_r: float = 0.0,
        trailing_stop_distance_pips: float = 0.0,
        session_start_utc: str = "00:00",
        session_end_utc: str = "23:59",
        cfg=None,
    ) -> None:
        self.symbol = symbol
        self.bars = bars.reset_index(drop=True)
        self.slippage_pips = slippage_pips
        self.pip_size = pip_size
        self.avg_spread_window = avg_spread_window

        from datetime import datetime
        self.session_start = datetime.strptime(session_start_utc, "%H:%M").time()
        self.session_end = datetime.strptime(session_end_utc, "%H:%M").time()

        self._exit_mgr = ExitManager(
            r_multiple=r_multiple,
            sl_spread_buffer=sl_spread_buffer,
            breakeven_at_r=breakeven_at_r,
            max_hold_bars=max_hold_bars,
            pip_size=pip_size,
            trailing_stop_activation_r=trailing_stop_activation_r,
            trailing_stop_distance_pips=trailing_stop_distance_pips,
        )

        if cfg is not None:
            self._processor = StrategyProcessor(symbol, cfg=cfg)
        else:
            # Build a minimal mock-config-free processor
            from src.strategy.order_block import OrderBlockRegister, DisplacementScanner, StrategyProcessor as SP
            self._processor = SP.__new__(SP)
            self._processor.symbol = symbol
            self._processor._register = OrderBlockRegister(
                symbol, max_ob_age_bars, max_ob_per_symbol, ob_stack_tolerance
            )
            self._processor._scanner = DisplacementScanner(
                displacement_threshold=displacement_threshold
            )
            from collections import deque
            self._processor._window_size = 25
            self._processor._bar_deque = deque(maxlen=25)
            self._processor._pending_candidates = []

        self._open_positions: List[BacktestPosition] = []
        self._completed_trades: List[BacktestTrade] = []

    # ── Main replay loop ──────────────────────────────────────────────────────

    def run(self) -> "BacktestReport":
        """
        Replay all bars sequentially. Returns a BacktestReport.

        Critical invariant: bar N can only see data from bars 0..N.
        No future data is referenced anywhere in this loop.
        """
        n = len(self.bars)
        log.info("Backtest started: %d bars for %s", n, self.symbol)

        for bar_idx in range(n):
            # ── Strict causality: pass ONLY bars[0..bar_idx] to the processor ──
            bar = self.bars.iloc[bar_idx]

            # Compute avg spread from bars seen so far (causal)
            seen = self.bars.iloc[max(0, bar_idx - self.avg_spread_window): bar_idx + 1]
            avg_spread_pips = compute_average_spread(seen, self.avg_spread_window)

            # ── 1. Evaluate open positions first ─────────────────────────────
            self._evaluate_open_positions(bar, bar_idx)

            # ── 2. Run strategy processor ─────────────────────────────────────
            signals = self._processor.process_bar(bar, bar_idx, avg_spread_pips)

            # ── 3. Enter new positions on confirmed signals ───────────────────
            current_time = bar["time"].time()
            if self.session_start <= self.session_end:
                in_session = self.session_start <= current_time <= self.session_end
            else:
                in_session = current_time >= self.session_start or current_time <= self.session_end

            for sig in signals:
                if not in_session:
                    log.debug("Skipping signal at bar %d — outside session", bar_idx)
                    continue
                if self._open_positions:
                    # Simplified: only 1 position at a time per symbol
                    log.debug("Skipping signal at bar %d — position already open", bar_idx)
                    continue
                self._enter_position(sig, bar, bar_idx, avg_spread_pips)

        log.info("Backtest complete: %d trades", len(self._completed_trades))
        return BacktestReport(self._completed_trades, self.bars)

    # ── Position management ───────────────────────────────────────────────────

    def _enter_position(
        self,
        signal: Signal,
        bar: pd.Series,
        bar_idx: int,
        avg_spread_pips: float,
    ) -> None:
        """Simulate entry with slippage."""
        slippage = self.slippage_pips * self.pip_size
        if signal.direction == Direction.BULLISH:
            entry_price = signal.entry_price + slippage   # filled at ask
        else:
            entry_price = signal.entry_price - slippage   # filled at bid

        spread_pips = float(bar.get("spread", 2))
        plan = self._exit_mgr.create_exit_plan(
            signal=signal,
            entry_price=entry_price,
            current_bar=bar_idx,
            current_spread_pips=spread_pips,
        )
        pos = BacktestPosition(
            signal=signal,
            plan=plan,
            entry_price=entry_price,
            entry_bar=bar_idx,
            current_sl=plan.stop_loss,
        )
        self._open_positions.append(pos)
        log.debug("Entered %s at %.5f (bar %d)", signal.direction.name, entry_price, bar_idx)

    def _evaluate_open_positions(self, bar: pd.Series, bar_idx: int) -> None:
        """Check all open positions for exit conditions on this bar."""
        still_open = []
        for pos in self._open_positions:
            reason, exit_price = self._exit_mgr.evaluate_bar(
                pos.plan, bar, bar_idx, pos.current_sl
            )
            if reason == ExitReason.BREAKEVEN:
                # Move SL, keep position open
                pos.current_sl = exit_price  # type: ignore[arg-type]
                still_open.append(pos)
            elif reason is not None and exit_price is not None:
                trade = self._close_position(pos, exit_price, bar_idx, reason)
                self._completed_trades.append(trade)
            else:
                still_open.append(pos)
        self._open_positions = still_open

    def _close_position(
        self,
        pos: BacktestPosition,
        exit_price: float,
        exit_bar: int,
        reason: ExitReason,
    ) -> BacktestTrade:
        """Convert a closed BacktestPosition into a BacktestTrade."""
        if pos.signal.direction == Direction.BULLISH:
            pnl_pips = (exit_price - pos.entry_price) / self.pip_size
        else:
            pnl_pips = (pos.entry_price - exit_price) / self.pip_size

        trade = BacktestTrade(
            symbol=self.symbol,
            direction=pos.signal.direction,
            entry_bar=pos.entry_bar,
            exit_bar=exit_bar,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.plan.stop_loss,
            take_profit=pos.plan.take_profit,
            exit_reason=reason,
            risk_pips=pos.plan.risk_pips,
            pnl_pips=pnl_pips,
        )
        log.debug(
            "Closed %s at %.5f (bar %d) reason=%s pnl_pips=%.1f",
            pos.signal.direction.name, exit_price, exit_bar, reason.name, pnl_pips,
        )
        return trade


# ── BacktestReport ────────────────────────────────────────────────────────────

@dataclass
class BacktestReport:
    """
    Contains all completed trades and computed performance metrics.
    Populated by BacktestEngine.run().
    """
    trades: List[BacktestTrade]
    bars: pd.DataFrame

    # ── Core metrics ─────────────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> List[BacktestTrade]:
        return [t for t in self.trades if t.is_win]

    @property
    def losses(self) -> List[BacktestTrade]:
        return [t for t in self.trades if not t.is_win]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.wins) / len(self.trades)

    @property
    def total_pnl_pips(self) -> float:
        return sum(t.pnl_pips for t in self.trades)

    @property
    def total_pnl_r(self) -> float:
        return sum(t.pnl_r for t in self.trades)

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss. Returns 0 if no trades or no losses."""
        gross_profit = sum(t.pnl_pips for t in self.wins)
        gross_loss = abs(sum(t.pnl_pips for t in self.losses))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def expectancy_r(self) -> float:
        """Average R-multiple per trade."""
        if not self.trades:
            return 0.0
        return self.total_pnl_r / len(self.trades)

    @property
    def max_drawdown_pips(self) -> float:
        """
        Maximum peak-to-trough drawdown in pips (cumulative P&L basis).
        Returns 0 if no trades.
        """
        if not self.trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.pnl_pips
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_consecutive_losses(self) -> int:
        """Longest streak of consecutive losing trades."""
        if not self.trades:
            return 0
        max_streak = 0
        current_streak = 0
        for t in self.trades:
            if not t.is_win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "total_pnl_pips": round(self.total_pnl_pips, 2),
            "total_pnl_r": round(self.total_pnl_r, 4),
            "max_drawdown_pips": round(self.max_drawdown_pips, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
        }
