"""
Module 5: Event-Driven Backtesting Engine
==========================================
Per SDD Rev 2, Section 6 & 6.1:

  - Bar-by-bar replay (strict causal order — no lookahead)
  - Same OrderBlockRegister + ExitManager logic used live
  - Fills at bid/ask midpoint + configurable slippage model
  - Vectorized pandas ops allowed ONLY within a single bar's context

Rev 3 enhancements:
  - Adaptive ATR-based SL: passes rolling ATR to create_exit_plan()
  - Partial TP simulation: BacktestPosition tracks half_closed state;
    on PARTIAL_TP the position is halved and SL moved to entry (free runner)
  - Signal quality filter: signals below fvg_quality_threshold are skipped
  - Session strength filter: entries skipped in low-volatility hours

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
from src.data.atr_calculator import rolling_atr_pips   # Rev 3
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
    current_sl: float       # may be updated by breakeven/trailing logic
    half_closed: bool = False  # Rev 3: True after partial TP fired
    position_fraction: float = 1.0  # Rev 3: remaining fraction (0.5 after partial TP)


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
        # Rev 3: ATR SL
        use_atr_sl: bool = False,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.0,
        # Rev 3: Partial TP
        partial_tp_enabled: bool = False,
        partial_tp_r: float = 1.0,
        partial_tp_fraction: float = 0.5,
        # Rev 3: Signal quality filter
        fvg_quality_threshold: float = 0.0,
        # Rev 4: Trend filter
        trend_filter_enabled: bool = False,
        trend_ema_period: int = 200,
        cfg=None,
    ) -> None:
        self.symbol = symbol
        self.bars = bars.reset_index(drop=True)
        self.slippage_pips = slippage_pips
        self.pip_size = pip_size
        self.avg_spread_window = avg_spread_window
        self.fvg_quality_threshold = fvg_quality_threshold  # Rev 3
        self.use_atr_sl = use_atr_sl
        self.atr_period = atr_period

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
            use_atr_sl=use_atr_sl,
            atr_sl_multiplier=atr_sl_multiplier,
            partial_tp_enabled=partial_tp_enabled,
            partial_tp_r=partial_tp_r,
            partial_tp_fraction=partial_tp_fraction,
        )

        if cfg is not None:
            self._processor = StrategyProcessor(symbol, cfg=cfg)
        else:
            # Build a minimal mock-config-free processor
            from src.strategy.order_block import OrderBlockRegister, DisplacementScanner, StrategyProcessor as SP
            self._processor = SP.__new__(SP)
            self._processor.symbol = symbol
            self._processor._cfg = type('Cfg', (), {
                'fvg_require_confluence': False,
                'fvg_filter_enabled': True,
                'trend_filter_enabled': trend_filter_enabled,
                'trend_ema_period': trend_ema_period,
            })()
            self._processor._register = OrderBlockRegister(
                symbol, max_ob_age_bars, max_ob_per_symbol, ob_stack_tolerance
            )
            self._processor._scanner = DisplacementScanner(
                displacement_threshold=displacement_threshold
            )
            from src.strategy.order_block import FairValueGapScanner
            self._processor._fvg_scanner = FairValueGapScanner()
            from collections import deque
            self._processor._window_size = 25
            self._processor._bar_deque = deque(maxlen=25)
            self._processor._pending_candidates = []
            
            # Rev 4: Initialize trend filter state for the mock processor
            self._processor._trend_filter_enabled = trend_filter_enabled
            self._processor._ema_period = trend_ema_period
            self._processor._ema_alpha = 2 / (trend_ema_period + 1)
            self._processor._current_ema = None

        # Rev 3: Pre-compute rolling ATR for the full bar series
        if use_atr_sl:
            self._atr_series = rolling_atr_pips(self.bars, period=atr_period, pip_size=pip_size)
        else:
            self._atr_series = None

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

            # Compute avg spread from bars seen so far (causal) and convert MT5 points to pips (1 pip = 10 points)
            seen = self.bars.iloc[max(0, bar_idx - self.avg_spread_window): bar_idx + 1]
            avg_spread_pips = compute_average_spread(seen, self.avg_spread_window) / 10.0

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
                # Rev 3: Signal logging to console
                print(f"[SIGNAL] {sig.direction.name} at bar {bar_idx} (Close: {bar['close']:.5f}, OB: {sig.ob_bottom:.5f}-{sig.ob_top:.5f}, FVG: {sig.fvg_confluence})")
                
                if not in_session:
                    log.debug("Skipping signal at bar %d — outside session", bar_idx)
                    continue
                # Rev 3: quality gate
                if self.fvg_quality_threshold > 0 and sig.quality_score < self.fvg_quality_threshold:
                    log.debug("Skipping signal at bar %d — quality %.2f < %.2f",
                              bar_idx, sig.quality_score, self.fvg_quality_threshold)
                    continue
                if self._open_positions:
                    # Simplified: only 1 position at a time per symbol
                    log.debug("Skipping signal at bar %d — position already open", bar_idx)
                    continue
                # Rev 3: pass rolling ATR to entry
                atr_pips = float(self._atr_series.iloc[bar_idx]) if self._atr_series is not None else 0.0
                self._enter_position(sig, bar, bar_idx, avg_spread_pips, atr_pips=atr_pips)

        log.info("Backtest complete: %d trades", len(self._completed_trades))
        return BacktestReport(self._completed_trades, self.bars)

    # ── Position management ───────────────────────────────────────────────────

    def _enter_position(
        self,
        signal: Signal,
        bar: pd.Series,
        bar_idx: int,
        avg_spread_pips: float,
        atr_pips: float = 0.0,          # Rev 3
    ) -> None:
        """Simulate entry with slippage. Rev 3: passes ATR to exit plan."""
        slippage = self.slippage_pips * self.pip_size
        if signal.direction == Direction.BULLISH:
            entry_price = signal.entry_price + slippage   # filled at limit (ask)
        else:
            entry_price = signal.entry_price - slippage   # filled at limit (bid)

        # Convert MT5 points to pips for current spread
        spread_pips = float(bar.get("spread", 20)) / 10.0
        plan = self._exit_mgr.create_exit_plan(
            signal=signal,
            entry_price=entry_price,
            current_bar=bar_idx,
            current_spread_pips=spread_pips,
            atr_pips=atr_pips,           # Rev 3
        )
        pos = BacktestPosition(
            signal=signal,
            plan=plan,
            entry_price=entry_price,
            entry_bar=bar_idx,
            current_sl=plan.stop_loss,
        )
        self._open_positions.append(pos)
        log.debug("Entered %s at %.5f (bar %d) atr_pips=%.1f",
                  signal.direction.name, entry_price, bar_idx, atr_pips)

    def _evaluate_open_positions(self, bar: pd.Series, bar_idx: int) -> None:
        """Check all open positions for exit conditions on this bar.
        Rev 3: handles PARTIAL_TP by halving position and moving SL to entry.
        """
        still_open = []
        for pos in self._open_positions:
            reason, exit_price = self._exit_mgr.evaluate_bar(
                pos.plan, bar, bar_idx, pos.current_sl
            )
            if reason == ExitReason.BREAKEVEN:
                # Move SL, keep position open
                pos.current_sl = exit_price  # type: ignore[arg-type]
                still_open.append(pos)
            elif reason == ExitReason.PARTIAL_TP and not pos.half_closed:
                # Rev 3: Partial close — record a partial trade, halve position, move SL to entry
                partial_pnl = self._compute_partial_pnl(pos, exit_price, pos.plan.partial_tp_fraction)
                partial_trade = self._build_partial_trade(pos, exit_price, bar_idx, partial_pnl)
                self._completed_trades.append(partial_trade)
                # Free runner: move SL to entry (risk-free)
                pos.current_sl = pos.entry_price
                pos.half_closed = True
                pos.position_fraction = 1.0 - pos.plan.partial_tp_fraction
                still_open.append(pos)
                log.debug("Partial TP: %.1f pips on %.0f%% at bar %d",
                          partial_pnl, pos.plan.partial_tp_fraction * 100, bar_idx)
            elif reason is not None and exit_price is not None:
                trade = self._close_position(pos, exit_price, bar_idx, reason)
                self._completed_trades.append(trade)
            else:
                still_open.append(pos)
        self._open_positions = still_open

    def _compute_partial_pnl(self, pos: BacktestPosition, exit_price: float, fraction: float) -> float:
        """Compute P&L in pips for the partial close fraction."""
        if pos.signal.direction == Direction.BULLISH:
            return (exit_price - pos.entry_price) / self.pip_size * fraction
        else:
            return (pos.entry_price - exit_price) / self.pip_size * fraction

    def _build_partial_trade(
        self,
        pos: BacktestPosition,
        exit_price: float,
        exit_bar: int,
        pnl_pips: float,
    ) -> "BacktestTrade":
        """Build a BacktestTrade record for the partial close."""
        return BacktestTrade(
            symbol=self.symbol,
            direction=pos.signal.direction,
            entry_bar=pos.entry_bar,
            exit_bar=exit_bar,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.plan.stop_loss,
            take_profit=pos.plan.partial_tp_price,
            exit_reason=ExitReason.PARTIAL_TP,
            risk_pips=pos.plan.risk_pips,
            pnl_pips=pnl_pips,
        )

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
