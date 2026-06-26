"""
Live Trading Entry Point
========================
Orchestrates all modules in the correct order:

  1. Load config
  2. Connect to MT5 (with reconnect policy)
  3. Initialise strategy processor(s), risk manager, exit manager, executor
  4. Poll for new M1 bars in a tight loop
  5. Process each bar through the strategy pipeline
  6. Submit orders that pass execution quality filters
  7. Monitor open positions for exit conditions
  8. Fail safe on any unhandled exception (flatten + halt)

Run:
    python -m src.main

Stop gracefully with Ctrl+C or SIGTERM.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime, time as dt_time, timezone
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd

from src.connection.mt5_client import MT5Client
from src.core.config import get_config
from src.core.logger import get_logger
from src.data.market_data import compute_average_spread, fetch_bars
from src.data.atr_calculator import compute_atr_pips          # Rev 3
from src.data.news_loader import refresh_news_if_stale         # Rev 3
from src.execution.exit_manager import ExitManager, ExitPlan, ExitReason
from src.execution.order_executor import ExecutionConfig, OrderExecutor
from src.execution.risk_manager import DailyLossTracker, SymbolInfo, calculate_lot_size
from src.execution.session_scorer import SessionScorer         # Rev 3
from src.strategy.order_block import StrategyProcessor
from src.strategy.signal import Direction, Signal

log = get_logger(__name__)


# ── Symbol pip sizes (extend as needed) ──────────────────────────────────────
SYMBOL_PIP_SIZE: Dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "AUDUSD": 0.0001,
}


def _get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Fetch symbol metadata from MT5 and wrap in a SymbolInfo dataclass."""
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error("Symbol info not available for %s", symbol)
        return None
    return SymbolInfo(
        symbol=symbol,
        contract_size=info.trade_contract_size,
        tick_size=info.point,
        tick_value=info.trade_tick_value,
        volume_min=info.volume_min,
        volume_max=info.volume_max,
        volume_step=info.volume_step,
    )


def _flatten_all_positions() -> None:
    """Emergency: close all open positions at market price."""
    log.critical("FAIL SAFE: Flattening all open positions.")
    positions = mt5.positions_get()
    if not positions:
        return
    for pos in positions:
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "comment": "fail_safe_flatten",
        }
        result = mt5.order_send(request)
        log.info("Flatten ticket=%d result=%s", pos.ticket, result)


class TradingBot:
    """
    Main trading bot. Manages the full lifecycle:
      connect → poll bars → strategy → execute → monitor → shutdown.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._client = MT5Client()
        self._running = False

        # Per-symbol state
        self._processors: Dict[str, StrategyProcessor] = {}
        self._open_plans: Dict[int, tuple] = {}  # ticket → (ExitPlan, current_sl)

        # Risk management
        self._daily_tracker: Optional[DailyLossTracker] = None
        self._exit_mgr = ExitManager(
            r_multiple=self._cfg.r_multiple_tp,
            sl_spread_buffer=self._cfg.sl_spread_buffer,
            breakeven_at_r=self._cfg.breakeven_at_r,
            max_hold_bars=self._cfg.max_hold_bars,
            use_atr_sl=self._cfg.use_atr_sl,
            atr_sl_multiplier=self._cfg.atr_sl_multiplier,
            partial_tp_enabled=self._cfg.partial_tp_enabled,
            partial_tp_r=self._cfg.partial_tp_r,
            partial_tp_fraction=self._cfg.partial_tp_fraction,
        )

        # Execution quality
        session_start = dt_time(*[int(x) for x in self._cfg.session_start_utc.split(":")])
        session_end = dt_time(*[int(x) for x in self._cfg.session_end_utc.split(":")])
        exec_cfg = ExecutionConfig(
            session_start_utc=session_start,
            session_end_utc=session_end,
            spread_filter_multiplier=self._cfg.spread_filter_multiplier,
            signal_latency_budget_s=self._cfg.signal_latency_budget_s,
            news_blackout_minutes=self._cfg.news_blackout_minutes,
            fvg_quality_threshold=self._cfg.fvg_quality_threshold,    # Rev 3
            session_strength_min=self._cfg.session_strength_min,       # Rev 3
        )
        self._executor = OrderExecutor(exec_cfg)

        # Bar cache per symbol
        self._bar_cache: Dict[str, pd.DataFrame] = {}
        self._bar_idx: Dict[str, int] = {}

        # Rev 3: Session scorer (built from first bar fetch) and news loader
        self._session_scorer: Optional[SessionScorer] = None
        self._news_last_refresh: Optional[datetime] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect and enter the main polling loop."""
        if not self._client.connect():
            log.critical("Failed to connect to MT5. Exiting.")
            sys.exit(1)

        # Initialise daily loss tracker
        account = mt5.account_info()
        self._daily_tracker = DailyLossTracker(
            starting_equity=account.equity,
            max_daily_loss_pct=self._cfg.max_daily_loss_pct,
        )
        log.info("Bot started. Equity=%.2f. Symbols=%s",
                 account.equity, self._cfg.symbols)

        # Initialise per-symbol processors
        for sym in self._cfg.symbols:
            self._processors[sym] = StrategyProcessor(sym, cfg=self._cfg)
            self._bar_idx[sym] = 0

        # Rev 3: Load news and build initial session scorer from recent bars
        self._refresh_news()
        self._init_session_scorer()

        self._running = True
        self._main_loop()

    def stop(self) -> None:
        """Graceful shutdown."""
        log.info("Shutdown requested.")
        self._running = False
        self._client.disconnect()

    def fail_safe(self, reason: str = "unhandled exception") -> None:
        """Emergency shutdown: flatten all, halt entries."""
        log.critical("FAIL SAFE triggered: %s", reason)
        self._running = False
        try:
            _flatten_all_positions()
        except Exception as e:
            log.error("Error during fail-safe flatten: %s", e)
        self._client.disconnect()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        """Poll MT5 every second, process new bars when they appear."""
        last_times: Dict[str, Optional[pd.Timestamp]] = {s: None for s in self._cfg.symbols}

        while self._running:
            if not self._client.is_connected:
                log.warning("Connection lost. Attempting reconnect…")
                if not self._client.reconnect():
                    break  # hard stop

            if self._daily_tracker and self._daily_tracker.is_halted:
                log.critical("Daily loss ceiling hit — bot halted. Exiting loop.")
                break

            # Rev 3: Periodic news refresh (every NEWS_CACHE_MAX_AGE_HOURS)
            self._refresh_news()

            for symbol in self._cfg.symbols:
                try:
                    result = fetch_bars(symbol, count=200)
                    bars = result.bars
                    if bars.empty:
                        continue

                    latest_time = bars.iloc[-1]["time"]
                    if latest_time == last_times[symbol]:
                        continue  # no new bar yet

                    last_times[symbol] = latest_time
                    self._process_new_bar(symbol, bars)

                except Exception as e:
                    log.exception("Unhandled exception for %s: %s", symbol, e)
                    self.fail_safe(str(e))
                    return

            time.sleep(1)

    def _process_new_bar(self, symbol: str, bars: pd.DataFrame) -> None:
        """Process the latest bar for a symbol through the full pipeline."""
        bar = bars.iloc[-1]
        idx = self._bar_idx[symbol]
        self._bar_idx[symbol] += 1

        avg_spread = compute_average_spread(bars)
        current_spread_pips = float(bar.get("spread", 2))

        # Rev 3: ATR for adaptive SL
        atr_pips = compute_atr_pips(bars, period=self._cfg.atr_period) if self._cfg.use_atr_sl else 0.0

        # Rev 3: Session strength score for this hour
        session_strength = 1.0
        if self._session_scorer and self._cfg.session_strength_min > 0:
            session_strength = self._session_scorer.get_hour_strength(bar["time"].hour)

        # ── Run strategy ──────────────────────────────────────────────────────
        signals = self._processors[symbol].process_bar(bar, idx, avg_spread)

        # ── Execute new signals ───────────────────────────────────────────────
        for signal in signals:
            now = datetime.now(timezone.utc)
            filter_result = self._executor.run_filters(
                signal, current_spread_pips, avg_spread,
                now_utc=now,
                session_strength_score=session_strength,   # Rev 3
            )
            if not filter_result.passed:
                log.info("Signal filtered: %s", filter_result.reason)
                continue

            self._submit_order(signal, bar, idx, current_spread_pips, atr_pips=atr_pips)

    def _submit_order(
        self,
        signal: Signal,
        bar: pd.Series,
        bar_idx: int,
        spread_pips: float,
        atr_pips: float = 0.0,        # Rev 3
    ) -> None:
        """Build and send a market order for a confirmed signal."""
        sym_info = _get_symbol_info(signal.symbol)
        if sym_info is None:
            return

        account = mt5.account_info()
        if account is None:
            log.error("Could not fetch account info — skipping order.")
            return

        pip_size = SYMBOL_PIP_SIZE.get(signal.symbol, 0.0001)
        sl_pips = abs(signal.entry_price - signal.ob_bottom) / pip_size + self._cfg.sl_spread_buffer * spread_pips

        lots = calculate_lot_size(
            account.equity, self._cfg.risk_pct, sl_pips, sym_info
        )
        tick = mt5.symbol_info_tick(signal.symbol)
        if tick is None:
            return

        if signal.direction == Direction.BULLISH:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        plan = self._exit_mgr.create_exit_plan(
            signal, price, bar_idx, spread_pips, atr_pips=atr_pips  # Rev 3
        )
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": plan.stop_loss,
            "tp": plan.take_profit,
            "deviation": 5,
            "magic": 20240102,
            "comment": f"ob_scalper_{signal.direction.name[:1]}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("Order filled: ticket=%d %s %.2f lots at %.5f",
                     result.order, signal.direction.name, lots, price)
            self._client.update_internal_positions(
                {result.order: {"ticket": result.order, "plan": plan}}
            )
        elif result and result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            # One retry
            if self._executor.should_retry_requote(
                float(mt5.symbol_info_tick(signal.symbol).ask - mt5.symbol_info_tick(signal.symbol).bid) / pip_size,
                spread_pips,
                retry_count=0,
            ):
                result2 = mt5.order_send(request)
                log.info("Requote retry result: %s", result2)
        else:
            log.error("Order send failed: %s", result)

    # ── Rev 3: Helper methods ────────────────────────────────────────────────

    def _refresh_news(self) -> None:
        """Refresh investing.com news calendar if stale or on first call."""
        try:
            events = refresh_news_if_stale(
                currencies=list(self._cfg.news_currencies),
                max_age_hours=self._cfg.news_cache_max_age_hours,
            )
            self._executor.set_news_events(events)
            self._news_last_refresh = datetime.now(timezone.utc)
            if events:
                log.info("News blackout: %d events loaded for %s",
                         len(events), self._cfg.news_currencies)
        except Exception as e:
            log.warning("Failed to refresh news calendar: %s", e)

    def _init_session_scorer(self) -> None:
        """Build session strength profile from recent historical bars."""
        if not self._cfg.symbols:
            return
        sym = self._cfg.symbols[0]
        try:
            result = fetch_bars(sym, count=5000)
            self._session_scorer = SessionScorer(result.bars)
            log.info("Session scorer built from %d bars for %s", len(result.bars), sym)
        except Exception as e:
            log.warning("Failed to build session scorer: %s", e)


def main() -> None:
    """Entry point — set up signal handlers and start the bot."""
    bot = TradingBot()

    def _on_shutdown(signum, frame):
        log.info("Signal %d received — stopping bot.", signum)
        bot.stop()

    signal.signal(signal.SIGINT, _on_shutdown)
    signal.signal(signal.SIGTERM, _on_shutdown)

    try:
        bot.start()
    except Exception as e:
        log.exception("Fatal error in main: %s", e)
        bot.fail_safe(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
