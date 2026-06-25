"""
Module 1: Core Engine & Connection Manager
==========================================
Handles MT5 authentication, reconnection with exponential backoff,
position reconciliation, and hard-stop state management.

Reconnect policy (per SDD Rev 2, Section 2.1):
  - Retry mt5.login() at 2s, 4s, 8s, 16s, 30s (capped) intervals.
  - After 10 consecutive failures (~5 min), enter hard-stop requiring manual restart.
  - On successful reconnect, reconcile internal state vs. broker book before
    resuming signal generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import MetaTrader5 as mt5

from src.core.config import get_config
from src.core.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BACKOFF_SEQUENCE = [2, 4, 8, 16, 30]  # seconds between retries
_MAX_RETRIES = 10                        # hard-stop after this many failures


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    HARD_STOP = auto()  # manual restart required


@dataclass
class ReconciliationResult:
    """Result of position / order reconciliation after reconnect."""
    open_positions: List[dict] = field(default_factory=list)
    pending_orders: List[dict] = field(default_factory=list)
    discrepancies: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MT5Client
# ---------------------------------------------------------------------------

class MT5Client:
    """
    Wraps the MetaTrader5 Python API with:
      - Explicit connect/disconnect lifecycle
      - Exponential backoff reconnect
      - Position reconciliation on reconnect
      - Hard-stop after max retries
    """

    def __init__(self) -> None:
        self._state = ConnectionState.DISCONNECTED
        self._retry_count = 0
        self._internal_positions: dict[int, dict] = {}  # ticket → position snapshot

    # ── public state ────────────────────────────────────────────────────────

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    @property
    def is_hard_stopped(self) -> bool:
        return self._state == ConnectionState.HARD_STOP

    # ── lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialise the MT5 terminal and log in.

        Returns True on success, False if already in HARD_STOP state.
        Raises RuntimeError only in programmer-error scenarios (e.g. double connect).
        """
        if self._state == ConnectionState.HARD_STOP:
            log.error("MT5Client in HARD_STOP — manual restart required.")
            return False

        cfg = get_config()
        self._state = ConnectionState.CONNECTING
        log.info("Connecting to MT5 server '%s' (login %d)…",
                 cfg.mt5_server, cfg.mt5_login)

        if not mt5.initialize():
            err = mt5.last_error()
            log.error("mt5.initialize() failed: %s", err)
            self._state = ConnectionState.DISCONNECTED
            return False

        if not mt5.login(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server):
            err = mt5.last_error()
            log.error("mt5.login() failed: %s", err)
            mt5.shutdown()
            self._state = ConnectionState.DISCONNECTED
            return False

        self._state = ConnectionState.CONNECTED
        self._retry_count = 0
        log.info("MT5 connected successfully.")
        return True

    def disconnect(self) -> None:
        """Gracefully shut down the MT5 connection."""
        mt5.shutdown()
        self._state = ConnectionState.DISCONNECTED
        log.info("MT5 disconnected.")

    # ── reconnection ─────────────────────────────────────────────────────────

    def reconnect(self, _sleep_fn=time.sleep) -> bool:
        """
        Attempt exponential-backoff reconnection.

        Args:
            _sleep_fn: Injectable sleep function (use mock in tests).

        Returns True if reconnection succeeded, False if hard-stop triggered.
        """
        if self._state == ConnectionState.HARD_STOP:
            return False

        cfg = get_config()
        log.warning("Connection lost. Starting exponential backoff reconnect…")

        for attempt in range(1, _MAX_RETRIES + 1):
            delay = _BACKOFF_SEQUENCE[min(attempt - 1, len(_BACKOFF_SEQUENCE) - 1)]
            log.info("Reconnect attempt %d/%d — waiting %ds…",
                     attempt, _MAX_RETRIES, delay)
            _sleep_fn(delay)

            # Re-initialise terminal
            if not mt5.initialize():
                log.warning("mt5.initialize() failed on attempt %d.", attempt)
                self._retry_count += 1
                continue

            if mt5.login(cfg.mt5_login, cfg.mt5_password, cfg.mt5_server):
                self._state = ConnectionState.CONNECTED
                self._retry_count = 0
                log.info("Reconnected on attempt %d.", attempt)
                self._reconcile_positions()
                return True
            else:
                err = mt5.last_error()
                log.warning("Login failed on attempt %d: %s", attempt, err)
                mt5.shutdown()
                self._retry_count += 1

        # All retries exhausted
        log.critical(
            "Hard-stop: %d consecutive reconnect failures. Manual restart required.",
            _MAX_RETRIES
        )
        self._state = ConnectionState.HARD_STOP
        return False

    # ── position reconciliation ───────────────────────────────────────────────

    def _reconcile_positions(self) -> ReconciliationResult:
        """
        Compare live broker book vs. internal position cache.
        Logs discrepancies but does NOT auto-close positions —
        that decision is left to the caller / risk manager.
        """
        result = ReconciliationResult()

        raw_positions = mt5.positions_get()
        raw_orders = mt5.orders_get()

        live_positions = list(raw_positions) if raw_positions else []
        live_orders = list(raw_orders) if raw_orders else []

        result.open_positions = [self._position_to_dict(p) for p in live_positions]
        result.pending_orders = [self._order_to_dict(o) for o in live_orders]

        live_tickets = {p.ticket for p in live_positions} if live_positions else set()
        internal_tickets = set(self._internal_positions.keys())

        for ticket in internal_tickets - live_tickets:
            msg = f"Orphaned internal position ticket={ticket} not found on broker."
            log.warning(msg)
            result.discrepancies.append(msg)

        for ticket in live_tickets - internal_tickets:
            msg = f"Untracked broker position ticket={ticket} not in internal cache."
            log.warning(msg)
            result.discrepancies.append(msg)

        # Sync internal cache to broker truth
        self._internal_positions = {
            p.ticket: self._position_to_dict(p) for p in live_positions
        } if live_positions else {}

        log.info(
            "Reconciliation complete: %d live positions, %d discrepancies.",
            len(live_positions), len(result.discrepancies)
        )
        return result

    def update_internal_positions(self, positions: dict[int, dict]) -> None:
        """Allow the execution engine to update the internal position cache."""
        self._internal_positions.update(positions)

    def clear_internal_positions(self) -> None:
        self._internal_positions.clear()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _position_to_dict(pos) -> dict:
        return {
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": pos.type,
            "volume": pos.volume,
            "price_open": pos.price_open,
            "sl": pos.sl,
            "tp": pos.tp,
        }

    @staticmethod
    def _order_to_dict(order) -> dict:
        return {
            "ticket": order.ticket,
            "symbol": order.symbol,
            "type": order.type,
            "volume_initial": order.volume_initial,
            "price_open": order.price_open,
        }
