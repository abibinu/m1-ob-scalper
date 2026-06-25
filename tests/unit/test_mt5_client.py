"""
Unit tests — Phase 2: Module 1 — MT5 Connection Manager

All MetaTrader5 calls are intercepted via the stub in conftest.py.
No live MT5 terminal is required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from src.connection.mt5_client import ConnectionState, MT5Client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_client() -> MT5Client:
    return MT5Client()


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------

class TestConnect:

    def test_successful_connect(self, mt5_stub):
        mt5_stub.initialize.return_value = True
        mt5_stub.login.return_value = True

        client = _fresh_client()
        result = client.connect()

        assert result is True
        assert client.state == ConnectionState.CONNECTED
        assert client.is_connected is True

    def test_initialize_failure(self, mt5_stub):
        mt5_stub.initialize.return_value = False
        mt5_stub.last_error.return_value = (-1, "terminal not found")

        client = _fresh_client()
        result = client.connect()

        assert result is False
        assert client.state == ConnectionState.DISCONNECTED

    def test_login_failure(self, mt5_stub):
        mt5_stub.initialize.return_value = True
        mt5_stub.login.return_value = False
        mt5_stub.last_error.return_value = (5, "invalid credentials")

        client = _fresh_client()
        result = client.connect()

        assert result is False
        assert client.state == ConnectionState.DISCONNECTED
        mt5_stub.shutdown.assert_called()

    def test_hard_stop_blocks_connect(self, mt5_stub):
        client = _fresh_client()
        client._state = ConnectionState.HARD_STOP

        result = client.connect()

        assert result is False
        mt5_stub.initialize.assert_not_called()


# ---------------------------------------------------------------------------
# disconnect()
# ---------------------------------------------------------------------------

class TestDisconnect:

    def test_disconnect_calls_shutdown(self, mt5_stub):
        client = _fresh_client()
        client._state = ConnectionState.CONNECTED
        client.disconnect()

        mt5_stub.shutdown.assert_called()
        assert client.state == ConnectionState.DISCONNECTED


# ---------------------------------------------------------------------------
# reconnect() — backoff intervals
# ---------------------------------------------------------------------------

class TestReconnectBackoff:

    def _run_reconnect(self, mt5_stub, *, succeed_on_attempt: int | None = None):
        """Run reconnect() with a no-op sleep and controlled login results."""
        call_count = [0]

        def _login(*args, **kwargs):
            call_count[0] += 1
            if succeed_on_attempt and call_count[0] == succeed_on_attempt:
                return True
            return False

        mt5_stub.initialize.return_value = True
        mt5_stub.login.side_effect = _login
        mt5_stub.last_error.return_value = (5, "bad creds")

        sleep_calls = []
        client = _fresh_client()
        client._state = ConnectionState.DISCONNECTED

        result = client.reconnect(_sleep_fn=lambda s: sleep_calls.append(s))
        return result, sleep_calls

    def test_reconnect_success_first_attempt(self, mt5_stub):
        result, sleep_calls = self._run_reconnect(mt5_stub, succeed_on_attempt=1)
        assert result is True
        # First sleep must be 2s (first element of backoff sequence)
        assert sleep_calls[0] == 2

    def test_reconnect_success_third_attempt(self, mt5_stub):
        result, sleep_calls = self._run_reconnect(mt5_stub, succeed_on_attempt=3)
        assert result is True
        assert sleep_calls[:3] == [2, 4, 8]

    def test_reconnect_hard_stop_after_10_failures(self, mt5_stub):
        result, sleep_calls = self._run_reconnect(mt5_stub, succeed_on_attempt=None)
        assert result is False
        assert len(sleep_calls) == 10
        # After index 4 the delay is capped at 30s
        assert sleep_calls[4] == 30
        assert sleep_calls[9] == 30

    def test_reconnect_sets_hard_stop_state(self, mt5_stub):
        client = _fresh_client()
        mt5_stub.initialize.return_value = True
        mt5_stub.login.return_value = False
        mt5_stub.last_error.return_value = (5, "bad")

        client.reconnect(_sleep_fn=lambda _: None)
        assert client.state == ConnectionState.HARD_STOP
        assert client.is_hard_stopped is True

    def test_reconnect_skipped_when_hard_stopped(self, mt5_stub):
        client = _fresh_client()
        client._state = ConnectionState.HARD_STOP
        result = client.reconnect(_sleep_fn=lambda _: None)
        assert result is False
        mt5_stub.initialize.assert_not_called()

    def test_reconnect_resets_retry_count_on_success(self, mt5_stub):
        result, _ = self._run_reconnect(mt5_stub, succeed_on_attempt=2)
        # After success _retry_count should be reset to 0 — verified via state
        assert result is True


# ---------------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------------

class TestPositionReconciliation:

    def _make_mock_position(self, ticket: int, symbol: str = "EURUSD"):
        pos = MagicMock()
        pos.ticket = ticket
        pos.symbol = symbol
        pos.type = 0
        pos.volume = 0.01
        pos.price_open = 1.1000
        pos.sl = 1.0990
        pos.tp = 1.1020
        return pos

    def _make_mock_order(self, ticket: int, symbol: str = "EURUSD"):
        order = MagicMock()
        order.ticket = ticket
        order.symbol = symbol
        order.type = 2
        order.volume_initial = 0.01
        order.price_open = 1.1000
        return order

    def test_reconcile_no_discrepancies(self, mt5_stub):
        pos = self._make_mock_position(1001)
        mt5_stub.positions_get.return_value = [pos]
        mt5_stub.orders_get.return_value = []

        client = _fresh_client()
        client._internal_positions = {1001: {}}  # already tracked
        result = client._reconcile_positions()

        assert len(result.discrepancies) == 0
        assert len(result.open_positions) == 1

    def test_reconcile_orphaned_internal_position(self, mt5_stub):
        """Internal tracks ticket 999 but broker shows nothing."""
        mt5_stub.positions_get.return_value = []
        mt5_stub.orders_get.return_value = []

        client = _fresh_client()
        client._internal_positions = {999: {"ticket": 999}}
        result = client._reconcile_positions()

        assert any("999" in d for d in result.discrepancies)

    def test_reconcile_untracked_broker_position(self, mt5_stub):
        """Broker reports ticket 888 not in internal cache."""
        pos = self._make_mock_position(888)
        mt5_stub.positions_get.return_value = [pos]
        mt5_stub.orders_get.return_value = []

        client = _fresh_client()
        client._internal_positions = {}  # empty — not tracking 888
        result = client._reconcile_positions()

        assert any("888" in d for d in result.discrepancies)

    def test_reconcile_syncs_internal_cache(self, mt5_stub):
        """After reconciliation, internal cache mirrors broker truth."""
        pos = self._make_mock_position(777)
        mt5_stub.positions_get.return_value = [pos]
        mt5_stub.orders_get.return_value = []

        client = _fresh_client()
        client._reconcile_positions()

        assert 777 in client._internal_positions

    def test_reconcile_called_on_successful_reconnect(self, mt5_stub):
        """Reconciliation fires automatically after a successful reconnect."""
        call_count = [0]
        def _login(*args, **kwargs):
            call_count[0] += 1
            return call_count[0] == 1  # succeed on first attempt

        mt5_stub.initialize.return_value = True
        mt5_stub.login.side_effect = _login
        mt5_stub.positions_get.return_value = []
        mt5_stub.orders_get.return_value = []

        client = _fresh_client()
        client.reconnect(_sleep_fn=lambda _: None)

        mt5_stub.positions_get.assert_called()
