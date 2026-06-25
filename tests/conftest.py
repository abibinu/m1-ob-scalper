"""
Shared pytest fixtures and test utilities.

Key design principle: MetaTrader5 is a Windows-only DLL-backed package.
We stub it out at import time so the entire test suite runs on any platform
and without a live MT5 terminal.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# MT5 stub — installed before any src import touches MetaTrader5
# ---------------------------------------------------------------------------

def _make_mt5_stub() -> types.ModuleType:
    """Return a MagicMock module with the MT5 constants and callables our
    code relies on."""
    mt5 = MagicMock(name="MetaTrader5")

    # Common constants
    mt5.TIMEFRAME_M1 = 1
    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.TRADE_ACTION_DEAL = 1
    mt5.ORDER_FILLING_IOC = 2
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.TRADE_RETCODE_REQUOTE = 10004

    # Default "happy path" return values
    mt5.initialize.return_value = True
    mt5.login.return_value = True
    mt5.shutdown.return_value = None
    mt5.last_error.return_value = (0, "no error")

    return mt5


# Install the stub into sys.modules before any project imports happen.
if "MetaTrader5" not in sys.modules:
    sys.modules["MetaTrader5"] = _make_mt5_stub()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mt5_stub():
    """Reset all MT5 stub call counts & side effects before every test."""
    stub = sys.modules["MetaTrader5"]
    stub.reset_mock()
    # Restore safe defaults after reset
    stub.initialize.return_value = True
    stub.login.return_value = True
    stub.shutdown.return_value = None
    stub.last_error.return_value = (0, "no error")
    yield


@pytest.fixture
def mt5_stub():
    """Expose the already-installed MT5 stub for per-test configuration."""
    return sys.modules["MetaTrader5"]


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Clear the lru_cache on get_config() between tests so env-var patches
    take effect cleanly."""
    from src.core.config import get_config
    get_config.cache_clear()
    yield
    get_config.cache_clear()
