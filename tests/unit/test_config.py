"""
Unit tests — Phase 1: Config & Logger
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Config falls back to sensible defaults with no env vars set."""

    def test_default_symbols(self):
        from src.core.config import get_config
        cfg = get_config()
        assert "EURUSD" in cfg.symbols

    def test_default_risk_pct(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.risk_pct == 0.5

    def test_default_max_daily_loss(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.max_daily_loss_pct == 2.0

    def test_default_ob_age(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.max_ob_age_bars == 75

    def test_default_max_ob_per_symbol(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.max_ob_per_symbol == 5

    def test_default_r_multiple(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.r_multiple_tp == 2.0

    def test_default_session_window(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.session_start_utc == "07:00"
        assert cfg.session_end_utc == "12:00"

    def test_symbols_tuple_derived(self):
        """symbols_tuple must equal tuple(symbols)."""
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.symbols_tuple == tuple(cfg.symbols)


class TestConfigEnvOverrides:
    """Env-var overrides are picked up correctly."""

    def test_override_risk_pct(self, monkeypatch):
        monkeypatch.setenv("RISK_PCT", "0.25")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.risk_pct == 0.25

    def test_override_symbols_multi(self, monkeypatch):
        monkeypatch.setenv("SYMBOLS", "EURUSD,GBPUSD,USDJPY")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.symbols == ["EURUSD", "GBPUSD", "USDJPY"]
        assert len(cfg.symbols_tuple) == 3

    def test_override_max_ob_age(self, monkeypatch):
        monkeypatch.setenv("MAX_OB_AGE_BARS", "100")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.max_ob_age_bars == 100

    def test_override_session(self, monkeypatch):
        monkeypatch.setenv("SESSION_START_UTC", "08:00")
        monkeypatch.setenv("SESSION_END_UTC", "13:00")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.session_start_utc == "08:00"
        assert cfg.session_end_utc == "13:00"

    def test_override_spread_filter(self, monkeypatch):
        monkeypatch.setenv("SPREAD_FILTER_MULTIPLIER", "2.0")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.spread_filter_multiplier == 2.0

    def test_override_slippage(self, monkeypatch):
        monkeypatch.setenv("SLIPPAGE_PIPS", "1.0")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.slippage_pips == 1.0

    def test_mt5_login_int(self, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN", "99887766")
        from src.core.config import get_config
        cfg = get_config()
        assert cfg.mt5_login == 99887766
        assert isinstance(cfg.mt5_login, int)


class TestConfigImmutability:
    """Config dataclass is frozen — mutation must raise."""

    def test_frozen(self):
        from src.core.config import get_config
        cfg = get_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.risk_pct = 99.0  # type: ignore[misc]


class TestConfigSingleton:
    """get_config() returns the same object within a test (cache works)."""

    def test_same_object(self):
        from src.core.config import get_config
        a = get_config()
        b = get_config()
        assert a is b


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------

class TestLogger:

    def test_get_logger_returns_logger(self):
        import logging
        from src.core.logger import get_logger
        log = get_logger("test.module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test.module"

    def test_get_logger_idempotent(self):
        from src.core.logger import get_logger
        a = get_logger("test.idem")
        b = get_logger("test.idem")
        assert a is b

    def test_log_dir_created(self, tmp_path, monkeypatch):
        """Logger creates the log directory if absent."""
        import importlib
        import logging

        log_dir = tmp_path / "test_logs"
        monkeypatch.setenv("LOG_DIR", str(log_dir))

        # Force re-initialisation by reloading the module
        import src.core.logger as logger_mod
        logger_mod._INITIALIZED = False
        logger_mod.LOG_DIR = log_dir
        logger_mod.LOG_FILE = log_dir / "ob_scalper.log"
        logger_mod._setup_root_logger()

        assert log_dir.exists()
