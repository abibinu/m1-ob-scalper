"""
Core configuration module.

Loads all settings from environment variables (or a .env file) into a
typed, immutable Config dataclass.  The module exposes a single
``get_config()`` factory that is cached after the first call so every
other module always sees the same object.

Rev 3 additions:
  - FVG confluence filter parameters
  - Adaptive ATR-based SL/TP parameters
  - Session strength filter parameters
  - Partial TP parameters
  - News loader parameters
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

# Load .env file from the project root (no-op if file is absent)
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


def _env_list(key: str, default: str) -> List[str]:
    raw = os.environ.get(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # MT5 connection
    mt5_login: int
    mt5_password: str
    mt5_server: str

    # Trading
    symbols: List[str]
    risk_pct: float                # e.g. 0.5 → 0.5 % per trade
    max_daily_loss_pct: float      # e.g. 2.0 → halt at 2 % daily loss
    max_ob_age_bars: int           # OB expiry window in M1 bars
    max_ob_per_symbol: int         # Max concurrent OBs in register
    ob_stack_tolerance: float      # Merge OBs within N × avg spread
    displacement_threshold: float  # Min range multiplier vs. lookback avg

    # Exit
    r_multiple_tp: float           # Take-profit in R-multiples
    breakeven_at_r: float          # 0 = disabled
    max_hold_bars: int             # Time-based exit threshold
    sl_spread_buffer: float        # Extra SL buffer in spread multiples

    # Execution quality filters
    spread_filter_multiplier: float
    signal_latency_budget_s: float
    news_blackout_minutes: int

    # Session (UTC)
    session_start_utc: str         # "HH:MM"
    session_end_utc: str           # "HH:MM"

    # Backtest / validation
    slippage_pips: float
    walk_forward_split: float      # Fraction of data used as in-sample
    min_profit_factor: float       # Gate criterion

    # ── Rev 3: FVG Confluence Filter ─────────────────────────────────────────
    fvg_filter_enabled: bool       # Compute FVG confluence on every signal
    fvg_require_confluence: bool   # Hard-reject signals with no FVG overlap
    fvg_quality_threshold: float   # Min quality_score to enter (0.0 = disabled)

    # ── Rev 3: Adaptive ATR-Based SL/TP ──────────────────────────────────────
    use_atr_sl: bool               # Use ATR instead of fixed spread buffer for SL
    atr_period: int                # ATR calculation period in M1 bars
    atr_sl_multiplier: float       # SL = ob_boundary ± ATR * multiplier

    # ── Rev 3: Session Strength Filter ───────────────────────────────────────
    session_strength_min: float    # Min hourly volatility score [0–1] to enter

    # ── Rev 3: Partial Take Profit ────────────────────────────────────────────
    partial_tp_enabled: bool       # Close partial position at partial_tp_r
    partial_tp_r: float            # Close fraction at this R level (e.g. 1.0)
    partial_tp_fraction: float     # Fraction of position to close (e.g. 0.5)

    # ── Rev 3: Trailing stop ──────────────────────────────────────────────────
    trailing_stop_activation_r: float
    trailing_stop_distance_pips: float

    # ── Rev 3: News loader ───────────────────────────────────────────────────
    news_currencies: List[str]     # Currencies to monitor (e.g. ["USD","EUR","GBP"])
    news_cache_max_age_hours: float  # Refresh news cache if older than this

    # Derived convenience fields
    symbols_tuple: tuple = field(init=False)

    def __post_init__(self) -> None:
        # frozen=True means we must use object.__setattr__ for derived fields
        object.__setattr__(self, "symbols_tuple", tuple(self.symbols))


# ---------------------------------------------------------------------------
# Factory (cached singleton)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config, populated from environment variables."""
    return Config(
        mt5_login=_env_int("MT5_LOGIN", 0),
        mt5_password=_env_str("MT5_PASSWORD", ""),
        mt5_server=_env_str("MT5_SERVER", "Vantage-Demo"),

        symbols=_env_list("SYMBOLS", "EURUSD"),
        risk_pct=_env_float("RISK_PCT", 0.5),
        max_daily_loss_pct=_env_float("MAX_DAILY_LOSS_PCT", 2.0),
        max_ob_age_bars=_env_int("MAX_OB_AGE_BARS", 75),
        max_ob_per_symbol=_env_int("MAX_OB_PER_SYMBOL", 5),
        ob_stack_tolerance=_env_float("OB_STACK_TOLERANCE", 1.5),
        displacement_threshold=_env_float("DISPLACEMENT_THRESHOLD", 1.5),

        r_multiple_tp=_env_float("R_MULTIPLE_TP", 2.0),
        breakeven_at_r=_env_float("BREAKEVEN_AT_R", 1.0),
        max_hold_bars=_env_int("MAX_HOLD_BARS", 40),
        sl_spread_buffer=_env_float("SL_SPREAD_BUFFER", 1.5),

        spread_filter_multiplier=_env_float("SPREAD_FILTER_MULTIPLIER", 1.5),
        signal_latency_budget_s=_env_float("SIGNAL_LATENCY_BUDGET_S", 1.5),
        news_blackout_minutes=_env_int("NEWS_BLACKOUT_MINUTES", 15),

        session_start_utc=_env_str("SESSION_START_UTC", "07:00"),
        session_end_utc=_env_str("SESSION_END_UTC", "12:00"),

        slippage_pips=_env_float("SLIPPAGE_PIPS", 0.5),
        walk_forward_split=_env_float("WALK_FORWARD_SPLIT", 0.7),
        min_profit_factor=_env_float("MIN_PROFIT_FACTOR", 1.2),

        # Rev 3: FVG
        fvg_filter_enabled=_env_bool("FVG_FILTER_ENABLED", True),
        fvg_require_confluence=_env_bool("FVG_REQUIRE_CONFLUENCE", False),
        fvg_quality_threshold=_env_float("FVG_QUALITY_THRESHOLD", 0.0),

        # Rev 3: ATR SL
        use_atr_sl=_env_bool("USE_ATR_SL", False),
        atr_period=_env_int("ATR_PERIOD", 14),
        atr_sl_multiplier=_env_float("ATR_SL_MULTIPLIER", 1.0),

        # Rev 3: Session strength
        session_strength_min=_env_float("SESSION_STRENGTH_MIN", 0.0),

        # Rev 3: Partial TP
        partial_tp_enabled=_env_bool("PARTIAL_TP_ENABLED", False),
        partial_tp_r=_env_float("PARTIAL_TP_R", 1.0),
        partial_tp_fraction=_env_float("PARTIAL_TP_FRACTION", 0.5),

        # Rev 3: Trailing stop
        trailing_stop_activation_r=_env_float("TRAILING_STOP_ACTIVATION_R", 0.0),
        trailing_stop_distance_pips=_env_float("TRAILING_STOP_DISTANCE_PIPS", 0.0),

        # Rev 3: News
        news_currencies=_env_list("NEWS_CURRENCIES", "USD,EUR,GBP"),
        news_cache_max_age_hours=_env_float("NEWS_CACHE_MAX_AGE_HOURS", 12.0),
    )
