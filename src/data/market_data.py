"""
Module 2: Market Data Layer
============================
Handles M1 OHLCV bar ingestion from MT5 and enforces data integrity
rules per SDD Rev 2, Section 3.1:

  - Gap detection  : flag bars where timestamp jump > 2 × M1 interval
  - Deduplication  : remove bars with duplicate timestamps (reconnect artefacts)
  - Zero-volume    : flag zero tick_volume bars; exclude from displacement detection

The public surface is one function, ``fetch_bars()``, that returns a
clean, validated DataFrame ready for the strategy processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from src.core.logger import get_logger

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
M1_SECONDS = 60           # Expected interval between M1 bars
GAP_MULTIPLIER = 2        # Flag if gap > GAP_MULTIPLIER × M1_SECONDS
REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DataFetchResult:
    bars: pd.DataFrame                  # Clean bar data
    gaps_detected: List[pd.Timestamp] = field(default_factory=list)
    duplicates_removed: int = 0
    zero_volume_bars: List[pd.Timestamp] = field(default_factory=list)


# ── Core functions ────────────────────────────────────────────────────────────

def fetch_bars(
    symbol: str,
    count: int = 500,
    *,
    _mt5=None,          # injectable for testing
) -> DataFetchResult:
    """
    Fetch the last ``count`` M1 bars for ``symbol`` from the MT5 terminal,
    apply integrity checks, and return a validated DataFetchResult.

    Args:
        symbol: MT5 symbol string, e.g. "EURUSD".
        count:  Number of M1 bars to fetch (most recent first, reversed to ascending).
        _mt5:   Optional MT5 module override (used in tests).

    Returns:
        DataFetchResult with a clean DataFrame and integrity metadata.

    Raises:
        RuntimeError if MT5 returns no data.
    """
    _mt5 = _mt5 or mt5
    raw = _mt5.copy_rates_from_pos(symbol, _mt5.TIMEFRAME_M1, 0, count)
    if raw is None or len(raw) == 0:
        err = _mt5.last_error()
        raise RuntimeError(f"No data returned for {symbol}: {err}")

    df = pd.DataFrame(raw)

    # Ensure time is a proper datetime index (MT5 returns UTC Unix timestamps)
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    df = df.sort_values("time").reset_index(drop=True)

    result = DataFetchResult(bars=df)
    result = _remove_duplicates(df, result)
    result = _detect_gaps(result.bars, result)
    result = _flag_zero_volume(result.bars, result)

    log.debug(
        "fetch_bars(%s, %d): %d clean bars, %d gaps, %d dupes, %d zero-vol",
        symbol, count,
        len(result.bars),
        len(result.gaps_detected),
        result.duplicates_removed,
        len(result.zero_volume_bars),
    )
    return result


# ── Integrity checks ──────────────────────────────────────────────────────────

def _remove_duplicates(df: pd.DataFrame, result: DataFetchResult) -> DataFetchResult:
    """De-duplicate rows with identical timestamps (reconnect artefact)."""
    before = len(df)
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
    result.duplicates_removed = before - len(df)
    if result.duplicates_removed:
        log.warning("Removed %d duplicate timestamp(s) from bar data.",
                    result.duplicates_removed)
    result.bars = df
    return result


def _detect_gaps(df: pd.DataFrame, result: DataFetchResult) -> DataFetchResult:
    """
    Flag timestamps where the gap to the prior bar exceeds
    GAP_MULTIPLIER × M1_SECONDS (i.e., one or more bars are missing).
    """
    if len(df) < 2:
        return result

    times = df["time"]
    deltas = times.diff().dt.total_seconds()
    threshold = GAP_MULTIPLIER * M1_SECONDS

    gap_mask = deltas > threshold
    gap_timestamps = list(times[gap_mask])

    if gap_timestamps:
        log.warning("Detected %d gap(s) in bar data: %s", len(gap_timestamps), gap_timestamps)

    result.gaps_detected = gap_timestamps
    return result


def _flag_zero_volume(df: pd.DataFrame, result: DataFetchResult) -> DataFetchResult:
    """
    Flag bars with tick_volume == 0.

    These bars are NOT removed from the DataFrame (downstream code may still
    need them for price reference) but are recorded so the strategy processor
    can exclude them from displacement detection.
    """
    zero_mask = df["tick_volume"] == 0
    zero_times = list(df.loc[zero_mask, "time"])
    if zero_times:
        log.warning("Found %d zero-volume bar(s); excluding from displacement detection.",
                    len(zero_times))
    result.zero_volume_bars = zero_times
    result.bars = df
    return result


def validate_bar_row(row: pd.Series) -> bool:
    """
    Return True if a single bar row passes basic sanity checks.
    Used by the live data pipeline to gate each incoming bar.
    """
    if row.get("tick_volume", 0) == 0:
        return False
    if row.get("high", 0) < row.get("low", 1):   # inverted OHLC
        return False
    if row.get("open", 0) <= 0 or row.get("close", 0) <= 0:
        return False
    return True


def compute_average_spread(df: pd.DataFrame, window: int = 20) -> float:
    """
    Compute the rolling average spread over the last ``window`` bars.
    Used by the execution quality filter (spread ceiling check).
    """
    if "spread" not in df.columns or len(df) < 1:
        return 0.0
    spread_slice = df["spread"].tail(window)
    return float(spread_slice.mean()) if len(spread_slice) > 0 else 0.0
