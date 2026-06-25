"""
Unit tests — Phase 3: Module 2 — Market Data Layer

All MT5 API calls are replaced with synthetic DataFrame fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import (
    DataFetchResult,
    compute_average_spread,
    fetch_bars,
    validate_bar_row,
    _detect_gaps,
    _flag_zero_volume,
    _remove_duplicates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    n: int = 10,
    start: datetime | None = None,
    interval_s: int = 60,
    tick_volume: int = 100,
    spread: int = 2,
) -> pd.DataFrame:
    """Generate a clean synthetic M1 bar DataFrame."""
    if start is None:
        # Use a fixed epoch well away from 0 so unix-second encoding stays unique
        start = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
    times = pd.date_range(start=start, periods=n, freq=f"{interval_s}s", tz="UTC")
    base = 1.1000
    step = 0.0001 / max(n - 1, 1)
    df = pd.DataFrame(
        {
            "time": times,
            "open": [round(base + step * i, 6) for i in range(n)],
            "high": [round(base + step * i + 0.0005, 6) for i in range(n)],
            "low":  [round(base + step * i - 0.0005, 6) for i in range(n)],
            "close":[round(base + step * i + 0.0002, 6) for i in range(n)],
            "tick_volume": [tick_volume] * n,
            "spread": [spread] * n,
        }
    )
    return df


def _make_mt5_mock(bars_df: pd.DataFrame) -> MagicMock:
    """Wrap a DataFrame as a structured NumPy record array (as MT5 returns)."""
    mock = MagicMock()
    mock.TIMEFRAME_M1 = 1
    mock.last_error.return_value = (0, "no error")

    df_copy = bars_df.copy()

    # Convert tz-aware datetimes → unix seconds (int64) properly
    # .view(np.int64) gives nanoseconds; // 10^9 gives seconds
    unix_seconds = (
        df_copy["time"]
        .values.astype("datetime64[s]")
        .astype(np.int64)
    )

    dtypes = [
        ("time", np.int64),
        ("open", np.float64),
        ("high", np.float64),
        ("low", np.float64),
        ("close", np.float64),
        ("tick_volume", np.int64),
        ("spread", np.int32),
    ]
    arr = np.zeros(len(df_copy), dtype=dtypes)
    arr["time"] = unix_seconds
    for col in ("open", "high", "low", "close", "tick_volume", "spread"):
        if col in df_copy.columns:
            arr[col] = df_copy[col].values

    mock.copy_rates_from_pos.return_value = arr
    return mock


# ---------------------------------------------------------------------------
# fetch_bars()
# ---------------------------------------------------------------------------

class TestFetchBars:

    def test_returns_data_fetch_result(self):
        df = _make_bars(20)
        mt5_mock = _make_mt5_mock(df)
        result = fetch_bars("EURUSD", 20, _mt5=mt5_mock)
        assert isinstance(result, DataFetchResult)

    def test_bars_count_correct(self):
        df = _make_bars(50)
        mt5_mock = _make_mt5_mock(df)
        result = fetch_bars("EURUSD", 50, _mt5=mt5_mock)
        assert len(result.bars) == 50

    def test_raises_on_no_data(self):
        mock = MagicMock()
        mock.TIMEFRAME_M1 = 1
        mock.copy_rates_from_pos.return_value = None
        mock.last_error.return_value = (-1, "no data")
        with pytest.raises(RuntimeError, match="No data"):
            fetch_bars("EURUSD", _mt5=mock)

    def test_raises_on_empty_data(self):
        mock = MagicMock()
        mock.TIMEFRAME_M1 = 1
        mock.copy_rates_from_pos.return_value = []
        mock.last_error.return_value = (-1, "empty")
        with pytest.raises(RuntimeError, match="No data"):
            fetch_bars("EURUSD", _mt5=mock)

    def test_bars_sorted_ascending(self):
        df = _make_bars(10)
        # shuffle to simulate unordered MT5 output
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        mt5_mock = _make_mt5_mock(df)
        result = fetch_bars("EURUSD", 10, _mt5=mt5_mock)
        times = result.bars["time"].tolist()
        assert times == sorted(times)

    def test_time_column_is_datetime(self):
        df = _make_bars(5)
        mt5_mock = _make_mt5_mock(df)
        result = fetch_bars("EURUSD", 5, _mt5=mt5_mock)
        assert pd.api.types.is_datetime64_any_dtype(result.bars["time"])


# ---------------------------------------------------------------------------
# _remove_duplicates()
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:

    def test_no_duplicates(self):
        df = _make_bars(5)
        r = DataFetchResult(bars=df)
        r = _remove_duplicates(df, r)
        assert r.duplicates_removed == 0
        assert len(r.bars) == 5

    def test_removes_duplicate_timestamps(self):
        df = _make_bars(5)
        df_with_dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        r = DataFetchResult(bars=df_with_dup)
        r = _remove_duplicates(df_with_dup, r)
        assert r.duplicates_removed == 1
        assert len(r.bars) == 5

    def test_keeps_first_occurrence(self):
        df = _make_bars(3)
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        r = DataFetchResult(bars=df_with_dup)
        r = _remove_duplicates(df_with_dup, r)
        assert len(r.bars) == 3


# ---------------------------------------------------------------------------
# _detect_gaps()
# ---------------------------------------------------------------------------

class TestDetectGaps:

    def test_no_gaps_clean_data(self):
        df = _make_bars(10)
        r = DataFetchResult(bars=df)
        r = _detect_gaps(df, r)
        assert r.gaps_detected == []

    def test_detects_single_gap(self):
        """Insert a 3-minute jump between bar 4 and bar 5."""
        df = _make_bars(10)
        # Shift all bars from index 5 onward by 2 extra minutes
        df.loc[5:, "time"] = df.loc[5:, "time"] + pd.Timedelta(minutes=2)
        r = DataFetchResult(bars=df)
        r = _detect_gaps(df, r)
        assert len(r.gaps_detected) == 1

    def test_detects_multiple_gaps(self):
        df = _make_bars(10)
        # Create gaps at indices 3 and 7
        df.loc[3:, "time"] = df.loc[3:, "time"] + pd.Timedelta(minutes=5)
        df.loc[7:, "time"] = df.loc[7:, "time"] + pd.Timedelta(minutes=5)
        r = DataFetchResult(bars=df)
        r = _detect_gaps(df, r)
        assert len(r.gaps_detected) >= 2

    def test_no_gap_on_exactly_two_minutes(self):
        """A 2-minute gap equals exactly 2× M1 interval — not flagged (>2× required)."""
        df = _make_bars(5)
        df.loc[3:, "time"] = df.loc[3:, "time"] + pd.Timedelta(minutes=1)
        r = DataFetchResult(bars=df)
        r = _detect_gaps(df, r)
        assert r.gaps_detected == []

    def test_single_bar_no_gap(self):
        df = _make_bars(1)
        r = DataFetchResult(bars=df)
        r = _detect_gaps(df, r)
        assert r.gaps_detected == []


# ---------------------------------------------------------------------------
# _flag_zero_volume()
# ---------------------------------------------------------------------------

class TestFlagZeroVolume:

    def test_no_zero_volume(self):
        df = _make_bars(5, tick_volume=100)
        r = DataFetchResult(bars=df)
        r = _flag_zero_volume(df, r)
        assert r.zero_volume_bars == []

    def test_flags_zero_volume_bars(self):
        df = _make_bars(5, tick_volume=100)
        df.loc[2, "tick_volume"] = 0
        r = DataFetchResult(bars=df)
        r = _flag_zero_volume(df, r)
        assert len(r.zero_volume_bars) == 1

    def test_zero_volume_bars_NOT_removed(self):
        """Zero-volume bars stay in the DataFrame — only flagged."""
        df = _make_bars(5, tick_volume=100)
        df.loc[1, "tick_volume"] = 0
        df.loc[3, "tick_volume"] = 0
        r = DataFetchResult(bars=df)
        r = _flag_zero_volume(df, r)
        assert len(r.bars) == 5   # still 5 rows
        assert len(r.zero_volume_bars) == 2

    def test_multiple_zero_volume(self):
        df = _make_bars(5, tick_volume=0)  # all zero
        r = DataFetchResult(bars=df)
        r = _flag_zero_volume(df, r)
        assert len(r.zero_volume_bars) == 5


# ---------------------------------------------------------------------------
# validate_bar_row()
# ---------------------------------------------------------------------------

class TestValidateBarRow:

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {"tick_volume": 100, "high": 1.1010, "low": 1.1000,
                    "open": 1.1005, "close": 1.1007}
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_valid_bar(self):
        assert validate_bar_row(self._make_row()) is True

    def test_zero_volume_invalid(self):
        assert validate_bar_row(self._make_row(tick_volume=0)) is False

    def test_inverted_ohlc_invalid(self):
        assert validate_bar_row(self._make_row(high=1.0990, low=1.1010)) is False

    def test_zero_open_invalid(self):
        assert validate_bar_row(self._make_row(open=0)) is False

    def test_zero_close_invalid(self):
        assert validate_bar_row(self._make_row(close=0)) is False


# ---------------------------------------------------------------------------
# compute_average_spread()
# ---------------------------------------------------------------------------

class TestComputeAverageSpread:

    def test_returns_mean(self):
        df = _make_bars(20, spread=4)
        avg = compute_average_spread(df, window=20)
        assert abs(avg - 4.0) < 1e-9

    def test_uses_last_n_bars(self):
        df = _make_bars(30, spread=2)
        df.loc[25:, "spread"] = 10
        avg = compute_average_spread(df, window=5)
        assert avg > 2.0  # last 5 bars have high spread

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["spread"])
        assert compute_average_spread(df) == 0.0

    def test_missing_spread_column(self):
        df = _make_bars(5)
        df = df.drop(columns=["spread"])
        assert compute_average_spread(df) == 0.0
