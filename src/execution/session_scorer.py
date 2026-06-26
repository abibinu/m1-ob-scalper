"""
Session Strength Scorer
========================
Rev 3 Enhancement: Filters entries during historically low-volatility hours
even within the configured session window.

For a high-frequency scalping bot, entering during a "dead zone" (e.g. 12:00–13:00 UTC
lull between London and NY sessions) produces low-expectancy signals that increase
drawdown without proportional return.

This module computes a per-UTC-hour volatility profile from historical bar data
and exposes a single filter: ``get_hour_strength(hour_utc)`` → float [0.0–1.0].

Usage:
    scorer = SessionScorer(bars)
    if scorer.get_hour_strength(now_utc.hour) < cfg.session_strength_min:
        skip_entry()
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.logger import get_logger

log = get_logger(__name__)


class SessionScorer:
    """
    Computes a volatility strength score (0.0–1.0) for each UTC hour based on
    historical M1 bar data.

    Score = (hour_avg_range / global_max_hour_avg_range)

    Hours with no data default to 0.5 (neutral, pass-through).

    Args:
        bars:        Historical M1 OHLCV DataFrame with UTC 'time' column.
        min_bars:    Minimum bars per hour to consider it statistically valid.
                     Hours below this threshold get a neutral 0.5 score.
    """

    def __init__(self, bars: pd.DataFrame, min_bars: int = 20) -> None:
        self._scores: Dict[int, float] = {}
        self._build_profile(bars, min_bars)

    def _build_profile(self, bars: pd.DataFrame, min_bars: int) -> None:
        """Compute per-hour average range and normalise to [0, 1]."""
        if bars.empty or "time" not in bars.columns:
            log.warning("SessionScorer: empty bars or missing 'time' column — all hours neutral")
            return

        df = bars.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], utc=True)
        elif df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")

        df["bar_range"] = df["high"] - df["low"]
        df["hour_utc"] = df["time"].dt.hour

        hour_stats = df.groupby("hour_utc").agg(
            avg_range=("bar_range", "mean"),
            count=("bar_range", "count"),
        )

        # Zero-out hours with insufficient data
        hour_stats.loc[hour_stats["count"] < min_bars, "avg_range"] = 0.0

        max_range = hour_stats["avg_range"].max()
        if max_range <= 0:
            log.warning("SessionScorer: all hours have zero range — scores default to 0.5")
            return

        for hour, row in hour_stats.iterrows():
            self._scores[int(hour)] = float(row["avg_range"] / max_range)

        log.info(
            "SessionScorer built: %d hour buckets. Peak hour=%d (score=1.0)",
            len(self._scores),
            hour_stats["avg_range"].idxmax(),
        )

    def get_hour_strength(self, hour_utc: int) -> float:
        """
        Return the normalised volatility strength for the given UTC hour.

        Returns:
            float in [0.0, 1.0].
            0.5 if the hour has no historical data (neutral pass-through).
        """
        return self._scores.get(hour_utc, 0.5)

    def summary(self) -> Dict[int, float]:
        """Return the full hour → score mapping for logging/debugging."""
        return dict(sorted(self._scores.items()))
