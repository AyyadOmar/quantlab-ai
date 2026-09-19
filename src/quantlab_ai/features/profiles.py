"""Predeclared feature sets; no selection is fitted on evaluation outcomes."""
from __future__ import annotations

import pandas as pd

PRICE_LEVEL_REPLACEMENTS = {
    "sma_10": "distance_sma_10", "sma_20": "distance_sma_20",
    "ema_10": "distance_ema_10", "ema_20": "distance_ema_20",
    "macd_line": "relative_macd_line", "macd_signal": "relative_macd_signal",
    "macd_hist": "relative_macd_hist",
}
COMPACT_COLUMNS = [
    "return_1d", "return_5d", "distance_sma_20", "volatility_10",
    "volume_ratio", "rsi_14", "intraday_range", "open_close_gap",
    "relative_macd_hist", "obv_zscore_20", "market_return_1d",
    "relative_strength_5d", "rolling_beta_20", "rolling_correlation_20",
]


def feature_columns(profile: str) -> list[str]:
    from .builder import FEATURE_COLUMNS
    from .context import MARKET_COLUMNS, EARNINGS_COLUMNS
    from .earnings import SURPRISE_COLUMNS, SCHEDULE_COLUMNS
    for suffix, extra in [("_surprise", SURPRISE_COLUMNS), ("_schedule", SCHEDULE_COLUMNS),
                          ("_events", SURPRISE_COLUMNS + SCHEDULE_COLUMNS)]:
        if profile.endswith(suffix):
            base = profile[:-len(suffix)]
            if base not in {"relative", "compact"}:
                raise ValueError(f"Unknown earnings profile: {profile}")
            return feature_columns(base) + extra
    for suffix, extra in [("_market", MARKET_COLUMNS), ("_earnings", EARNINGS_COLUMNS),
                          ("_context", MARKET_COLUMNS + EARNINGS_COLUMNS)]:
        if profile.endswith(suffix):
            base = profile[:-len(suffix)]
            if base not in {"relative", "compact"}:
                raise ValueError(f"Unknown context feature profile: {profile}")
            return feature_columns(base) + extra
    if profile == "original":
        return list(FEATURE_COLUMNS)
    if profile == "relative":
        return [PRICE_LEVEL_REPLACEMENTS.get(column, column) for column in FEATURE_COLUMNS]
    if profile == "compact":
        return list(COMPACT_COLUMNS)
    raise ValueError(f"Unknown feature profile: {profile}")


def add_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, destination in PRICE_LEVEL_REPLACEMENTS.items():
        result[destination] = (result["close"] / result[source] - 1 if source.startswith(("sma_", "ema_"))
                               else result[source] / result["close"])
    return result
