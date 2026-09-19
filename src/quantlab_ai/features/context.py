"""Context joins with explicit historical availability boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_COLUMNS = ["technology_return_1d", "technology_return_5d", "relative_to_technology_5d",
                  "vix_level", "vix_change_5d", "long_yield_level", "long_yield_change_5d", "yield_curve_proxy"]
EARNINGS_COLUMNS = ["earnings_days_since_filing", "earnings_recent_5d", "earnings_history_known"]


def add_market_context(features: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = features.sort_values("date").copy()
    result["date"] = pd.to_datetime(result["date"])
    for name in ["technology", "volatility", "long_yield", "short_yield"]:
        source = histories[name][["date", "close"]].copy()
        source["date"] = pd.to_datetime(source["date"])
        source = source.sort_values("date")
        if source.date.duplicated().any() or not np.isfinite(source.close).all():
            raise ValueError(f"Invalid {name} context history.")
        if name == "technology":
            source["technology_return_1d"] = source.close.pct_change(fill_method=None)
            source["technology_return_5d"] = source.close.pct_change(5, fill_method=None)
        elif name == "volatility":
            source["vix_level"] = source.close
            source["vix_change_5d"] = source.close.pct_change(5, fill_method=None)
        elif name == "long_yield":
            source["long_yield_level"] = source.close
            source["long_yield_change_5d"] = source.close.diff(5)
        else:
            source["short_yield_level"] = source.close
        source_date = f"{name}_source_date"
        source = source.drop(columns="close").rename(columns={"date": source_date})
        result = pd.merge_asof(result, source, left_on="date", right_on=source_date,
                               direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=7))
    result["relative_to_technology_5d"] = result.return_5d - result.technology_return_5d
    result["yield_curve_proxy"] = result.long_yield_level - result.short_yield_level
    if not np.isfinite(result[MARKET_COLUMNS].to_numpy()).all():
        raise ValueError("Missing, stale, or incomplete market context; cannot silently shorten the baseline sample.")
    return result


def add_earnings_context(features: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    result = features.sort_values("date").copy()
    result["date"] = pd.to_datetime(result["date"])
    if events is None:
        # ETF benchmarks do not have company quarterly earnings.
        for column in EARNINGS_COLUMNS:
            result[column] = 0.0
        return result
    if events.empty:
        raise ValueError("Company filing history is empty; absence must not be interpreted as no earnings events.")
    data = events[["available_date", "filing_date"]].copy()
    data["available_date"] = pd.to_datetime(data["available_date"])
    data["filing_date"] = pd.to_datetime(data["filing_date"])
    if (data.available_date <= data.filing_date).any():
        raise ValueError("An earnings filing must not be usable until after its filing date.")
    data = data.sort_values(["available_date", "filing_date"]).drop_duplicates("available_date", keep="last")
    result = pd.merge_asof(result, data, left_on="date", right_on="available_date", direction="backward")
    age = (result.date - result.filing_date).dt.days
    result["earnings_history_known"] = age.notna().astype(float)
    result["earnings_days_since_filing"] = age.clip(upper=180).fillna(180).astype(float)
    result["earnings_recent_5d"] = (age.notna() & age.le(5)).astype(float)
    return result
