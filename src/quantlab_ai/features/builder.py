from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings
from .indicators import TechnicalIndicatorFactory


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "sma_10",
    "sma_20",
    "ema_10",
    "ema_20",
    "momentum_10",
    "volatility_10",
    "volume_ratio",
    "rsi_14",
    "intraday_range",
    "open_close_gap",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    "atr_14",
    "obv_zscore_20",
    "market_return_1d",
    "market_return_5d",
    "relative_strength_5d",
    "rolling_beta_20",
    "rolling_correlation_20",
]


@dataclass
class FeatureBuilder:
    settings: Settings

    def build(self, raw_data: pd.DataFrame, ticker: str, context_data: pd.DataFrame | None = None) -> pd.DataFrame:
        engineered = self._prepare_features(raw_data, context_data)
        engineered["target"] = (engineered["close"].shift(-1) > engineered["close"]).astype(int)
        engineered["next_day_return"] = engineered["close"].shift(-1) / engineered["close"] - 1
        engineered = self._ensure_feature_columns(engineered)
        engineered[FEATURE_COLUMNS] = engineered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        engineered = engineered.dropna().reset_index(drop=True)

        output_path = self.settings.processed_data_dir / f"{ticker.lower()}_features.csv"
        engineered.to_csv(output_path, index=False)
        return engineered

    def build_for_inference(
        self,
        raw_data: pd.DataFrame,
        context_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        engineered = self._prepare_features(raw_data, context_data)
        engineered = self._ensure_feature_columns(engineered)
        engineered[FEATURE_COLUMNS] = engineered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        engineered = engineered.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        return engineered

    def _prepare_features(self, raw_data: pd.DataFrame, context_data: pd.DataFrame | None = None) -> pd.DataFrame:
        engineered = TechnicalIndicatorFactory.add_indicators(raw_data)
        if context_data is not None:
            engineered = self._merge_market_context(engineered, context_data)
        else:
            engineered = self._add_self_market_context(engineered)
        return engineered

    def _merge_market_context(self, asset_data: pd.DataFrame, context_data: pd.DataFrame) -> pd.DataFrame:
        context = context_data[["date", "close"]].copy()
        context = context.rename(columns={"close": "market_close"})
        context["market_return_1d"] = context["market_close"].pct_change()
        context["market_return_5d"] = context["market_close"].pct_change(5)

        merged = asset_data.merge(context, on="date", how="left")
        merged["relative_strength_5d"] = merged["return_5d"] - merged["market_return_5d"]
        merged["rolling_beta_20"] = self._rolling_beta(
            merged["return_1d"],
            merged["market_return_1d"],
            window=20,
        )
        merged["rolling_correlation_20"] = merged["return_1d"].rolling(window=20).corr(merged["market_return_1d"])
        return merged

    def _add_self_market_context(self, asset_data: pd.DataFrame) -> pd.DataFrame:
        data = asset_data.copy()
        data["market_return_1d"] = data["return_1d"]
        data["market_return_5d"] = data["return_5d"]
        data["relative_strength_5d"] = 0.0
        data["rolling_beta_20"] = 1.0
        data["rolling_correlation_20"] = 1.0
        return data

    def _ensure_feature_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        default_columns = {
            "market_return_1d": data.get("return_1d", pd.Series(0.0, index=data.index)),
            "market_return_5d": data.get("return_5d", pd.Series(0.0, index=data.index)),
            "relative_strength_5d": pd.Series(0.0, index=data.index),
            "rolling_beta_20": pd.Series(1.0, index=data.index),
            "rolling_correlation_20": pd.Series(1.0, index=data.index),
        }
        for column, default_value in default_columns.items():
            if column not in data.columns:
                data[column] = default_value
        return data

    @staticmethod
    def _rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
        covariance = asset_returns.rolling(window=window).cov(market_returns)
        variance = market_returns.rolling(window=window).var()
        return covariance / variance.replace(0, np.nan)
