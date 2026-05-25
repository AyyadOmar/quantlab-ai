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
]


@dataclass
class FeatureBuilder:
    settings: Settings

    def build(self, raw_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        engineered = TechnicalIndicatorFactory.add_indicators(raw_data)
        engineered["target"] = (engineered["close"].shift(-1) > engineered["close"]).astype(int)
        engineered["next_day_return"] = engineered["close"].shift(-1) / engineered["close"] - 1
        engineered[FEATURE_COLUMNS] = engineered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        engineered = engineered.dropna().reset_index(drop=True)

        output_path = self.settings.processed_data_dir / f"{ticker.lower()}_features.csv"
        engineered.to_csv(output_path, index=False)
        return engineered
