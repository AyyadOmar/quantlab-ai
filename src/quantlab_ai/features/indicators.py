from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicatorFactory:
    @staticmethod
    def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        close = data["close"]
        volume = data["volume"]

        data["return_1d"] = close.pct_change()
        data["return_5d"] = close.pct_change(5)
        data["sma_10"] = close.rolling(window=10).mean()
        data["sma_20"] = close.rolling(window=20).mean()
        data["ema_10"] = close.ewm(span=10, adjust=False).mean()
        data["ema_20"] = close.ewm(span=20, adjust=False).mean()
        data["momentum_10"] = close / close.shift(10) - 1
        data["volatility_10"] = data["return_1d"].rolling(window=10).std() * np.sqrt(252)
        data["volume_sma_10"] = volume.rolling(window=10).mean()
        data["volume_ratio"] = volume / data["volume_sma_10"]
        data["rsi_14"] = TechnicalIndicatorFactory._calculate_rsi(close, period=14)
        data["intraday_range"] = (data["high"] - data["low"]) / data["close"]
        data["open_close_gap"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)

        return data

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        average_gain = gains.rolling(window=period).mean()
        average_loss = losses.rolling(window=period).mean()
        rs = average_gain / average_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
