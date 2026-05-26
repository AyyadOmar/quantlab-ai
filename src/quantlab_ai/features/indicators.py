from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicatorFactory:
    @staticmethod
    def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        close = data["close"]
        volume = data["volume"]
        high = data["high"]
        low = data["low"]

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
        data["macd_line"], data["macd_signal"], data["macd_hist"] = TechnicalIndicatorFactory._calculate_macd(close)
        bollinger_middle, bollinger_upper, bollinger_lower = TechnicalIndicatorFactory._calculate_bollinger_bands(close)
        band_width = (bollinger_upper - bollinger_lower).replace(0, np.nan)
        data["bollinger_bandwidth"] = band_width / bollinger_middle.replace(0, np.nan)
        data["bollinger_percent_b"] = (close - bollinger_lower) / band_width
        data["atr_14"] = TechnicalIndicatorFactory._calculate_atr(high, low, close, period=14) / close.replace(0, np.nan)
        obv = TechnicalIndicatorFactory._calculate_obv(close, volume)
        data["obv_zscore_20"] = TechnicalIndicatorFactory._rolling_zscore(obv, window=20)

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

    @staticmethod
    def _calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    @staticmethod
    def _calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        middle_band = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        return middle_band, upper_band, lower_band

    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0.0)
        return (direction * volume).cumsum()

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)
