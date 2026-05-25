from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from ..config import Settings
from ..utils.logging import get_logger


@dataclass
class MarketDataLoader:
    settings: Settings

    def __post_init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def download(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.logger.info("Downloading historical data for %s", ticker)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No market data returned for ticker {ticker}")

        data = data.reset_index()
        data.columns = [self._normalize_column_name(column) for column in data.columns]
        data["ticker"] = ticker
        return data

    def cache_to_csv(self, ticker: str, data: pd.DataFrame) -> str:
        output_path = self.settings.raw_data_dir / f"{ticker.lower()}_raw.csv"
        data.to_csv(output_path, index=False)
        self.logger.info("Raw data saved to %s", output_path)
        return str(output_path)

    @staticmethod
    def _normalize_column_name(column: object) -> str:
        if isinstance(column, tuple):
            primary = str(column[0]) if column else "column"
            column_name = primary
        else:
            column_name = str(column)
        return column_name.lower().replace(" ", "_")
