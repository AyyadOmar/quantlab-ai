from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

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
        if self.settings.use_cached_data:
            path = self.settings.raw_data_dir / f"{ticker.lower()}_raw.csv"
            data = pd.read_csv(path, parse_dates=["date"])
            data = data.loc[(data["date"] >= pd.Timestamp(start_date)) & (data["date"] < pd.Timestamp(end_date))].copy()
            if data.empty:
                raise ValueError(f"No cached prices for {ticker} in the requested period.")
            return self.completed_sessions(data)
        self.logger.info("Downloading historical data for %s", ticker)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No market data returned for ticker {ticker}")

        data = data.reset_index()
        data.columns = [self._normalize_column_name(column) for column in data.columns]
        data["ticker"] = ticker
        return self.completed_sessions(data)

    @staticmethod
    def completed_sessions(data: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
        # Conservative cutoff for this project's US-equity universe. Waiting until
        # 16:15 ET also avoids treating an early-close session as final too soon.
        current = now or datetime.now(ZoneInfo("America/New_York"))
        current = current.astimezone(ZoneInfo("America/New_York"))
        dates = pd.to_datetime(data["date"]).dt.date
        mask = dates < current.date()
        if current.time() >= time(16, 15):
            mask |= dates == current.date()
        result = data.loc[mask].sort_values("date").reset_index(drop=True)
        if result.empty:
            raise ValueError("No completed sessions are available.")
        return result

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
