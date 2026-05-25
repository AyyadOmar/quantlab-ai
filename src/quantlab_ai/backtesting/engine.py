from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict


@dataclass
class BacktestEngine:
    settings: Settings

    def run(self, predictions: pd.DataFrame, model_name: str, ticker: str) -> BacktestResult:
        frame = predictions.copy().reset_index(drop=True)
        fee_rate = self.settings.trading_fee_bps / 10_000

        frame["strategy_return"] = np.where(frame["signal"] == 1, frame["next_day_return"] - fee_rate, 0.0)
        frame["equity_curve"] = (1 + frame["strategy_return"]).cumprod()
        frame["benchmark_curve"] = (1 + frame["next_day_return"].fillna(0)).cumprod()
        frame["drawdown"] = frame["equity_curve"] / frame["equity_curve"].cummax() - 1

        trades = frame.loc[frame["signal"] == 1, ["date", "close", "prob_up", "next_day_return", "strategy_return"]].copy()
        metrics = self._metrics(frame)

        trades_path = self.settings.backtests_dir / f"{ticker.lower()}_{model_name}_trades.csv"
        summary_path = self.settings.backtests_dir / f"{ticker.lower()}_{model_name}_summary.json"
        trades.to_csv(trades_path, index=False)
        summary_path.write_text(json.dumps(metrics, indent=2))

        return BacktestResult(
            trades=trades,
            equity_curve=frame[["date", "equity_curve", "benchmark_curve", "drawdown"]],
            metrics=metrics,
        )

    def _metrics(self, frame: pd.DataFrame) -> dict:
        returns = frame["strategy_return"].fillna(0.0)
        sharpe_denominator = returns.std(ddof=0)
        sharpe_ratio = 0.0
        if sharpe_denominator > 0:
            sharpe_ratio = ((returns.mean() * 252) - self.settings.risk_free_rate) / (sharpe_denominator * np.sqrt(252))

        winning_trades = frame.loc[frame["signal"] == 1, "strategy_return"]
        win_rate = float((winning_trades > 0).mean()) if not winning_trades.empty else 0.0

        return {
            "total_return": float(frame["equity_curve"].iloc[-1] - 1),
            "benchmark_return": float(frame["benchmark_curve"].iloc[-1] - 1),
            "max_drawdown": float(frame["drawdown"].min()),
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": win_rate,
            "trade_count": int((frame["signal"] == 1).sum()),
        }
