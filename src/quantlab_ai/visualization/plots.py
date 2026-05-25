from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import seaborn as sns

from ..config import Settings


@dataclass
class PlotService:
    settings: Settings

    def plot_equity_curve(self, equity_curve: pd.DataFrame, ticker: str, model_name: str) -> str:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(pd.to_datetime(equity_curve["date"]), equity_curve["equity_curve"], label="Strategy")
        axis.plot(pd.to_datetime(equity_curve["date"]), equity_curve["benchmark_curve"], label="Buy & Hold")
        if "always_long_curve" in equity_curve:
            axis.plot(pd.to_datetime(equity_curve["date"]), equity_curve["always_long_curve"], label="Always Long", alpha=0.8)
        if "momentum_curve" in equity_curve:
            axis.plot(pd.to_datetime(equity_curve["date"]), equity_curve["momentum_curve"], label="Momentum", alpha=0.8)
        axis.set_title(f"{ticker} Strategy vs Benchmarks")
        axis.set_ylabel("Growth of $1")
        axis.legend()
        axis.grid(alpha=0.3)

        output_path = self.settings.plots_dir / f"{ticker.lower()}_{model_name}_equity_curve.png"
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return str(output_path)

    def plot_confusion_matrix(self, confusion: list[list[int]], ticker: str, model_name: str) -> str:
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=axis)
        axis.set_title(f"{ticker} {model_name} Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

        output_path = self.settings.plots_dir / f"{ticker.lower()}_{model_name}_confusion_matrix.png"
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return str(output_path)

    def plot_probability_distribution(self, predictions: pd.DataFrame, ticker: str, model_name: str) -> str:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.hist(predictions["prob_up"], bins=20, color="#1f77b4", alpha=0.8)
        axis.set_title(f"{ticker} Probability of Upward Movement")
        axis.set_xlabel("Predicted Probability")
        axis.set_ylabel("Frequency")

        output_path = self.settings.plots_dir / f"{ticker.lower()}_{model_name}_probabilities.png"
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return str(output_path)

    def plot_candlestick(self, raw_data: pd.DataFrame, ticker: str) -> str:
        chart_data = raw_data.copy()
        chart_data["date"] = pd.to_datetime(chart_data["date"])
        chart_data = chart_data.set_index("date")[["open", "high", "low", "close", "volume"]]

        output_path = self.settings.plots_dir / f"{ticker.lower()}_candlestick.png"
        mpf.plot(chart_data.tail(120), type="candle", volume=True, style="yahoo", savefig=output_path)
        return str(output_path)
