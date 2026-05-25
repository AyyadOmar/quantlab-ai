from __future__ import annotations

from dataclasses import dataclass

from .backtesting.engine import BacktestEngine
from .config import Settings
from .data.loader import MarketDataLoader
from .data.repository import ExperimentRepository
from .features.builder import FeatureBuilder
from .models.classical import ClassicalModelTrainer
from .models.lstm import LSTMTrainer
from .utils.logging import get_logger
from .visualization.plots import PlotService


@dataclass
class PipelineRunner:
    settings: Settings

    def __post_init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.loader = MarketDataLoader(self.settings)
        self.repository = ExperimentRepository(self.settings)
        self.builder = FeatureBuilder(self.settings)
        self.backtester = BacktestEngine(self.settings)
        self.plot_service = PlotService(self.settings)
        self.repository.initialize()

    def run(self, ticker: str, start_date: str, end_date: str, model_name: str) -> dict:
        raw_data = self.loader.download(ticker, start_date, end_date)
        self.loader.cache_to_csv(ticker, raw_data)
        features = self.builder.build(raw_data, ticker)

        if model_name == "lstm":
            trainer = LSTMTrainer(self.settings)
        else:
            trainer = ClassicalModelTrainer(self.settings, model_name=model_name)

        artifacts = trainer.train(features)
        backtest = self.backtester.run(artifacts.predictions, artifacts.model_name, ticker)

        self.plot_service.plot_candlestick(raw_data, ticker)
        self.plot_service.plot_equity_curve(backtest.equity_curve, ticker, artifacts.model_name)
        self.plot_service.plot_confusion_matrix(
            artifacts.metrics["confusion_matrix"],
            ticker,
            artifacts.model_name,
        )
        self.plot_service.plot_probability_distribution(artifacts.predictions, ticker, artifacts.model_name)

        combined_metrics = {
            "classification": artifacts.metrics,
            "backtest": backtest.metrics,
        }
        self.repository.log_experiment(
            ticker=ticker,
            model_name=artifacts.model_name,
            start_date=start_date,
            end_date=end_date,
            metrics=combined_metrics,
            artifact_path=artifacts.artifact_path,
        )
        self.logger.info("Pipeline complete for %s with %s", ticker, artifacts.model_name)
        return combined_metrics
