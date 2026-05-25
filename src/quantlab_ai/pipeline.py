from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

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
        context_data = None
        if ticker != self.settings.market_context_ticker:
            context_data = self.loader.download(self.settings.market_context_ticker, start_date, end_date)
            self.loader.cache_to_csv(self.settings.market_context_ticker, context_data)
        features = self.builder.build(raw_data, ticker, context_data=context_data)

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

        cross_validation_path = self.settings.backtests_dir / f"{ticker.lower()}_{artifacts.model_name}_cross_validation.json"
        cross_validation_path.write_text(json.dumps(artifacts.cross_validation, indent=2))

        combined_metrics = {
            "classification": artifacts.metrics,
            "cross_validation": artifacts.cross_validation,
            "backtest": backtest.metrics,
            "benchmarks": backtest.benchmark_metrics,
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

    def predict_latest(self, ticker: str, start_date: str, end_date: str, model_name: str) -> list[dict]:
        raw_data, context_data = self._load_market_and_context(ticker, start_date, end_date)
        training_features = self.builder.build(raw_data, ticker, context_data=context_data)
        inference_features = self.builder.build_for_inference(raw_data, context_data=context_data)

        model_names = ["logistic_regression", "random_forest", "xgboost", "lstm"] if model_name == "all" else [model_name]
        outputs: list[dict] = []

        for current_model in model_names:
            trainer = LSTMTrainer(self.settings) if current_model == "lstm" else ClassicalModelTrainer(self.settings, model_name=current_model)
            latest_prediction = trainer.predict_latest(training_features, inference_features)
            self.repository.log_live_prediction(
                ticker=ticker,
                model_name=latest_prediction.model_name,
                as_of_date=latest_prediction.as_of_date,
                prediction=latest_prediction.prediction,
                prob_up=latest_prediction.prob_up,
                signal=latest_prediction.signal,
                artifact_path=latest_prediction.artifact_path,
            )
            outputs.append(
                {
                    "ticker": ticker,
                    "model_name": latest_prediction.model_name,
                    "as_of_date": latest_prediction.as_of_date,
                    "prediction": latest_prediction.prediction,
                    "prob_up": latest_prediction.prob_up,
                    "signal": latest_prediction.signal,
                }
            )

        return outputs

    def resolve_live_predictions(self, ticker: str | None = None) -> dict:
        pending_predictions = self.repository.get_pending_live_predictions(ticker=ticker)
        if not pending_predictions:
            return {"resolved": 0, "summary": self.repository.summarize_live_predictions(ticker=ticker)}

        grouped_by_ticker: dict[str, list[dict]] = {}
        for row in pending_predictions:
            grouped_by_ticker.setdefault(row["ticker"], []).append(row)

        resolved_count = 0
        for current_ticker, predictions in grouped_by_ticker.items():
            earliest_date = min(str(row["as_of_date"]).split(" ")[0] for row in predictions)
            end_date = (date.today() + timedelta(days=5)).isoformat()
            raw_data = self.loader.download(current_ticker, earliest_date, end_date)
            if raw_data.empty:
                continue

            raw_data = raw_data.sort_values("date").reset_index(drop=True)
            for row in predictions:
                as_of_date = str(row["as_of_date"]).split(" ")[0]
                matching_index = raw_data.index[raw_data["date"].astype(str).str.startswith(as_of_date)].tolist()
                if not matching_index:
                    continue
                index = matching_index[0]
                if index + 1 >= len(raw_data):
                    continue

                current_close = float(raw_data.loc[index, "close"])
                next_close = float(raw_data.loc[index + 1, "close"])
                actual_direction = int(next_close > current_close)
                actual_return = float(next_close / current_close - 1)
                self.repository.resolve_live_prediction(
                    prediction_id=int(row["id"]),
                    actual_direction=actual_direction,
                    actual_return=actual_return,
                )
                resolved_count += 1

        return {
            "resolved": resolved_count,
            "summary": self.repository.summarize_live_predictions(ticker=ticker),
        }

    def live_prediction_summary(self, ticker: str | None = None) -> dict:
        return self.repository.summarize_live_predictions(ticker=ticker)

    def _load_market_and_context(self, ticker: str, start_date: str, end_date: str) -> tuple:
        raw_data = self.loader.download(ticker, start_date, end_date)
        self.loader.cache_to_csv(ticker, raw_data)
        context_data = None
        if ticker != self.settings.market_context_ticker:
            context_data = self.loader.download(self.settings.market_context_ticker, start_date, end_date)
            self.loader.cache_to_csv(self.settings.market_context_ticker, context_data)
        return raw_data, context_data
