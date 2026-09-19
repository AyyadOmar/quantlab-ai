from __future__ import annotations

import json
import hashlib
import platform
from dataclasses import asdict
from importlib.metadata import version

import pandas as pd
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean
from pathlib import Path

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
        raw_data, context_data = self._load_market_and_context(ticker, start_date, end_date)
        features = self.builder.build(raw_data, ticker, context_data=context_data)

        if model_name == "lstm":
            trainer = LSTMTrainer(self.settings)
        else:
            trainer = ClassicalModelTrainer(self.settings, model_name=model_name)

        artifacts = trainer.train(features)
        threshold_report = artifacts.threshold_report
        backtest = self.backtester.run(artifacts.predictions, artifacts.model_name, ticker)
        stem = f"{ticker.lower()}_{artifacts.model_name}"
        artifacts.predictions.to_csv(self.settings.backtests_dir / f"{stem}_predictions.csv", index=False)
        (self.settings.backtests_dir / f"{stem}_threshold_sweep.json").write_text(json.dumps(threshold_report, indent=2))

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
            "protocol": self.settings.protocol,
            "target": "next_session_close > next_session_open",
            "execution": "Signal after completed close; enter next open, exit that close; long or cash.",
            "reproducibility": {
                "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self.settings).items()},
                "raw_data_sha256": hashlib.sha256(raw_data.to_csv(index=False).encode()).hexdigest(),
                "context_data_sha256": hashlib.sha256(context_data.to_csv(index=False).encode()).hexdigest() if context_data is not None else None,
                "python": platform.python_version(),
                "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "torch"]},
            },
            "classification": artifacts.metrics,
            "cross_validation": artifacts.cross_validation,
            "threshold_sweep": threshold_report,
            "backtest": backtest.metrics,
            "benchmarks": backtest.benchmark_metrics,
        }
        (self.settings.backtests_dir / f"{stem}_experiment.json").write_text(json.dumps(combined_metrics, indent=2))
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

    def run_batch(self, tickers: list[str], start_date: str, end_date: str, model_name: str) -> dict:
        experiments: list[dict] = []
        failures: list[dict] = []
        for ticker in tickers:
            self.logger.info("Starting batch experiment for %s with %s", ticker, model_name)
            try:
                metrics = self.run(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    model_name=model_name,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.exception("Batch experiment failed for %s with %s", ticker, model_name)
                failures.append(
                    {
                        "ticker": ticker,
                        "model_name": model_name,
                        "error": str(error),
                    }
                )
                continue

            experiments.append(
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "classification_accuracy": float(metrics["classification"]["accuracy"]),
                    "classification_precision": float(metrics["classification"]["precision"]),
                    "classification_recall": float(metrics["classification"]["recall"]),
                    "classification_f1": float(metrics["classification"]["f1"]),
                    "classification_roc_auc": float(metrics["classification"]["roc_auc"]),
                    "mean_cv_accuracy": float(metrics["cross_validation"]["summary"]["mean_accuracy"]),
                    "mean_cv_precision": float(metrics["cross_validation"]["summary"]["mean_precision"]),
                    "mean_cv_recall": float(metrics["cross_validation"]["summary"]["mean_recall"]),
                    "mean_cv_f1": float(metrics["cross_validation"]["summary"]["mean_f1"]),
                    "mean_cv_roc_auc": float(metrics["cross_validation"]["summary"]["mean_roc_auc"]),
                    "latest_validation_threshold": float(metrics["threshold_sweep"]["latest_validation_threshold"]),
                    "strategy_return": float(metrics["backtest"]["total_return"]),
                    "sharpe_ratio": float(metrics["backtest"]["sharpe_ratio"]),
                    "max_drawdown": float(metrics["backtest"]["max_drawdown"]),
                    "trade_count": int(metrics["backtest"]["trade_count"]),
                    "buy_and_hold_return": float(metrics["benchmarks"]["buy_and_hold"]["total_return"]),
                }
            )

        aggregate = {
            "model_name": model_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "experiment_count": len(experiments),
            "failure_count": len(failures),
            "mean_strategy_return": float(mean([row["strategy_return"] for row in experiments])) if experiments else 0.0,
            "mean_sharpe_ratio": float(mean([row["sharpe_ratio"] for row in experiments])) if experiments else 0.0,
            "mean_cv_accuracy": float(mean([row["mean_cv_accuracy"] for row in experiments])) if experiments else 0.0,
            "mean_cv_roc_auc": float(mean([row["mean_cv_roc_auc"] for row in experiments])) if experiments else 0.0,
        }

        report = {
            "aggregate": aggregate,
            "experiments": experiments,
            "failures": failures,
        }

        report_path = self.settings.backtests_dir / f"batch_{model_name}_leaderboard.json"
        report_path.write_text(json.dumps(report, indent=2))
        return report

    def run_baselines(self, tickers: list[str], start_date: str, end_date: str) -> dict:
        rows, failures = [], []
        for ticker in tickers:
            for model in ["logistic_regression", "xgboost"]:
                try:
                    result = self.run(ticker, start_date, end_date, model)
                    classification = result["classification"]
                    baselines = result["cross_validation"]["classification_baselines"]
                    trading = result["backtest"]
                    rows.append({"ticker": ticker, "model": model,
                                 "accuracy": classification["accuracy"],
                                 "always_up_accuracy": baselines["always_up"]["accuracy"],
                                 "roc_auc": classification["roc_auc"],
                                 "brier_score": classification["brier_score"],
                                 "prevalence_brier_score": baselines["training_prevalence"]["brier_score"],
                                 "log_loss": classification["log_loss"],
                                 "strategy_return": trading["total_return"],
                                 "always_long_return": result["benchmarks"]["always_long"]["total_return"],
                                 "momentum_return": result["benchmarks"]["momentum"]["total_return"],
                                 "buy_and_hold_return": result["benchmarks"]["buy_and_hold"]["total_return"],
                                 "max_drawdown": trading["max_drawdown"],
                                 "sharpe_ratio": trading["sharpe_ratio"],
                                 "active_session_fraction": trading["active_session_fraction"],
                                 "trade_count": trading["trade_count"],
                                 "evaluation_sessions": trading["evaluation_sessions"],
                                 "start_date": trading["start_date"], "end_date": trading["end_date"]})
                except Exception as error:
                    self.logger.exception("Baseline failed for %s %s", ticker, model)
                    failures.append({"ticker": ticker, "model": model, "error": str(error)})
        report = {"protocol": self.settings.protocol, "experiments": rows, "failures": failures}
        (self.settings.backtests_dir / "baseline_comparison.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame(rows).to_csv(self.settings.backtests_dir / "baseline_comparison.csv", index=False)
        lines = ["# Corrected baseline comparison", "",
                 "Signal after close; next-session open-to-close long/cash trades. Thresholds selected on earlier validation only.", "",
                 f"Costs per side: {self.settings.trading_fee_bps:g} bps fee + {self.settings.slippage_bps:g} bps slippage. Cash earns zero. Fixed model settings; no search on test results. These costs are assumptions, not measured fills.", "",
                 "Historical walk-forward results are a research baseline, not an untouched prospective trial. The dates were used in earlier project research.", "",
                 "Buy-and-hold includes overnight exposure and uses adjusted prices; always-long trades only open-to-close, matching the model's holding window. Returns are cumulative, not annualized.", "",
                 "| Ticker | Model | Accuracy | Always up | AUC | Brier ↓ | Prior Brier ↓ | Net return | Always long | Momentum | Buy & hold | Max drawdown | Trades |", 
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['ticker']} | {row['model']} | {row['accuracy']:.1%} | {row['always_up_accuracy']:.1%} | {row['roc_auc']:.3f} | {row['brier_score']:.3f} | {row['prevalence_brier_score']:.3f} | {row['strategy_return']:.1%} | {row['always_long_return']:.1%} | {row['momentum_return']:.1%} | {row['buy_and_hold_return']:.1%} | {row['max_drawdown']:.1%} | {row['trade_count']} |")
        lines += ["", "Each experiment JSON records exact windows, costs, data hashes and package versions. Daily CSVs contain every signal and execution price. Brier and log loss measure probability quality; lower is better. Very few trades do not establish a reliable strategy."]
        if rows:
            lines += ["", f"Execution coverage: {rows[0]['start_date']} through {rows[0]['end_date']} ({rows[0]['evaluation_sessions']} sessions for the first experiment)."]
        if failures:
            lines += ["", "Failures: " + json.dumps(failures)]
        (self.settings.backtests_dir / "baseline_comparison.md").write_text("\n".join(lines) + "\n")
        return report

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

                next_open = float(raw_data.loc[index + 1, "open"])
                next_close = float(raw_data.loc[index + 1, "close"])
                actual_direction = int(next_close > next_open)
                actual_return = float(next_close / next_open - 1)
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
        if not self.settings.use_cached_data:
            self.loader.cache_to_csv(ticker, raw_data)
        context_data = None
        if ticker != self.settings.market_context_ticker:
            context_data = self.loader.download(self.settings.market_context_ticker, start_date, end_date)
            if not self.settings.use_cached_data:
                self.loader.cache_to_csv(self.settings.market_context_ticker, context_data)
        return raw_data, context_data
