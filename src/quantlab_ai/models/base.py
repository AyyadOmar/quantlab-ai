from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelArtifacts:
    model_name: str
    metrics: dict
    cross_validation: dict
    predictions: pd.DataFrame
    artifact_path: str
    threshold_report: dict


@dataclass
class LatestPrediction:
    model_name: str
    as_of_date: str
    prediction: int
    prob_up: float
    signal: int
    artifact_path: str


class PredictiveModel(ABC):
    @abstractmethod
    def train(self, features: pd.DataFrame) -> ModelArtifacts:
        raise NotImplementedError

    @abstractmethod
    def predict_latest(self, training_features: pd.DataFrame, inference_features: pd.DataFrame) -> LatestPrediction:
        raise NotImplementedError

    def walk_forward_splits(self, features: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        n = len(features)
        first_test = max(50, int(n * self.settings.walk_forward_initial_train_size))
        window = max(20, int(n * self.settings.walk_forward_test_size))
        validation_size = max(20, int(n * self.settings.walk_forward_validation_size))
        gap = self.settings.split_gap
        if first_test - validation_size - 2 * gap < 50 or first_test >= n:
            raise ValueError("Insufficient history for disjoint training, validation and testing.")
        if not features["date"].is_monotonic_increasing or features["date"].duplicated().any():
            raise ValueError("Features must have unique chronological signal dates.")
        splits = []
        for test_start in range(first_test, n, window):
            validation_end = test_start - gap
            validation_start = validation_end - validation_size
            train = features.iloc[:validation_start - gap].copy()
            validation = features.iloc[validation_start:validation_end].copy()
            test = features.iloc[test_start:min(test_start + window, n)].copy()
            if not (pd.to_datetime(train["execution_date"]).max() < pd.to_datetime(validation["date"]).min()
                    and pd.to_datetime(validation["execution_date"]).max() < pd.to_datetime(test["date"]).min()):
                raise ValueError("Label outcomes overlap a later evaluation window.")
            splits.append((train, validation, test))
        return splits

    def evaluate_walk_forward(self, features: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
        from ..backtesting.engine import BacktestEngine
        engine = BacktestEngine(self.settings)
        predictions, reports = [], []
        for fold, (train, validation, test) in enumerate(self.walk_forward_splits(features), 1):
            fitted = self.fit_frame(train)
            self.last_evaluated_model = fitted
            validation_predictions = self.prediction_frame(features, validation, fitted)
            report = engine.evaluate_thresholds(validation_predictions, self.model_name, str(features["ticker"].iloc[0]))
            threshold = report["best_threshold"]["threshold"]
            test_predictions = self.prediction_frame(features, test, fitted)
            test_predictions["signal"] = (test_predictions["prob_up"] >= threshold).astype(int)
            test_predictions["threshold"] = threshold
            test_predictions["fold"] = fold
            test_predictions["training_prevalence"] = float(train["target"].mean())
            predictions.append(test_predictions)
            report["fold"] = fold
            for name, frame in [("train", train), ("validation", validation), ("test", test)]:
                report[name] = {"start_date": str(frame["date"].iloc[0]), "end_date": str(frame["date"].iloc[-1]),
                                "last_outcome_date": str(frame["execution_date"].iloc[-1]), "rows": len(frame)}
            reports.append(report)
        combined = combine_fold_predictions(predictions)
        cv = build_cross_validation_report(combined)
        cv["scheme"] = "expanding_train_validation_test_with_label_gap"
        cv["split_gap_sessions"] = self.settings.split_gap
        cv["windows"] = [{k: report[k] for k in ["fold", "train", "validation", "test"]} for report in reports]
        return combined, cv, {"selection_data": "validation_only", "folds": reports,
                              "latest_validation_threshold": reports[-1]["best_threshold"]["threshold"]}

    def fit_for_latest(self, features: pd.DataFrame, inference: pd.DataFrame) -> tuple:
        from ..backtesting.engine import BacktestEngine
        validation_size = max(20, int(len(features) * self.settings.walk_forward_validation_size))
        validation = features.iloc[-validation_size:]
        train = features.iloc[:len(features) - validation_size - self.settings.split_gap]
        if len(train) < 50 or pd.to_datetime(features["execution_date"]).max() > pd.to_datetime(inference["date"].iloc[-1]):
            raise ValueError("Live training requires sufficient, already resolved history.")
        if pd.to_datetime(train["execution_date"]).max() >= pd.to_datetime(validation["date"]).min():
            raise ValueError("Live training labels overlap validation.")
        fitted = self.fit_frame(train)
        report = BacktestEngine(self.settings).evaluate_thresholds(
            self.prediction_frame(features, validation, fitted), self.model_name, str(features["ticker"].iloc[0]))
        return fitted, report["best_threshold"]["threshold"]

    def prediction_frame(self, history: pd.DataFrame, frame: pd.DataFrame, fitted: object) -> pd.DataFrame:
        columns = ["date", "execution_date", "close", "target", "next_day_return", "entry_open", "exit_close",
                   "benchmark_entry", "benchmark_exit", "return_1d"]
        result = frame[columns].copy()
        result["prob_up"] = self.predict_frame(fitted, frame, history)
        result["prediction"] = (result["prob_up"] >= 0.5).astype(int)
        return result


def combine_fold_predictions(prediction_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not prediction_frames:
        raise ValueError("Walk-forward evaluation produced no prediction frames.")
    combined = pd.concat(prediction_frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def aggregate_classification_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["target"].to_numpy()
    y_pred = predictions["prediction"].to_numpy()
    y_prob = predictions["prob_up"].to_numpy()

    from .evaluator import evaluate_classifier

    metrics = evaluate_classifier(y_true, y_pred, y_prob).to_dict()
    metrics["fold_count"] = int(predictions["fold"].nunique()) if "fold" in predictions else 1
    metrics["evaluation_rows"] = int(len(predictions))
    return metrics


def build_cross_validation_report(predictions: pd.DataFrame) -> dict:
    from .evaluator import evaluate_classifier

    folds: list[dict] = []
    for fold_id, fold_frame in predictions.groupby("fold", sort=True):
        fold_metrics = evaluate_classifier(
            fold_frame["target"].to_numpy(),
            fold_frame["prediction"].to_numpy(),
            fold_frame["prob_up"].to_numpy(),
        ).to_dict()
        fold_metrics["fold"] = int(fold_id)
        fold_metrics["rows"] = int(len(fold_frame))
        fold_metrics["start_date"] = str(fold_frame["date"].iloc[0])
        fold_metrics["end_date"] = str(fold_frame["date"].iloc[-1])
        folds.append(fold_metrics)

    if not folds:
        return {"scheme": "walk_forward", "folds": [], "summary": {}}

    weights = [fold["rows"] for fold in folds]
    summary = {
        "mean_accuracy": float(np.average([fold["accuracy"] for fold in folds], weights=weights)),
        "mean_precision": float(np.average([fold["precision"] for fold in folds], weights=weights)),
        "mean_recall": float(np.average([fold["recall"] for fold in folds], weights=weights)),
        "mean_f1": float(np.average([fold["f1"] for fold in folds], weights=weights)),
        "mean_roc_auc": float(np.average([fold["roc_auc"] for fold in folds], weights=weights)),
        "fold_count": len(folds),
        "averaging": "weighted_by_evaluation_rows",
    }
    return {
        "scheme": "walk_forward",
        "folds": folds,
        "summary": summary,
        "classification_baselines": classification_baselines(predictions),
    }


def classification_baselines(predictions: pd.DataFrame) -> dict:
    from .evaluator import evaluate_classifier
    target = predictions["target"].to_numpy()
    probabilities = {
        "always_up": np.ones(len(predictions)),
        "training_prevalence": predictions["training_prevalence"].to_numpy(),
        "momentum": predictions["return_1d"].gt(0).astype(float).to_numpy(),
    }
    return {name: evaluate_classifier(target, (prob >= 0.5).astype(int), prob).to_dict()
            for name, prob in probabilities.items()}
