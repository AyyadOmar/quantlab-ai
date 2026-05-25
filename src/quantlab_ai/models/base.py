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

    def walk_forward_splits(self, features: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        total_rows = len(features)
        initial_train_size = max(50, int(total_rows * self.settings.walk_forward_initial_train_size))
        test_window = max(20, int(total_rows * self.settings.walk_forward_test_size))
        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        train_end = initial_train_size
        while train_end + test_window <= total_rows:
            train_frame = features.iloc[:train_end].copy()
            test_frame = features.iloc[train_end : train_end + test_window].copy()
            splits.append((train_frame, test_frame))
            train_end += test_window

        if not splits:
            fallback_split = max(1, int(total_rows * 0.8))
            splits.append((features.iloc[:fallback_split].copy(), features.iloc[fallback_split:].copy()))

        return [(train_frame, test_frame) for train_frame, test_frame in splits if not test_frame.empty]


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

    summary = {
        "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
        "mean_precision": float(np.mean([fold["precision"] for fold in folds])),
        "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
        "mean_f1": float(np.mean([fold["f1"] for fold in folds])),
        "mean_roc_auc": float(np.mean([fold["roc_auc"] for fold in folds])),
        "fold_count": len(folds),
    }
    return {
        "scheme": "walk_forward",
        "folds": folds,
        "summary": summary,
    }
