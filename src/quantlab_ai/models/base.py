from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelArtifacts:
    model_name: str
    metrics: dict
    predictions: pd.DataFrame
    artifact_path: str


class PredictiveModel(ABC):
    @abstractmethod
    def train(self, features: pd.DataFrame) -> ModelArtifacts:
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
