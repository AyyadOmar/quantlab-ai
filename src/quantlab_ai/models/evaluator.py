from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )
