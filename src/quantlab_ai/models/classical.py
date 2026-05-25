from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..features.builder import FEATURE_COLUMNS
from .base import ModelArtifacts, PredictiveModel, aggregate_classification_metrics, combine_fold_predictions
from .registry import ModelRegistry

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def clip_extreme_values(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -10.0, 10.0)


@dataclass
class ClassicalModelTrainer(PredictiveModel):
    settings: Settings
    model_name: str

    def __post_init__(self) -> None:
        self.registry = ModelRegistry(self.settings)

    def train(self, features: pd.DataFrame) -> ModelArtifacts:
        estimator_template = self._build_estimator()
        prediction_frames: list[pd.DataFrame] = []

        for fold_index, (train_frame, test_frame) in enumerate(self.walk_forward_splits(features), start=1):
            x_train = train_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
            y_train = train_frame["target"]
            x_test = test_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

            model = clone(estimator_template)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            y_prob = self._probabilities(model, x_test)

            fold_predictions = test_frame[["date", "close", "target", "next_day_return"]].copy()
            fold_predictions["prediction"] = y_pred
            fold_predictions["prob_up"] = y_prob
            fold_predictions["signal"] = (fold_predictions["prob_up"] >= self.settings.signal_threshold).astype(int)
            fold_predictions["fold"] = fold_index
            prediction_frames.append(fold_predictions)

        predictions = combine_fold_predictions(prediction_frames)
        metrics = aggregate_classification_metrics(predictions)

        final_model = clone(estimator_template)
        final_model.fit(
            features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan),
            features["target"],
        )

        artifact_path = self.registry.save_joblib(
            artifact_name=f"{self.model_name}_{features['ticker'].iloc[0].lower()}",
            payload=final_model,
        )
        return ModelArtifacts(
            model_name=self.model_name,
            metrics=metrics,
            predictions=predictions,
            artifact_path=artifact_path,
        )

    def _build_estimator(self) -> Pipeline | RandomForestClassifier:
        if self.model_name == "logistic_regression":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clipper",
                        FunctionTransformer(
                            clip_extreme_values,
                            validate=False,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            solver="liblinear",
                            C=0.1,
                            random_state=self.settings.random_state,
                        ),
                    ),
                ]
            )
        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                random_state=self.settings.random_state,
            )
        if self.model_name == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed. Install it or choose another model.")
            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=self.settings.random_state,
            )
        raise ValueError(f"Unsupported classical model: {self.model_name}")

    @staticmethod
    def _probabilities(model: object, x_test: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_test)[:, 1]
        return model.predict(x_test)
