from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..features.builder import FEATURE_COLUMNS
from .base import ModelArtifacts, PredictiveModel
from .evaluator import evaluate_classifier
from .registry import ModelRegistry

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class ClassicalModelTrainer(PredictiveModel):
    settings: Settings
    model_name: str

    def __post_init__(self) -> None:
        self.registry = ModelRegistry(self.settings)

    def train(self, features: pd.DataFrame) -> ModelArtifacts:
        train_frame, test_frame = self._chronological_split(features)
        x_train = train_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        y_train = train_frame["target"]
        x_test = test_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        y_test = test_frame["target"]

        model = self._build_estimator()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_prob = self._probabilities(model, x_test)
        metrics = evaluate_classifier(y_test.to_numpy(), y_pred, y_prob).to_dict()

        predictions = test_frame[["date", "close", "target", "next_day_return"]].copy()
        predictions["prediction"] = y_pred
        predictions["prob_up"] = y_prob
        predictions["signal"] = (predictions["prob_up"] >= self.settings.signal_threshold).astype(int)

        artifact_path = self.registry.save_joblib(
            artifact_name=f"{self.model_name}_{features['ticker'].iloc[0].lower()}",
            payload=model,
        )
        return ModelArtifacts(
            model_name=self.model_name,
            metrics=metrics,
            predictions=predictions,
            artifact_path=artifact_path,
        )

    def _chronological_split(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_index = int(len(features) * (1 - self.settings.test_size))
        return features.iloc[:split_index].copy(), features.iloc[split_index:].copy()

    def _build_estimator(self) -> Pipeline | RandomForestClassifier:
        if self.model_name == "logistic_regression":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            solver="liblinear",
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
