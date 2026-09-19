from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..features.profiles import feature_columns
from .base import (
    LatestPrediction,
    ModelArtifacts,
    PredictiveModel,
    aggregate_classification_metrics,
)
from .registry import ModelRegistry



def clip_extreme_values(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -10.0, 10.0)


@dataclass
class ClassicalModelTrainer(PredictiveModel):
    settings: Settings
    model_name: str
    feature_set: str = "original"
    model_profile: str = "baseline"

    def __post_init__(self) -> None:
        self.registry = ModelRegistry(self.settings)
        self.feature_columns = feature_columns(self.feature_set)
        if self.model_profile not in {"baseline", "regularized"}:
            raise ValueError("Unknown model profile.")
        if self.model_profile == "regularized" and self.model_name not in {"logistic_regression", "xgboost"}:
            raise ValueError("Regularized profile is defined for logistic regression and XGBoost only.")

    @property
    def artifact_name(self) -> str:
        if self.feature_set == "original" and self.model_profile == "baseline":
            return self.model_name
        return f"{self.model_name}_{self.feature_set}_{self.model_profile}"

    def train(self, features: pd.DataFrame) -> ModelArtifacts:
        predictions, cross_validation, threshold_report = self.evaluate_walk_forward(features)
        metrics = aggregate_classification_metrics(predictions)

        final_model = self.last_evaluated_model

        artifact_path = self.registry.save_joblib(
            artifact_name=f"{self.artifact_name}_{features['ticker'].iloc[0].lower()}",
            payload={"model": final_model, "protocol": self.settings.protocol,
                     "feature_columns": self.feature_columns,
                     "threshold": threshold_report["latest_validation_threshold"],
                     "trained_through": threshold_report["folds"][-1]["train"]["last_outcome_date"]},
        )
        return ModelArtifacts(
            model_name=self.model_name,
            metrics=metrics,
            cross_validation=cross_validation,
            predictions=predictions,
            artifact_path=artifact_path,
            threshold_report=threshold_report,
        )

    def predict_latest(self, training_features: pd.DataFrame, inference_features: pd.DataFrame) -> LatestPrediction:
        model, threshold = self.fit_for_latest(training_features, inference_features)

        latest_frame = inference_features.iloc[[-1]].copy()
        latest_x = latest_frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        prediction = int(model.predict(latest_x)[0])
        prob_up = float(self._probabilities(model, latest_x)[0])
        signal = int(prob_up >= threshold)

        artifact_path = self.registry.save_joblib(
            artifact_name=f"{self.artifact_name}_{training_features['ticker'].iloc[0].lower()}_latest",
            payload={"model": model, "threshold": threshold, "protocol": self.settings.protocol, "feature_columns": self.feature_columns},
        )
        return LatestPrediction(
            model_name=self.model_name,
            as_of_date=str(latest_frame["date"].iloc[0]),
            prediction=prediction,
            prob_up=prob_up,
            signal=signal,
            artifact_path=artifact_path,
        )

    def fit_frame(self, frame: pd.DataFrame) -> object:
        model = self._build_estimator()
        model.fit(frame[self.feature_columns], frame["target"])
        return model

    def predict_frame(self, fitted: object, frame: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
        return self._probabilities(fitted, frame[self.feature_columns])

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
                            C=0.01 if self.model_profile == "regularized" else 0.1,
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
            from xgboost import XGBClassifier
            if self.model_profile == "regularized":
                return XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.03,
                                     min_child_weight=20, reg_lambda=10.0, reg_alpha=1.0,
                                     subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                     eval_metric="logloss", random_state=self.settings.random_state)
            return XGBClassifier(
                n_estimators=300,
                n_jobs=2,
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
