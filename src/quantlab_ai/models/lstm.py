from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import Settings
from ..features.builder import FEATURE_COLUMNS
from .base import (
    LatestPrediction,
    ModelArtifacts,
    PredictiveModel,
    aggregate_classification_metrics,
)
from .registry import ModelRegistry


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        logits = self.classifier(output[:, -1, :])
        return logits.squeeze(-1)


@dataclass
class LSTMTrainer(PredictiveModel):
    settings: Settings

    def __post_init__(self) -> None:
        self.model_name = "lstm"
        self.registry = ModelRegistry(self.settings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, features: pd.DataFrame) -> ModelArtifacts:
        predictions, cross_validation, threshold_report = self.evaluate_walk_forward(features)
        metrics = aggregate_classification_metrics(predictions)

        final_model, full_scaler = self.last_evaluated_model

        artifact_path = self.registry.save_torch(
            artifact_name=f"lstm_{features['ticker'].iloc[0].lower()}",
            payload={
                "model_state_dict": final_model.state_dict(),
                "scaler": full_scaler,
                "feature_columns": FEATURE_COLUMNS,
                "sequence_length": self.settings.lstm_sequence_length,
                "protocol": self.settings.protocol,
                "threshold": threshold_report["latest_validation_threshold"],
                "trained_through": threshold_report["folds"][-1]["train"]["last_outcome_date"],
            },
        )
        return ModelArtifacts(
            model_name="lstm",
            metrics=metrics,
            cross_validation=cross_validation,
            predictions=predictions,
            artifact_path=artifact_path,
            threshold_report=threshold_report,
        )

    def predict_latest(self, training_features: pd.DataFrame, inference_features: pd.DataFrame) -> LatestPrediction:
        (model, scaler), threshold = self.fit_for_latest(training_features, inference_features)

        inference_scaled = scaler.transform(inference_features[FEATURE_COLUMNS])
        sequence_length = self.settings.lstm_sequence_length
        if len(inference_scaled) < sequence_length:
            raise ValueError("Not enough rows to create an LSTM inference sequence.")

        latest_sequence = inference_scaled[-sequence_length:]
        probabilities, predictions = self._predict(model, np.array([latest_sequence]))
        artifact_path = self.registry.save_torch(
            artifact_name=f"lstm_{training_features['ticker'].iloc[0].lower()}_latest",
            payload={
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "feature_columns": FEATURE_COLUMNS,
                "sequence_length": sequence_length,
                "threshold": threshold, "protocol": self.settings.protocol,
            },
        )
        return LatestPrediction(
            model_name="lstm",
            as_of_date=str(inference_features["date"].iloc[-1]),
            prediction=int(predictions[0]),
            prob_up=float(probabilities[0]),
            signal=int(probabilities[0] >= threshold),
            artifact_path=artifact_path,
        )

    def fit_frame(self, frame: pd.DataFrame) -> tuple:
        scaler = StandardScaler()
        values = scaler.fit_transform(frame[FEATURE_COLUMNS])
        x, y = self._create_sequences(values, frame["target"].to_numpy())
        if not len(x):
            raise ValueError("Insufficient training history for LSTM sequences.")
        return self._train_model(x, y), scaler

    def predict_frame(self, fitted: tuple, frame: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
        model, scaler = fitted
        # Carry preceding feature history into every window; no test labels are used.
        prefix = history.loc[history["date"] < frame["date"].iloc[0]].tail(self.settings.lstm_sequence_length - 1)
        values = scaler.transform(pd.concat([prefix, frame])[FEATURE_COLUMNS])
        sequences, _ = self._create_sequences(values, np.zeros(len(values)))
        if len(sequences) != len(frame):
            raise ValueError("Insufficient history to cover every evaluation session.")
        return self._predict(model, sequences)[0]

    def _create_sequences(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence_length = self.settings.lstm_sequence_length
        sequences: list[np.ndarray] = []
        labels: list[int] = []
        for index in range(sequence_length - 1, len(x_values)):
            sequences.append(x_values[index - sequence_length + 1 : index + 1])
            labels.append(int(y_values[index]))
        return np.array(sequences), np.array(labels)

    def _train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> LSTMClassifier:
        torch.manual_seed(self.settings.random_state)
        model = LSTMClassifier(input_size=len(FEATURE_COLUMNS), hidden_size=self.settings.lstm_hidden_size).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.settings.lstm_learning_rate)

        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        model.train()
        for _ in range(self.settings.lstm_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def _predict(self, model: LSTMClassifier, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32).to(self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)
        return probabilities, predictions
