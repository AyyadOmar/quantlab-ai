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
from .base import ModelArtifacts, PredictiveModel, aggregate_classification_metrics, combine_fold_predictions
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
        self.registry = ModelRegistry(self.settings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, features: pd.DataFrame) -> ModelArtifacts:
        prediction_frames: list[pd.DataFrame] = []

        for fold_index, (train_frame, test_frame) in enumerate(self.walk_forward_splits(features), start=1):
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_frame[FEATURE_COLUMNS])
            test_scaled = scaler.transform(test_frame[FEATURE_COLUMNS])

            x_train, y_train = self._create_sequences(train_scaled, train_frame["target"].to_numpy())
            x_test, _ = self._create_sequences(test_scaled, test_frame["target"].to_numpy())

            if len(x_train) == 0 or len(x_test) == 0:
                continue

            model = self._train_model(x_train, y_train)
            probabilities, predictions = self._predict(model, x_test)

            aligned_test = test_frame.iloc[self.settings.lstm_sequence_length - 1 :].copy().reset_index(drop=True)
            aligned_test["prediction"] = predictions
            aligned_test["prob_up"] = probabilities
            aligned_test["signal"] = (aligned_test["prob_up"] >= self.settings.signal_threshold).astype(int)
            aligned_test["fold"] = fold_index
            prediction_frames.append(
                aligned_test[["date", "close", "target", "next_day_return", "prediction", "prob_up", "signal", "fold"]]
            )

        predictions = combine_fold_predictions(prediction_frames)
        metrics = aggregate_classification_metrics(predictions)

        full_scaler = StandardScaler()
        full_scaled = full_scaler.fit_transform(features[FEATURE_COLUMNS])
        x_full, y_full = self._create_sequences(full_scaled, features["target"].to_numpy())
        final_model = self._train_model(x_full, y_full)

        artifact_path = self.registry.save_torch(
            artifact_name=f"lstm_{features['ticker'].iloc[0].lower()}",
            payload={
                "model_state_dict": final_model.state_dict(),
                "scaler": full_scaler,
                "feature_columns": FEATURE_COLUMNS,
                "sequence_length": self.settings.lstm_sequence_length,
            },
        )
        return ModelArtifacts(
            model_name="lstm",
            metrics=metrics,
            predictions=predictions,
            artifact_path=artifact_path,
        )

    def _create_sequences(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence_length = self.settings.lstm_sequence_length
        sequences: list[np.ndarray] = []
        labels: list[int] = []
        for index in range(sequence_length - 1, len(x_values)):
            sequences.append(x_values[index - sequence_length + 1 : index + 1])
            labels.append(int(y_values[index]))
        return np.array(sequences), np.array(labels)

    def _train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> LSTMClassifier:
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
