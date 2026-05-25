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
from .base import ModelArtifacts, PredictiveModel
from .evaluator import evaluate_classifier
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
        train_frame, test_frame = self._chronological_split(features)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_frame[FEATURE_COLUMNS])
        test_scaled = scaler.transform(test_frame[FEATURE_COLUMNS])

        x_train, y_train = self._create_sequences(train_scaled, train_frame["target"].to_numpy())
        x_test, y_test = self._create_sequences(test_scaled, test_frame["target"].to_numpy())

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

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32).to(self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)

        aligned_test = test_frame.iloc[self.settings.lstm_sequence_length - 1 :].copy().reset_index(drop=True)
        metrics = evaluate_classifier(y_test, predictions, probabilities).to_dict()
        aligned_test["prediction"] = predictions
        aligned_test["prob_up"] = probabilities
        aligned_test["signal"] = (aligned_test["prob_up"] >= self.settings.signal_threshold).astype(int)

        artifact_path = self.registry.save_torch(
            artifact_name=f"lstm_{features['ticker'].iloc[0].lower()}",
            payload={
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "feature_columns": FEATURE_COLUMNS,
                "sequence_length": self.settings.lstm_sequence_length,
            },
        )
        return ModelArtifacts(
            model_name="lstm",
            metrics=metrics,
            predictions=aligned_test[["date", "close", "target", "next_day_return", "prediction", "prob_up", "signal"]],
            artifact_path=artifact_path,
        )

    def _chronological_split(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_index = int(len(features) * (1 - self.settings.test_size))
        return features.iloc[:split_index].copy(), features.iloc[split_index:].copy()

    def _create_sequences(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence_length = self.settings.lstm_sequence_length
        sequences: list[np.ndarray] = []
        labels: list[int] = []
        for index in range(sequence_length - 1, len(x_values)):
            sequences.append(x_values[index - sequence_length + 1 : index + 1])
            labels.append(int(y_values[index]))
        return np.array(sequences), np.array(labels)
