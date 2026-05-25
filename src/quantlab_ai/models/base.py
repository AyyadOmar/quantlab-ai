from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
