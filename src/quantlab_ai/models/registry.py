from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import torch

from ..config import Settings


@dataclass
class ModelRegistry:
    settings: Settings

    def save_joblib(self, artifact_name: str, payload: object) -> str:
        output_path = self.settings.models_dir / f"{artifact_name}.joblib"
        joblib.dump(payload, output_path)
        return str(output_path)

    def save_torch(self, artifact_name: str, payload: dict) -> str:
        output_path = self.settings.models_dir / f"{artifact_name}.pt"
        torch.save(payload, output_path)
        return str(output_path)

    def resolve(self, artifact_name: str) -> Path:
        return self.settings.models_dir / artifact_name
