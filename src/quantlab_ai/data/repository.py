from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ..config import Settings


@dataclass
class ExperimentRepository:
    settings: Settings

    def initialize(self) -> None:
        with sqlite3.connect(self.settings.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def log_experiment(
        self,
        ticker: str,
        model_name: str,
        start_date: str,
        end_date: str,
        metrics: dict,
        artifact_path: str,
    ) -> None:
        with sqlite3.connect(self.settings.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO experiments (
                    ticker, model_name, start_date, end_date, metrics_json, artifact_path
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    model_name,
                    start_date,
                    end_date,
                    json.dumps(metrics, indent=2),
                    artifact_path,
                ),
            )
            connection.commit()
