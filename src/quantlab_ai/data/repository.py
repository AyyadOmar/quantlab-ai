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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS live_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    prob_up REAL NOT NULL,
                    signal INTEGER NOT NULL,
                    actual_direction INTEGER,
                    actual_return REAL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    artifact_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
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

    def log_live_prediction(
        self,
        ticker: str,
        model_name: str,
        as_of_date: str,
        prediction: int,
        prob_up: float,
        signal: int,
        artifact_path: str,
    ) -> None:
        with sqlite3.connect(self.settings.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO live_predictions (
                    ticker, model_name, as_of_date, prediction, prob_up, signal, artifact_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    model_name,
                    as_of_date,
                    prediction,
                    prob_up,
                    signal,
                    artifact_path,
                ),
            )
            connection.commit()

    def get_pending_live_predictions(self, ticker: str | None = None) -> list[dict]:
        with sqlite3.connect(self.settings.database_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            if ticker:
                cursor.execute(
                    """
                    SELECT * FROM live_predictions
                    WHERE status = 'pending' AND ticker = ?
                    ORDER BY as_of_date ASC
                    """,
                    (ticker,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM live_predictions
                    WHERE status = 'pending'
                    ORDER BY ticker ASC, as_of_date ASC
                    """
                )
            return [dict(row) for row in cursor.fetchall()]

    def resolve_live_prediction(self, prediction_id: int, actual_direction: int, actual_return: float) -> None:
        with sqlite3.connect(self.settings.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE live_predictions
                SET actual_direction = ?, actual_return = ?, status = 'resolved', resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (actual_direction, actual_return, prediction_id),
            )
            connection.commit()

    def summarize_live_predictions(self, ticker: str | None = None) -> dict:
        with sqlite3.connect(self.settings.database_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            query = """
                SELECT
                    model_name,
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved_predictions,
                    SUM(CASE WHEN status = 'resolved' AND prediction = actual_direction THEN 1 ELSE 0 END) AS correct_predictions,
                    AVG(CASE WHEN status = 'resolved' THEN prob_up END) AS avg_probability
                FROM live_predictions
            """
            params: tuple = ()
            if ticker:
                query += " WHERE ticker = ?"
                params = (ticker,)
            query += " GROUP BY model_name ORDER BY model_name"
            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]

        summary_rows = []
        for row in rows:
            resolved = int(row["resolved_predictions"] or 0)
            correct = int(row["correct_predictions"] or 0)
            accuracy = float(correct / resolved) if resolved else 0.0
            summary_rows.append(
                {
                    "model_name": row["model_name"],
                    "total_predictions": int(row["total_predictions"] or 0),
                    "resolved_predictions": resolved,
                    "correct_predictions": correct,
                    "live_accuracy": accuracy,
                    "avg_probability": float(row["avg_probability"] or 0.0),
                }
            )

        return {"ticker": ticker, "models": summary_rows}
