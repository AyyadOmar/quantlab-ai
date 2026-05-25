from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    backtests_dir: Path = field(init=False)
    database_path: Path = field(init=False)
    test_size: float = 0.2
    random_state: int = 42
    signal_threshold: float = 0.55
    risk_free_rate: float = 0.01
    trading_fee_bps: float = 5.0
    market_context_ticker: str = "SPY"
    walk_forward_initial_train_size: float = 0.6
    walk_forward_test_size: float = 0.1
    lstm_sequence_length: int = 20
    lstm_epochs: int = 15
    lstm_hidden_size: int = 32
    lstm_learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.models_dir = self.project_root / "models"
        self.plots_dir = self.project_root / "visualizations"
        self.backtests_dir = self.project_root / "backtesting"
        self.database_path = self.data_dir / "quantlab.db"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.plots_dir,
            self.backtests_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
