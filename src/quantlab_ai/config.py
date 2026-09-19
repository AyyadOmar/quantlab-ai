from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


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
    threshold_sweep: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7)
    protocol: str = "open_to_close_v2"
    use_cached_data: bool = False
    risk_free_rate: float = 0.0
    trading_fee_bps: float = 5.0
    slippage_bps: float = 2.0
    market_context_ticker: str = "SPY"
    walk_forward_initial_train_size: float = 0.6
    walk_forward_test_size: float = 0.1
    walk_forward_validation_size: float = 0.15
    split_gap: int = 1
    lstm_sequence_length: int = 20
    lstm_epochs: int = 15
    lstm_hidden_size: int = 32
    lstm_learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed" / self.protocol
        self.models_dir = self.project_root / "models" / self.protocol
        self.plots_dir = self.project_root / "visualizations" / self.protocol
        self.backtests_dir = self.project_root / "backtesting" / self.protocol
        self.database_path = self.data_dir / f"quantlab_{self.protocol}.db"
        if not 0 <= self.trading_fee_bps < 10000 or not 0 <= self.slippage_bps < 10000:
            raise ValueError("Per-side costs must be between 0 and 10,000 basis points.")
        if self.split_gap < 1:
            raise ValueError("A gap of at least one session is required for next-session labels.")
        if not (0 < self.walk_forward_validation_size < self.walk_forward_initial_train_size < 1
                and 0 < self.walk_forward_test_size < 1):
            raise ValueError("Invalid chronological split proportions.")
        if not self.threshold_sweep or any(not 0 <= value <= 1 for value in self.threshold_sweep):
            raise ValueError("Threshold candidates must be probabilities.")

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
