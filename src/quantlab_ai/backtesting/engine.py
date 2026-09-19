from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict
    threshold_report: dict | None = None


@dataclass
class BacktestEngine:
    settings: Settings

    def run(self, predictions: pd.DataFrame, model_name: str, ticker: str) -> BacktestResult:
        return self.run_with_threshold(predictions, model_name, ticker, threshold=None)

    def run_with_threshold(self, predictions: pd.DataFrame, model_name: str, ticker: str,
                           threshold: float | None, persist_outputs: bool = True) -> BacktestResult:
        frame = predictions.copy().reset_index(drop=True)
        required = ["date", "execution_date", "entry_open", "exit_close", "next_day_return",
                    "benchmark_entry", "benchmark_exit", "return_1d", "prob_up"]
        if frame.empty or frame[required].isna().any().any():
            raise ValueError("Backtesting requires complete execution prices and predictions.")
        dates = pd.to_datetime(frame["execution_date"])
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Execution sessions must be unique and ordered.")
        if not (dates > pd.to_datetime(frame["date"])).all():
            raise ValueError("Execution must occur after the signal session.")
        if not frame["prob_up"].between(0, 1).all():
            raise ValueError("Predicted probabilities must be between zero and one.")
        prices = frame[["entry_open", "exit_close", "benchmark_entry", "benchmark_exit"]]
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
            raise ValueError("Execution prices must be finite and positive.")
        if not np.allclose(frame["next_day_return"], frame["exit_close"] / frame["entry_open"] - 1):
            raise ValueError("Target returns do not match the execution prices.")
        if threshold is not None:
            frame["signal"] = (frame["prob_up"] >= threshold).astype(int)
        elif "signal" not in frame:
            raise ValueError("Final evaluation requires signals selected before testing.")
        if not frame["signal"].isin([0, 1]).all():
            raise ValueError("This protocol supports long/cash signals only.")

        fee = self.settings.trading_fee_bps / 10000
        slip = self.settings.slippage_bps / 10000
        entry_factor = (1 + slip) * (1 + fee)
        exit_factor = (1 - slip) * (1 - fee)
        # Every active session is a separate round trip; cash earns zero.
        net_intraday = (frame["exit_close"] / frame["entry_open"]) * exit_factor / entry_factor - 1
        frame["strategy_return"] = net_intraday.where(frame["signal"].eq(1), 0.0)
        frame["always_long_return"] = net_intraday
        frame["momentum_signal"] = frame["return_1d"].gt(0).astype(int)
        frame["momentum_return"] = net_intraday.where(frame["momentum_signal"].eq(1), 0.0)

        # Passive exposure includes overnight returns and adjusted dividends, with
        # one initial entry and one final exit over the identical evaluation span.
        passive = frame["benchmark_exit"].pct_change()
        passive.iloc[0] = frame["benchmark_exit"].iloc[0] / frame["benchmark_entry"].iloc[0] / entry_factor - 1
        passive.iloc[-1] = (1 + passive.iloc[-1]) * exit_factor - 1
        frame["benchmark_return"] = passive
        for returns, curve in [("strategy_return", "equity_curve"),
                               ("always_long_return", "always_long_curve"),
                               ("momentum_return", "momentum_curve"),
                               ("benchmark_return", "benchmark_curve")]:
            frame[curve] = (1 + frame[returns]).cumprod()
        frame["drawdown"] = frame["equity_curve"] / frame["equity_curve"].cummax().clip(lower=1.0) - 1
        metrics = self._curve_metrics(frame["strategy_return"], frame["signal"])
        metrics.update({"protocol": self.settings.protocol, "threshold": threshold,
                        "threshold_policy": "validation_per_fold" if threshold is None else "fixed",
                        "fee_bps_per_side": self.settings.trading_fee_bps,
                        "slippage_bps_per_side": self.settings.slippage_bps,
                        "cash_return": 0.0,
                        "start_date": str(dates.iloc[0]), "end_date": str(dates.iloc[-1])})
        benchmarks = {
            "buy_and_hold": self._curve_metrics(passive, pd.Series(1, index=frame.index), trade_count=1),
            "always_long": self._curve_metrics(net_intraday, pd.Series(1, index=frame.index)),
            "momentum": self._curve_metrics(frame["momentum_return"], frame["momentum_signal"]),
        }
        benchmarks["buy_and_hold"]["exposure_note"] = "Continuous exposure including overnight; adjusted total-return approximation."
        benchmarks["always_long"]["exposure_note"] = "Enter each open, exit each close; same holding window as model."
        metrics["benchmark_return"] = benchmarks["buy_and_hold"]["total_return"]
        columns = ["date", "execution_date", "entry_open", "exit_close", "prob_up", "next_day_return", "strategy_return"]
        trades = frame.loc[frame["signal"].eq(1), columns].copy()
        if persist_outputs:
            stem = f"{ticker.lower()}_{model_name}"
            trades.to_csv(self.settings.backtests_dir / f"{stem}_trades.csv", index=False)
            frame.to_csv(self.settings.backtests_dir / f"{stem}_daily.csv", index=False)
            (self.settings.backtests_dir / f"{stem}_summary.json").write_text(json.dumps(metrics, indent=2))
            (self.settings.backtests_dir / f"{stem}_benchmarks.json").write_text(json.dumps(benchmarks, indent=2))
        curve = frame[["execution_date", "equity_curve", "benchmark_curve", "always_long_curve", "momentum_curve", "drawdown"]].rename(columns={"execution_date": "date"})
        return BacktestResult(trades, curve, metrics, benchmarks)

    def evaluate_thresholds(self, predictions: pd.DataFrame, model_name: str, ticker: str) -> dict:
        """Call only on an earlier validation window, never on final test rows."""
        results = []
        for threshold in self.settings.threshold_sweep:
            metrics = self.run_with_threshold(predictions, model_name, ticker, threshold, False).metrics
            results.append({key: metrics[key] for key in
                            ["threshold", "total_return", "max_drawdown", "sharpe_ratio", "win_rate", "trade_count"]})
        # Larger (less negative) drawdown is preferable in a tie.
        best = max(results, key=lambda row: (row["sharpe_ratio"], row["total_return"], row["max_drawdown"], row["threshold"]))
        return {"selection_data": "validation_only", "metric": "sharpe_then_return_then_drawdown",
                "thresholds": results, "best_threshold": best}

    def _curve_metrics(self, returns: pd.Series, signals: pd.Series, trade_count: int | None = None) -> dict:
        curve = (1 + returns).cumprod()
        drawdown = curve / curve.cummax().clip(lower=1.0) - 1
        std = returns.std(ddof=0)
        sharpe = ((returns.mean() * 252 - self.settings.risk_free_rate) / (std * np.sqrt(252))) if std > 0 else 0.0
        active = returns.loc[signals.eq(1)]
        count = int(signals.sum()) if trade_count is None else trade_count
        return {"total_return": float(curve.iloc[-1] - 1), "max_drawdown": float(drawdown.min()),
                "sharpe_ratio": float(sharpe), "win_rate": float((active > 0).mean()) if len(active) else 0.0,
                "trade_count": count, "order_count": count * 2,
                "active_session_fraction": float(signals.mean()), "evaluation_sessions": len(returns)}
