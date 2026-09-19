"""Controlled feature/regularization experiments on the v2 execution protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .backtesting.engine import BacktestEngine
from .config import Settings
from .data.loader import MarketDataLoader
from .features.builder import FeatureBuilder
from .features.profiles import feature_columns
from .models.base import aggregate_classification_metrics, classification_baselines, combine_fold_predictions
from .models.classical import ClassicalModelTrainer
from .models.evaluator import evaluate_classifier


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    model_profile: str

    @property
    def name(self) -> str:
        return f"{self.feature_set}_{self.model_profile}"


CANDIDATES = tuple(Candidate(features, profile)
                   for features in ("original", "relative", "compact")
                   for profile in ("baseline", "regularized"))


def choose_candidate(validation_scores: list[dict]) -> str:
    """Test scores are deliberately absent from this interface."""
    if not validation_scores:
        raise ValueError("Selection requires validation scores.")
    for row in validation_scores:
        if not np.isfinite([row["brier_score"], row["log_loss"]]).all():
            raise ValueError("Non-finite validation score.")
    return min(validation_scores, key=lambda row: (row["brier_score"], row["log_loss"],
                                                   row["feature_count"], row["candidate"]))["candidate"]


def evaluate_fold(settings: Settings, history: pd.DataFrame, train: pd.DataFrame,
                  validation: pd.DataFrame, test: pd.DataFrame, model_name: str,
                  candidates: tuple[Candidate, ...] = CANDIDATES) -> tuple[dict, dict, dict]:
    """Finish selection before producing any test predictions."""
    engine = BacktestEngine(settings)
    fitted, trainers, thresholds, scores, timings = {}, {}, {}, [], {}
    for candidate in candidates:
        name = candidate.name
        trainer = ClassicalModelTrainer(settings, model_name, candidate.feature_set, candidate.model_profile)
        started = perf_counter()
        model = trainer.fit_frame(train)
        timings[name] = perf_counter() - started
        val_predictions = trainer.prediction_frame(history, validation, model)
        metric = evaluate_classifier(validation["target"].to_numpy(), val_predictions["prediction"].to_numpy(),
                                     val_predictions["prob_up"].to_numpy()).to_dict()
        scores.append({"candidate": name, "feature_count": len(trainer.feature_columns),
                       "brier_score": metric["brier_score"], "log_loss": metric["log_loss"]})
        thresholds[name] = engine.evaluate_thresholds(val_predictions, model_name, str(history.ticker.iloc[0]))
        fitted[name], trainers[name] = model, trainer
    chosen = choose_candidate(scores)
    result = {}
    for candidate in candidates:
        name = candidate.name
        started = perf_counter()
        pred = trainers[name].prediction_frame(history, test, fitted[name])
        predict_seconds = perf_counter() - started
        threshold = thresholds[name]["best_threshold"]["threshold"]
        pred["signal"] = pred.prob_up.ge(threshold).astype(int)
        pred["threshold"] = threshold
        pred["training_prevalence"] = float(train.target.mean())
        pred["candidate"] = name
        result[name] = pred
        timings[name] = {"fit_seconds": timings[name], "predict_seconds": predict_seconds,
                         "feature_count": len(trainers[name].feature_columns)}
    report = {"selected_candidate": chosen, "selection_metric": "validation_brier_then_log_loss",
              "validation_scores": scores, "thresholds": thresholds, "timings": timings}
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        report[name] = {"start": str(frame.date.iloc[0]), "end": str(frame.date.iloc[-1]),
                        "last_outcome": str(frame.execution_date.iloc[-1]), "rows": len(frame)}
    selected = next(candidate for candidate in candidates if candidate.name == chosen)
    artifact = {"model": fitted[chosen], "feature_columns": trainers[chosen].feature_columns,
                "candidate": asdict(selected), "threshold": thresholds[chosen]["best_threshold"]["threshold"],
                "protocol": settings.protocol, "trained_through": str(train.execution_date.iloc[-1]),
                "validation_through": str(validation.execution_date.iloc[-1])}
    return result, report, artifact


def paired_brier_interval(frames: list[pd.DataFrame], seed: int) -> dict:
    # Average losses across tickers first, then resample 20-session blocks to
    # retain contemporaneous market dependence and some serial dependence.
    daily = pd.concat(frames).groupby("execution_date")["loss_difference"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    n, block = len(daily), min(20, len(daily))
    estimates = []
    for _ in range(1000):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        estimates.append(float(daily[indices].mean()))
    return {"selected_minus_original_brier": float(daily.mean()),
            "block_bootstrap_95_percent_interval": np.quantile(estimates, [.025, .975]).tolist(),
            "block_sessions": block, "bootstrap_repetitions": 1000,
            "note": "Descriptive paired uncertainty; historical dates already inspected in earlier research."}


def run_feature_study(settings: Settings, tickers: list[str], start: str, end: str) -> dict:
    settings = replace(settings, protocol="feature_study_v3")
    settings.ensure_directories()
    loader, builder, engine = MarketDataLoader(settings), FeatureBuilder(settings), BacktestEngine(settings)
    plan = {"protocol": settings.protocol, "execution_protocol": "open_to_close_v2",
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "feature_columns": {name: feature_columns(name) for name in ["original", "relative", "compact"]},
            "selection": "Within each model family and fold, minimize validation Brier, then log loss; tie-break by feature count and name.",
            "tickers": tickers, "start": start, "end_exclusive": end,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]},
            "historical_only": True, "default_models_changed": False}
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2))
    rows, provenance, selections = [], {}, {}
    differences = {model: [] for model in ["logistic_regression", "xgboost"]}
    market = loader.download(settings.market_context_ticker, start, end)
    for ticker in tickers:
        raw = market if ticker == settings.market_context_ticker else loader.download(ticker, start, end)
        context = None if ticker == settings.market_context_ticker else market
        features = builder.build(raw, ticker, context)
        all_columns = sorted(set(column for profile in ["original", "relative", "compact"] for column in feature_columns(profile)))
        if not np.isfinite(features[all_columns].to_numpy()).all():
            raise ValueError("Feature profiles must share finite values on identical evaluation rows.")
        provenance[ticker] = {"raw_sha256": sha256(raw.to_csv(index=False).encode()).hexdigest(),
                              "context_sha256": sha256(market.to_csv(index=False).encode()).hexdigest(),
                              "rows": len(features)}
        for model_name in differences:
            predictions = {candidate.name: [] for candidate in CANDIDATES}
            predictions["validation_selected"] = []
            folds = []
            splitter = ClassicalModelTrainer(settings, model_name)
            for fold, (train, validation, test) in enumerate(splitter.walk_forward_splits(features), 1):
                result, report, artifact = evaluate_fold(settings, features, train, validation, test, model_name)
                report["fold"] = fold
                folds.append(report)
                for name, pred in result.items():
                    pred["fold"] = fold
                    predictions[name].append(pred)
                predictions["validation_selected"].append(result[report["selected_candidate"]].copy())
            selected_path = splitter.registry.save_joblib(f"{model_name}_{ticker.lower()}_validation_selected", artifact)
            selections[f"{ticker}_{model_name}"] = {"folds": folds, "selected_artifact": selected_path}
            combined = {name: combine_fold_predictions(frames) for name, frames in predictions.items()}
            for name, pred in combined.items():
                metric = aggregate_classification_metrics(pred)
                benchmark = classification_baselines(pred)
                result = engine.run_with_threshold(pred, model_name, ticker, None, False)
                stem = f"{ticker.lower()}_{model_name}_{name}"
                pred.to_csv(settings.backtests_dir / f"{stem}_predictions.csv", index=False)
                result.equity_curve.to_csv(settings.backtests_dir / f"{stem}_equity.csv", index=False)
                rows.append({"ticker": ticker, "model": model_name, "candidate": name,
                             "accuracy": metric["accuracy"], "roc_auc": metric["roc_auc"],
                             "brier_score": metric["brier_score"], "log_loss": metric["log_loss"],
                             "always_up_accuracy": benchmark["always_up"]["accuracy"],
                             "prevalence_brier": benchmark["training_prevalence"]["brier_score"],
                             "net_return": result.metrics["total_return"], "max_drawdown": result.metrics["max_drawdown"],
                             "sharpe_ratio": result.metrics["sharpe_ratio"], "trades": result.metrics["trade_count"],
                             "active_session_fraction": result.metrics["active_session_fraction"],
                             "evaluation_rows": len(pred), "execution_start": str(pred.execution_date.iloc[0]),
                             "execution_end": str(pred.execution_date.iloc[-1]),
                             "fit_seconds": sum(f["timings"][name if name != "validation_selected" else f["selected_candidate"]]["fit_seconds"] for f in folds)})
            selected, original = combined["validation_selected"], combined["original_baseline"]
            paired = selected[["execution_date"]].copy()
            paired["loss_difference"] = (selected.prob_up-selected.target)**2 - (original.prob_up-original.target)**2
            differences[model_name].append(paired)
            print(f"Completed {ticker} / {model_name}", flush=True)
    uncertainty = {name: paired_brier_interval(frames, settings.random_state) for name, frames in differences.items()}
    report = {"plan": plan, "data": provenance, "results": rows, "selection": selections, "paired_uncertainty": uncertainty}
    (settings.backtests_dir / "feature_study.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(settings.backtests_dir / "feature_study.csv", index=False)
    write_report(settings, report)
    return report


def write_report(settings: Settings, report: dict) -> None:
    frame = pd.DataFrame(report["results"])
    lines = ["# Relative features and smaller-model study", "",
             "Historical research on the same cached dates as the corrected baseline. No new prospective data was used and no defaults were automatically promoted.", "",
             "Six predeclared candidates per model family: original (24), relative (24), and compact (14) features, each with baseline or regularized settings. Models and trading thresholds are selected using earlier validation only, before test predictions.", "",
             "Regularized logistic regression uses C=0.01 instead of 0.1. Regularized XGBoost uses 100 depth-2 trees instead of 300 depth-5 trees, with stronger penalties and minimum leaf weight. See study_plan.json and model code for exact settings.", "",
             "The 24-feature relative set replaces four price-level moving averages with close/average minus one, and divides three MACD values by close. The compact set removes repeated moving averages, momentum and volatility proxies; it is fixed in advance, not selected using outcomes.", "",
             "## Aggregate comparisons", "",
             "Equal-weight means across tickers; net returns are mean ticker returns, not a simulated portfolio. Fit time totals cover the fixed candidate's folds. Lower Brier and log loss are better. Static candidate rows are diagnostics, not a ranking used for selection.", "",
             "| Model | Candidate | Accuracy | AUC | Brier | Log loss | Mean net return | Mean fit seconds/ticker |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (model, candidate), group in frame.groupby(["model", "candidate"], sort=False):
        lines.append(f"| {model} | {candidate} | {group.accuracy.mean():.2%} | {group.roc_auc.mean():.3f} | {group.brier_score.mean():.4f} | {group.log_loss.mean():.4f} | {group.net_return.mean():.2%} | {group.fit_seconds.mean():.3f} |")
    lines += ["", "Selection evaluates all six candidates, so its actual search cost is their combined fit cost, not the selected-model fit time shown above. Timings are approximate local wall times.", "",
              "## Validation-selected model versus original baseline", "",
              "| Ticker | Model | Original accuracy | Selected accuracy | Always up | Original Brier | Selected Brier | Training-prior Brier | Selected net return | Trades |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (ticker, model), group in frame.groupby(["ticker", "model"], sort=False):
        original = group.loc[group.candidate.eq("original_baseline")].iloc[0]
        selected = group.loc[group.candidate.eq("validation_selected")].iloc[0]
        lines.append(f"| {ticker} | {model} | {original.accuracy:.2%} | {selected.accuracy:.2%} | {selected.always_up_accuracy:.2%} | {original.brier_score:.4f} | {selected.brier_score:.4f} | {selected.prevalence_brier:.4f} | {selected.net_return:.2%} | {selected.trades} |")
    lines += ["", "## Paired uncertainty", "", "Selected minus original Brier: negative values mean better probability predictions. Intervals resample 20-session blocks after averaging same-date loss differences across tickers. They are descriptive because these historical dates have already been inspected."]
    for model, interval in report["paired_uncertainty"].items():
        low, high = interval["block_bootstrap_95_percent_interval"]
        lines.append(f"- {model}: {interval['selected_minus_original_brier']:.4f}; 95% interval [{low:.4f}, {high:.4f}].")
    first = frame.iloc[0]
    lines += ["", "## Interpretation and limits", "",
              f"Evaluation: {first.execution_start} through {first.execution_end}, {first.evaluation_rows} sessions per candidate/ticker in this run. The final partial fold is retained.", "",
              f"Costs remain {settings.trading_fee_bps:g} bps fees plus {settings.slippage_bps:g} bps slippage per side; cash earns zero. Accuracy uses 0.5; trades use validation-selected thresholds. Improved probability scores need not improve trading returns or beat always-up accuracy.", "",
              "Each ticker/model's selected artifact stores its actual feature columns, model, threshold and cutoffs. Predictions, equity curves, full validation scores, data hashes and package versions are saved beside this report. Existing baseline and demo files are preserved.", "",
              "Next confirmation should use newly collected dates with this candidate list and selection rule frozen. Do not select the best static candidate by the test table."]
    (settings.backtests_dir / "feature_study.md").write_text("\n".join(lines) + "\n")
