"""Matched single-company and shared-company direction experiments."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data.loader import MarketDataLoader
from .direction_study import (
    CUTOFFS, MIN_VALIDATION_DOWN_CALLS, MODEL_NAMES, accuracy_interval,
    apply_direction_rule, down_call_audit, select_direction_rule,
)
from .features.builder import FeatureBuilder
from .models.base import aggregate_classification_metrics, combine_fold_predictions
from .models.classical import ClassicalModelTrainer


TICKERS = ("AAPL", "MSFT", "NVDA")


def pooled_training_frame(histories: dict[str, pd.DataFrame], train_dates, validation_start) -> pd.DataFrame:
    """Use exactly the same training dates for every company; never split panel rows randomly."""
    requested = pd.DatetimeIndex(train_dates)
    if requested.empty or requested.has_duplicates:
        raise ValueError("Training dates must be nonempty and unique.")
    selected = []
    for ticker, history in histories.items():
        frame = history.loc[history.date.isin(requested)].copy()
        if (len(frame) != len(requested) or frame.date.duplicated().any()
                or set(pd.to_datetime(frame.date)) != set(requested)):
            raise ValueError(f"Missing or duplicate training sessions for {ticker}.")
        if pd.to_datetime(frame.execution_date).max() >= pd.Timestamp(validation_start):
            raise ValueError("Pooled training outcomes overlap validation.")
        if not frame.ticker.eq(ticker).all():
            raise ValueError("Pooled source ticker mismatch.")
        selected.append(frame)
    if not selected:
        raise ValueError("Pooled training requires companies.")
    return pd.concat(selected, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def score_fitted_direction(trainer, fitted, history, validation, test):
    validation_prob = trainer.predict_frame(fitted, validation, history)
    selection = select_direction_rule(validation.target, validation_prob)
    pred = trainer.prediction_frame(history, test, fitted)
    pred["ticker"] = test.ticker.to_numpy()
    pred["fixed_prediction"] = pred.prediction
    pred["prediction"] = apply_direction_rule(pred.prob_up, selection["selected"])
    pred["direction_rule"] = selection["selected"]["rule"]
    pred["direction_cutoff"] = selection["selected"]["cutoff"]
    return pred, selection


def run_pooled_direction_study(settings: Settings, start: str, end: str) -> dict:
    settings = replace(settings, protocol="pooled_direction_study_v7", use_cached_data=True)
    settings.ensure_directories()
    plan = {"protocol": settings.protocol, "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_protocol": "open_to_close_v2", "tickers": TICKERS, "start": start, "end_exclusive": end,
            "cached_only": True, "feature_set": "compact", "model_profile": "regularized",
            "models": MODEL_NAMES, "scopes": ["per_company", "shared_companies"],
            "model_selection": "None: identical fixed compact/regularized specification in both scopes",
            "cutoffs": CUTOFFS, "minimum_validation_down_calls": MIN_VALIDATION_DOWN_CALLS,
            "direction_selection": "Same v6 rule, selected separately for each company using only its validation observations",
            "primary_comparison": "Report both scopes and both families against always-up on identical company/date rows",
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]},
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "direction_source_sha256": sha256(Path(__file__).with_name("direction_study.py").read_bytes()).hexdigest(),
            "historical_only": True, "defaults_changed": False, "trading_changed": False,
            "followup_reason": "v6 validation-selected cutoffs did not beat always-up; test shared training without adding features or searching settings"}
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2))
    loader, builder = MarketDataLoader(settings), FeatureBuilder(settings)
    market = loader.download(settings.market_context_ticker, start, end)
    histories, provenance = {}, {}
    for ticker in TICKERS:
        raw = loader.download(ticker, start, end)
        histories[ticker] = builder.build(raw, ticker, market)
        provenance[ticker] = {"raw_sha256": sha256(raw.to_csv(index=False).encode()).hexdigest(),
                              "market_sha256": sha256(market.to_csv(index=False).encode()).hexdigest()}
    anchor = histories[TICKERS[0]]
    # A common calendar makes cross-company labels and test observations line up exactly.
    for frame in histories.values():
        if not frame[["date", "execution_date"]].equals(anchor[["date", "execution_date"]]):
            raise ValueError("This matched experiment requires identical signal and execution calendars.")
    results, audits, aggregates = [], {}, {}
    for model_name in MODEL_NAMES:
        trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
        predictions = {(scope, ticker): [] for scope in ["per_company", "shared_companies"] for ticker in TICKERS}
        records = {key: [] for key in predictions}
        for fold, (reference_train, reference_val, reference_test) in enumerate(trainer.walk_forward_splits(anchor), 1):
            pooled = pooled_training_frame(histories, reference_train.date, reference_val.date.min())
            shared_model = trainer.fit_frame(pooled)
            for ticker, history in histories.items():
                train = history.loc[history.date.isin(reference_train.date)]
                validation = history.loc[history.date.isin(reference_val.date)]
                test = history.loc[history.date.isin(reference_test.date)]
                for scope in ["per_company", "shared_companies"]:
                    fit_frame = train if scope == "per_company" else pooled
                    fitted = trainer.fit_frame(train) if scope == "per_company" else shared_model
                    pred, selection = score_fitted_direction(trainer, fitted, history, validation, test)
                    pred["fold"] = fold
                    pred["training_prevalence"] = float(fit_frame.target.mean())
                    predictions[scope, ticker].append(pred)
                    record = {"fold": fold, "training_rows": len(fit_frame),
                              "train_start": str(fit_frame.date.min()), "train_end": str(fit_frame.date.max()),
                              "last_training_outcome": str(fit_frame.execution_date.max()),
                              "validation_start": str(validation.date.min()), "validation_end": str(validation.date.max()),
                              "last_validation_outcome": str(validation.execution_date.max()),
                              "test_start": str(test.date.min()), "test_end": str(test.date.max()),
                              "direction_selection": selection, "test_down_calls": down_call_audit(pred.target, pred.prediction)}
                    records[scope, ticker].append(record)
                    # Last-fold artifacts are research snapshots, not a live refit or promoted default.
                    trainer.registry.save_joblib(f"{ticker.lower()}_{model_name}_{scope}",
                        {"model": fitted, "feature_columns": trainer.feature_columns, "scope": scope,
                         "direction_rule": selection["selected"], "protocol": settings.protocol,
                         "trained_through": record["last_training_outcome"],
                         "validation_through": record["last_validation_outcome"], "classification_only": True})
            print(f"Completed shared training comparison: {model_name}, fold {fold}", flush=True)
        for scope in ["per_company", "shared_companies"]:
            combined_frames = []
            for ticker in TICKERS:
                frame = combine_fold_predictions(predictions[scope, ticker])
                stem = f"{ticker.lower()}_{model_name}_{scope}"
                frame.to_csv(settings.backtests_dir / f"{stem}_predictions.csv", index=False)
                audits[stem] = records[scope, ticker]
                results.append({"ticker": ticker, "model": model_name, "scope": scope,
                                **down_call_audit(frame.target, frame.prediction),
                                "fixed_cutoff_accuracy": float(np.mean(frame.fixed_prediction == frame.target)),
                                "probability_metrics": aggregate_classification_metrics(frame),
                                "training_prevalence_brier": float(np.mean((frame.training_prevalence - frame.target)**2)),
                                "uncertainty": accuracy_interval([frame], settings.random_state),
                                "execution_start": str(frame.execution_date.min()), "execution_end": str(frame.execution_date.max())})
                combined_frames.append(frame)
            full = pd.concat(combined_frames, ignore_index=True)
            aggregates[f"{model_name}_{scope}"] = {"model": model_name, "scope": scope,
                **down_call_audit(full.target, full.prediction),
                "fixed_cutoff_accuracy": float(np.mean(full.fixed_prediction == full.target)),
                "brier_score": float(np.mean((full.prob_up - full.target)**2)),
                "uncertainty": accuracy_interval(combined_frames, settings.random_state)}
    report = {"plan": plan, "data": provenance, "results": results, "aggregates": aggregates, "folds": audits}
    (settings.backtests_dir / "pooled_direction_study.json").write_text(json.dumps(report, indent=2))
    write_report(settings, report)
    return report


def write_report(settings: Settings, report: dict) -> None:
    lines = ["# Shared-company models versus always-up", "",
             "Follow-up after the v6 cutoff experiment failed to beat always-up. All dates remain historical development data already examined in prior studies; no new holdout was used.", "",
             "Both approaches use the same fixed 14 compact features and regularized settings. One trains on each company separately; the other fits once per fold on Apple, Microsoft and NVIDIA together, with equal observations per company and no company identifier. Only historical training rows are pooled. Every company shares the same date boundaries and label gaps. Scaling and imputation are fitted only on training data.", "",
             "Each company separately selects its direction cutoff on its own later validation data using the unchanged v6 rule: seven fixed cutoffs, at least 20 validation down calls, always-up fallback, ties favor fewer down calls. Model families and settings are fixed before scoring. All test days count. No earnings snapshot features or fresh data are used.", "",
             "## Matched aggregate comparisons", "",
             "These results cover three companies; compare scopes within this table. The earlier v6 aggregate covers five assets and has a different always-up rate.", "",
             "| Model | Training | Fixed 0.5 | Selected rule | Always-up | Gain (pp) | Correct / incorrect down | Brier | 95% gain interval (pp) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"].values():
        low, high = row["uncertainty"]["interval_95"]
        lines.append(f"| {row['model']} | {row['scope']} | {row['fixed_cutoff_accuracy']:.2%} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['accuracy_gain']*100:+.2f} | {row['correct_down']} / {row['incorrect_down']} | {row['brier_score']:.4f} | [{100*low:+.2f}, {100*high:+.2f}] |")
    lines += ["", "## Per company", "", "| Company | Model | Training | Selected rule | Always-up | Extra correct | Down calls |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in report["results"]:
        lines.append(f"| {row['ticker']} | {row['model']} | {row['scope']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['extra_correct_vs_always_up']:+d} | {row['down_calls']} |")
    first = report["results"][0]
    lines += ["", "## Limits and next decision", "",
              f"There are {first['rows']} test days per company, from {first['execution_start']} to {first['execution_end']}. The shared model gets three times as many rows, but correlated companies do not provide three times as much independent information.", "",
              "The descriptive intervals resample 20-session blocks, keeping companies on a date together, with 2,000 draws. They do not account for choosing this follow-up after earlier results or for multiple comparisons. A match achieved by predicting up every day does not demonstrate skill. No model is selected or promoted using this test table.", "",
              "This classification experiment changes neither default trading rules nor the website. Probability accuracy and profit are different questions. If neither approach establishes useful down-day discrimination, stop adjusting cutoffs on these dates. The next hypothesis should change the information available at prediction time and be specified before collecting a new confirmation period."]
    (settings.backtests_dir / "pooled_direction_study.md").write_text("\n".join(lines) + "\n")
