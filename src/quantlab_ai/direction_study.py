"""Validation-only classification cutoffs, with an explicit always-up fallback."""
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
from .feature_study import CANDIDATES, Candidate, choose_candidate
from .features.builder import FeatureBuilder
from .models.base import aggregate_classification_metrics, combine_fold_predictions
from .models.classical import ClassicalModelTrainer
from .models.evaluator import evaluate_classifier


CUTOFFS = (.35, .40, .45, .50, .55, .60, .65)
MIN_VALIDATION_DOWN_CALLS = 20
MODEL_NAMES = ("logistic_regression", "xgboost")


def _validate_arrays(target, probabilities):
    target, probabilities = np.asarray(target), np.asarray(probabilities, dtype=float)
    if (target.ndim != 1 or probabilities.ndim != 1 or len(target) != len(probabilities)
            or not len(target) or not np.isin(target, [0, 1]).all()
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Expected matching nonempty binary labels and finite probabilities in [0, 1].")
    return target, probabilities


def select_direction_rule(target, probabilities, cutoffs=CUTOFFS,
                          minimum_down_calls=MIN_VALIDATION_DOWN_CALLS) -> dict:
    """Accept validation arrays only. Ties prefer always-up, then fewer down calls."""
    target, probabilities = _validate_arrays(target, probabilities)
    if minimum_down_calls < 1 or not cutoffs or any(not 0 < cutoff < 1 for cutoff in cutoffs):
        raise ValueError("Invalid predeclared cutoff grid or minimum down-call count.")
    scores = [{"rule": "always_up", "cutoff": None, "correct": int(target.sum()),
               "down_calls": 0, "eligible": True}]
    for cutoff in cutoffs:
        down = probabilities < cutoff
        scores.append({"rule": "cutoff", "cutoff": float(cutoff),
                       "correct": int(np.sum((~down).astype(int) == target)),
                       "down_calls": int(down.sum()),
                       "eligible": bool(down.sum() >= minimum_down_calls)})
    chosen = min((row for row in scores if row["eligible"]),
                 key=lambda row: (-row["correct"], row["down_calls"], row["cutoff"] or 0))
    return {"selected": dict(chosen), "validation_rows": len(target), "scores": scores}


def apply_direction_rule(probabilities, rule: dict) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    _validate_arrays(np.ones(len(probabilities)), probabilities)
    if rule["rule"] == "always_up":
        return np.ones(len(probabilities), dtype=int)
    if rule["rule"] != "cutoff" or not 0 < rule["cutoff"] < 1:
        raise ValueError("Unknown direction rule.")
    return (probabilities >= rule["cutoff"]).astype(int)


def down_call_audit(target, prediction) -> dict:
    target, prediction = _validate_arrays(target, prediction)
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError("Predictions must be binary.")
    down = prediction == 0
    correct_down = int(np.sum(down & (target == 0)))
    incorrect_down = int(np.sum(down & (target == 1)))
    return {"rows": len(target), "accuracy": float(np.mean(target == prediction)),
            "always_up_accuracy": float(np.mean(target)), "down_calls": int(down.sum()),
            "correct_down": correct_down, "incorrect_down": incorrect_down,
            "down_precision": float(correct_down / down.sum()) if down.any() else None,
            "extra_correct_vs_always_up": correct_down - incorrect_down,
            "accuracy_gain": float((correct_down - incorrect_down) / len(target))}


def accuracy_interval(frames: list[pd.DataFrame], seed: int, repetitions: int = 2000) -> dict:
    """Resample dates together across assets, in circular 20-session blocks."""
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["ticker", "execution_date"]).any():
        raise ValueError("Duplicate asset/date observations in paired comparison.")
    combined["difference"] = ((combined.prediction == combined.target).astype(int)
                              - (combined.target == 1).astype(int))
    daily = combined.groupby("execution_date", sort=True).difference.mean().to_numpy()
    if not len(daily):
        raise ValueError("Uncertainty requires observations.")
    n, block = len(daily), min(20, len(daily))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        estimates.append(float(daily[indices].mean()))
    return {"accuracy_gain": float(daily.mean()),
            "interval_95": np.quantile(estimates, [.025, .975]).tolist(),
            "sessions": n, "block_sessions": block, "repetitions": repetitions,
            "interpretation": "Descriptive only: previously inspected historical dates; not adjusted for multiple experiments."}


def evaluate_direction_fold(settings, history, train, validation, test, model_name,
                            candidates=CANDIDATES):
    """Select model on validation Brier, then direction rule on validation accuracy."""
    models, trainers, val_probabilities, scores = {}, {}, {}, []
    for candidate in candidates:
        trainer = ClassicalModelTrainer(settings, model_name, candidate.feature_set, candidate.model_profile)
        model = trainer.fit_frame(train)
        prob = trainer.predict_frame(model, validation, history)
        metrics = evaluate_classifier(validation.target.to_numpy(), (prob >= .5).astype(int), prob).to_dict()
        scores.append({"candidate": candidate.name, "brier_score": metrics["brier_score"],
                       "log_loss": metrics["log_loss"], "feature_count": len(trainer.feature_columns)})
        models[candidate.name], trainers[candidate.name] = model, trainer
        val_probabilities[candidate.name] = prob
    selected = choose_candidate(scores)
    selection = select_direction_rule(validation.target.to_numpy(), val_probabilities[selected])
    # No test probabilities or outcomes are consulted until both decisions are final.
    trainer, model = trainers[selected], models[selected]
    pred = trainer.prediction_frame(history, test, model)
    pred["fixed_prediction"] = pred.prediction
    pred["prediction"] = apply_direction_rule(pred.prob_up.to_numpy(), selection["selected"])
    pred["ticker"] = test.ticker.to_numpy()
    pred["candidate"] = selected
    pred["direction_rule"] = selection["selected"]["rule"]
    pred["direction_cutoff"] = selection["selected"]["cutoff"]
    pred["training_prevalence"] = float(train.target.mean())
    report = {"selected_candidate": selected, "validation_model_scores": scores,
              "direction_selection": selection}
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        report[name] = {"start": str(frame.date.min()), "end": str(frame.date.max()),
                        "last_outcome": str(frame.execution_date.max()), "rows": len(frame)}
    artifact = {"model": model, "feature_columns": trainer.feature_columns,
                "candidate": selected, "direction_rule": selection["selected"],
                "protocol": settings.protocol, "classification_only": True,
                "trained_through": report["train"]["last_outcome"],
                "validation_through": report["validation"]["last_outcome"]}
    return pred, report, artifact


def run_direction_study(settings: Settings, tickers: list[str], start: str, end: str) -> dict:
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Supply a nonempty list of distinct tickers.")
    # Deliberately offline: this development study must not consume fresh holdout dates.
    settings = replace(settings, protocol="direction_study_v6", use_cached_data=True)
    settings.ensure_directories()
    plan = {"protocol": settings.protocol, "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_protocol": "open_to_close_v2", "tickers": tickers,
            "start": start, "end_exclusive": end, "cached_only": True,
            "model_candidates": [asdict(candidate) for candidate in CANDIDATES],
            "model_selection": "Validation Brier, then log loss, feature count, name (unchanged from v3)",
            "direction_cutoffs": CUTOFFS, "minimum_validation_down_calls": MIN_VALIDATION_DOWN_CALLS,
            "direction_selection": "Max validation correct count; ties fewer down calls then lower cutoff; always-up eligible",
            "primary_comparison": "Validation-selected direction vs always-up across all asset-days; report both model families",
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]},
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "historical_only": True, "defaults_changed": False,
            "no_trading_threshold_changes": True}
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2))
    loader, builder = MarketDataLoader(settings), FeatureBuilder(settings)
    market = loader.download(settings.market_context_ticker, start, end)
    results, audits, provenance = [], {}, {}
    paired = {model: [] for model in MODEL_NAMES}
    for ticker in tickers:
        raw = market if ticker == settings.market_context_ticker else loader.download(ticker, start, end)
        history = builder.build(raw, ticker, None if ticker == settings.market_context_ticker else market)
        provenance[ticker] = {"raw_sha256": sha256(raw.to_csv(index=False).encode()).hexdigest(),
                              "market_sha256": sha256(market.to_csv(index=False).encode()).hexdigest(),
                              "feature_rows": len(history)}
        for model_name in MODEL_NAMES:
            trainer = ClassicalModelTrainer(settings, model_name)
            frames, folds = [], []
            for fold, (train, val, test) in enumerate(trainer.walk_forward_splits(history), 1):
                pred, audit, artifact = evaluate_direction_fold(settings, history, train, val, test, model_name)
                pred["fold"], audit["fold"] = fold, fold
                audit["test_down_calls"] = down_call_audit(pred.target, pred.prediction)
                frames.append(pred)
                folds.append(audit)
            combined = combine_fold_predictions(frames)
            stem = f"{ticker.lower()}_{model_name}"
            combined.to_csv(settings.backtests_dir / f"{stem}_predictions.csv", index=False)
            artifact_path = trainer.registry.save_joblib(f"{stem}_direction", artifact)
            audits[stem] = {"folds": folds, "last_fold_artifact": artifact_path}
            row = {"ticker": ticker, "model": model_name, **down_call_audit(combined.target, combined.prediction),
                   "fixed_cutoff_accuracy": float(np.mean(combined.fixed_prediction == combined.target)),
                   "probability_metrics": aggregate_classification_metrics(combined),
                   "training_prevalence_brier": float(np.mean((combined.training_prevalence - combined.target) ** 2)),
                   "uncertainty": accuracy_interval([combined], settings.random_state),
                   "folds_with_positive_gain": sum(f["test_down_calls"]["extra_correct_vs_always_up"] > 0 for f in folds),
                   "fold_count": len(folds),
                   "always_up_selected_folds": sum(f["direction_selection"]["selected"]["rule"] == "always_up" for f in folds),
                   "execution_start": str(combined.execution_date.min()),
                   "execution_end": str(combined.execution_date.max())}
            results.append(row)
            paired[model_name].append(combined)
            print(f"Completed {ticker} / {model_name}: accuracy {row['accuracy']:.2%}, always-up {row['always_up_accuracy']:.2%}", flush=True)
    aggregates = {}
    for model_name, frames in paired.items():
        combined = pd.concat(frames, ignore_index=True)
        aggregates[model_name] = {**down_call_audit(combined.target, combined.prediction),
                                 "fixed_cutoff_accuracy": float(np.mean(combined.fixed_prediction == combined.target)),
                                 "uncertainty": accuracy_interval(frames, settings.random_state)}
    report = {"plan": plan, "data": provenance, "results": results, "aggregates": aggregates, "selection": audits}
    (settings.backtests_dir / "direction_study.json").write_text(json.dumps(report, indent=2))
    write_report(settings, report)
    return report


def write_report(settings: Settings, report: dict) -> None:
    lines = ["# Predicting down versus always-up", "",
             "Historical development experiment on previously inspected dates. No fresh holdout was downloaded or evaluated. Model defaults and website results were not changed.", "",
             "Models retain the six price-feature/regularization candidates and validation Brier selection from v3. Only the classification decision changes: choose among seven predeclared probability cutoffs (0.35 to 0.65, step 0.05) and always-up using validation accuracy. Cutoffs require at least 20 validation down calls; ties prefer fewer down calls, then a lower cutoff. Always-up wins accuracy ties against any eligible cutoff.", "",
             "An up prediction means the next session closes above its open. A down prediction includes flat sessions. Every test session is scored; no abstentions or discarded low-confidence days. The test result does not choose the rule. Earlier test dates may enter later training once historical, as in the existing expanding-window protocol.", "",
             "## Aggregate results", "",
             "Pooled asset-day accuracy on identical dates, with both prespecified model families reported. Correct down calls add successes versus always-up; incorrect down calls remove successes.", "",
             "| Model | Fixed 0.5 | Selected rule | Always-up | Gain (percentage points) | Correct / incorrect down | 95% gain interval (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["aggregates"].items():
        low, high = row["uncertainty"]["interval_95"]
        lines.append(f"| {name} | {row['fixed_cutoff_accuracy']:.2%} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {100 * row['accuracy_gain']:+.2f} | {row['correct_down']} / {row['incorrect_down']} | [{100*low:+.2f}, {100*high:+.2f}] |")
    lines += ["", "## Per asset", "", "| Asset | Model | Selected rule | Always-up | Extra correct | Down calls | Positive folds / all folds | Always-up fallback folds |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        lines.append(f"| {row['ticker']} | {row['model']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['extra_correct_vs_always_up']:+d} | {row['down_calls']} | {row['folds_with_positive_gain']} / {row['fold_count']} | {row['always_up_selected_folds']} |")
    first = report["results"][0]
    lines += ["", "## Interpretation", "",
              f"Evaluation outcomes span {first['execution_start']} to {first['execution_end']}. See JSON for each asset's row count, full cutoff scores, fold boundaries, and data hashes.", "",
              "Intervals use 2,000 circular bootstrap samples of 20-session blocks, keeping all assets on a date together. They are descriptive, not proof of an edge: the dates have been examined in earlier research, validation is reused for model and cutoff selection, and intervals do not correct for multiple experiments. Assets are correlated and are not independent replications.", "",
              "A cutoff changes classification decisions, not probabilities, their Brier scores, or ranking ability. An always-up fallback can match the benchmark without finding any down-day signal. A validation win does not guarantee a test win. Default trading thresholds, costs, trading signals, and website figures remain unchanged; no profit claim follows from this classification study.", "",
              "If the selected rules fail to improve, keep the baseline and test a separately specified shared model across assets. Any eventual candidate needs a frozen rule and confirmation on untouched dates before promotion."]
    (settings.backtests_dir / "direction_study.md").write_text("\n".join(lines) + "\n")
