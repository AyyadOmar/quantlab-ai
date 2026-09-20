"""Matched training-window study and a freeze made before fresh data is accessed."""
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
from .direction_study import (CUTOFFS, MIN_VALIDATION_DOWN_CALLS, MODEL_NAMES,
                              accuracy_interval, apply_direction_rule, down_call_audit, select_direction_rule)
from .features.builder import FeatureBuilder
from .models.base import aggregate_classification_metrics, combine_fold_predictions
from .models.classical import ClassicalModelTrainer
from .models.evaluator import evaluate_classifier

PROTOCOL = "recent_history_study_v8"
TICKERS = ("AAPL", "MSFT", "NVDA")
WINDOWS = {"full": None, "four_years": 4, "two_years": 2}


def window_training_rows(train: pd.DataFrame, window: str) -> pd.DataFrame:
    if window not in WINDOWS:
        raise ValueError("Unknown training window.")
    if train.empty or not train.date.is_monotonic_increasing or train.date.duplicated().any():
        raise ValueError("Training rows must have unique chronological dates.")
    years = WINDOWS[window]
    result = train if years is None else train.loc[train.date > train.date.max() - pd.DateOffset(years=years)]
    if len(result) < 50 or result.target.nunique() != 2:
        raise ValueError("Training window requires at least 50 rows and both classes.")
    return result.copy()


def choose_window(scores: list[dict]) -> str:
    if not scores:
        raise ValueError("Selection requires validation scores.")
    for score in scores:
        if score["window"] not in WINDOWS or not np.isfinite([score["brier_score"], score["log_loss"]]).all():
            raise ValueError("Invalid validation score.")
    order = {name: position for position, name in enumerate(WINDOWS)}
    return min(scores, key=lambda row: (-row["correct"], row["down_calls"], row["brier_score"],
                                       row["log_loss"], order[row["window"]]))["window"]


def fit_windows(settings, train, validation, model_name):
    if validation.empty or train.execution_date.max() >= validation.date.min():
        raise ValueError("Training outcomes must precede validation signal dates.")
    trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
    fitted, scores, audits = {}, [], {}
    for window in WINDOWS:
        subset = window_training_rows(train, window)
        model = trainer.fit_frame(subset)
        probabilities = trainer.predict_frame(model, validation, validation)
        selection = select_direction_rule(validation.target, probabilities)
        metric = evaluate_classifier(validation.target.to_numpy(), (probabilities >= .5).astype(int), probabilities).to_dict()
        scores.append({"window": window, "correct": selection["selected"]["correct"],
                       "down_calls": selection["selected"]["down_calls"],
                       "brier_score": metric["brier_score"], "log_loss": metric["log_loss"]})
        audits[window] = {"rows": len(subset), "start": str(subset.date.min()), "end": str(subset.date.max()),
                          "last_outcome": str(subset.execution_date.max()), "direction_selection": selection}
        fitted[window] = {"model": model, "feature_columns": trainer.feature_columns,
                          "direction_rule": selection["selected"], "window": window,
                          "training_prevalence": float(subset.target.mean()), "protocol": PROTOCOL,
                          "trained_through": str(subset.execution_date.max()),
                          "validation_through": str(validation.execution_date.max()), "classification_only": True}
    selected = choose_window(scores)
    return fitted, {"selected_window": selected, "validation_scores": scores, "training_windows": audits}


def predict_artifact(artifact, test):
    columns = ["date", "execution_date", "ticker", "target"]
    frame = test[columns].copy()
    probabilities = artifact["model"].predict_proba(test[artifact["feature_columns"]])[:, 1]
    frame["prob_up"] = probabilities
    frame["fixed_prediction"] = (probabilities >= .5).astype(int)
    frame["prediction"] = apply_direction_rule(probabilities, artifact["direction_rule"])
    frame["window"] = artifact["window"]
    frame["direction_rule"] = artifact["direction_rule"]["rule"]
    frame["direction_cutoff"] = artifact["direction_rule"]["cutoff"]
    frame["training_prevalence"] = artifact["training_prevalence"]
    return frame


def evaluate_recent_fold(settings, train, validation, test, model_name):
    if test.empty or validation.execution_date.max() >= test.date.min():
        raise ValueError("Validation outcomes must precede test signal dates.")
    artifacts, audit = fit_windows(settings, train, validation, model_name)
    predictions = {window: predict_artifact(artifact, test) for window, artifact in artifacts.items()}
    predictions["validation_selected"] = predictions[audit["selected_window"]].copy()
    for name, frame in [("validation", validation), ("test", test)]:
        audit[name] = {"start": str(frame.date.min()), "end": str(frame.date.max()),
                       "last_outcome": str(frame.execution_date.max()), "rows": len(frame)}
    return predictions, audit, artifacts


def summarize(frame, seed):
    return {**down_call_audit(frame.target, frame.prediction),
            "fixed_cutoff_accuracy": float(np.mean(frame.fixed_prediction == frame.target)),
            "brier_score": float(np.mean((frame.prob_up - frame.target)**2)),
            "uncertainty": accuracy_interval([frame], seed),
            "execution_start": str(frame.execution_date.min()), "execution_end": str(frame.execution_date.max())}


def freeze_confirmation(settings, histories, raw_hashes):
    destination = settings.backtests_dir / "confirmation_freeze.json"
    if destination.exists():
        raise FileExistsError("The confirmation freeze already exists; it cannot be replaced.")
    cutoff = min(frame.execution_date.max() for frame in histories.values())
    manifest = {"protocol": PROTOCOL, "frozen_at": datetime.now(timezone.utc).isoformat(),
                "known_through": str(cutoff), "confirmation_outcomes_after": str(cutoff),
                "companies": list(histories), "model_families": MODEL_NAMES,
                "no_refit_on_confirmation": True, "raw_hashes": raw_hashes, "artifacts": {},
                "primary": "Report each family's validation-selected window against always-up and the frozen full-history control. No family is selected using confirmation outcomes.",
                "source_hashes": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ["recent_history_study.py", "direction_study.py"]}}
    for ticker, history in histories.items():
        known = history.loc[history.execution_date <= cutoff]
        val_size = max(20, int(len(known) * settings.walk_forward_validation_size))
        validation = known.iloc[-val_size:]
        train = known.iloc[:-val_size - settings.split_gap]
        for model_name in MODEL_NAMES:
            artifacts, audit = fit_windows(settings, train, validation, model_name)
            trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
            key = f"{ticker.lower()}_{model_name}"
            entries = {}
            for window in dict.fromkeys(["full", audit["selected_window"]]):
                path = Path(trainer.registry.save_joblib(f"frozen_{key}_{window}", artifacts[window]))
                entries[window] = {"path": str(path.relative_to(settings.project_root)),
                                   "sha256": sha256(path.read_bytes()).hexdigest()}
            manifest["artifacts"][key] = {"ticker": ticker, "model": model_name,
                                           "selected_window": audit["selected_window"],
                                           "files": entries, "validation": audit,
                                           "validation_start": str(validation.date.min()),
                                           "validation_end": str(validation.date.max())}
    with destination.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    lines = ["# Frozen recent-history confirmation plan", "",
             f"Frozen at {manifest['frozen_at']}, before fetching any new prices in this experiment.",
             f"Only outcomes after {str(cutoff)[:10]} may enter confirmation. Frozen models are not refitted during confirmation.", "",
             "Each family is reported separately against always-up and its frozen full-history control. The recent-history choice uses only earlier validation accuracy, then fewer down calls, Brier, log loss, then full/four/two-year order. A fallback match is not evidence of skill.", "",
             "This is a previously unscored retrospective confirmation period, not a prediction recorded before those historical events. Results require continued prospective validation.", "",
             "| Company | Model | Selected window | Training through | Validation through |",
             "|---|---|---|---|---|"]
    for entry in manifest["artifacts"].values():
        window = entry["selected_window"]
        lines.append(f"| {entry['ticker']} | {entry['model']} | {window} | {entry['validation']['training_windows'][window]['last_outcome']} | {str(cutoff)} |")
    (settings.backtests_dir / "confirmation_plan.md").write_text("\n".join(lines) + "\n")
    return manifest


def run_recent_history_study(settings: Settings, start: str, end: str, freeze: bool = True):
    settings = replace(settings, protocol=PROTOCOL, use_cached_data=True)
    settings.ensure_directories()
    if freeze and (settings.backtests_dir / "confirmation_freeze.json").exists():
        raise FileExistsError("Existing freeze preserved. Use --historical-only to reproduce development results.")
    plan = {"protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
            "tickers": TICKERS, "start": start, "end_exclusive": end, "windows": WINDOWS,
            "window_definition": "Calendar years ending at the last training signal date, not the test date; use all available rows if less history exists.",
            "feature_set": "compact", "model_profile": "regularized", "models": MODEL_NAMES,
            "cutoffs": CUTOFFS, "minimum_validation_down_calls": MIN_VALIDATION_DOWN_CALLS,
            "selection": "Validation correct count descending, down calls ascending, Brier, log loss, then full/four/two-year order.",
            "static_windows": "Diagnostics only; never choose by test results.",
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
            "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]},
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "defaults_changed": False}
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2))
    loader, builder = MarketDataLoader(settings), FeatureBuilder(settings)
    market = loader.download("SPY", start, end)
    histories, provenance, results, selection = {}, {}, [], {}
    paired = {(model, window): [] for model in MODEL_NAMES for window in [*WINDOWS, "validation_selected"]}
    for ticker in TICKERS:
        raw = loader.download(ticker, start, end)
        history = builder.build(raw, ticker, market)
        histories[ticker] = history
        provenance[ticker] = {"raw_sha256": sha256(raw.to_csv(index=False).encode()).hexdigest(),
                              "market_sha256": sha256(market.to_csv(index=False).encode()).hexdigest()}
        for model_name in MODEL_NAMES:
            trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
            frames = {window: [] for window in [*WINDOWS, "validation_selected"]}
            audits = []
            for fold, (train, validation, test) in enumerate(trainer.walk_forward_splits(history), 1):
                predictions, audit, artifacts = evaluate_recent_fold(settings, train, validation, test, model_name)
                audit["fold"] = fold
                audit["test_scores"] = {}
                for window, frame in predictions.items():
                    frame["fold"] = fold
                    frames[window].append(frame)
                    audit["test_scores"][window] = down_call_audit(frame.target, frame.prediction)
                audits.append(audit)
            key = f"{ticker.lower()}_{model_name}"
            selection[key] = audits
            for window, folds in frames.items():
                combined = combine_fold_predictions(folds)
                combined.to_csv(settings.backtests_dir / f"{key}_{window}_predictions.csv", index=False)
                artifact = artifacts[audits[-1]["selected_window"] if window == "validation_selected" else window]
                trainer.registry.save_joblib(f"{key}_{window}", artifact)
                results.append({"ticker": ticker, "model": model_name, "window": window,
                                **summarize(combined, settings.random_state)})
                paired[model_name, window].append(combined)
            print(f"Completed training-window comparison: {ticker} / {model_name}", flush=True)
    aggregates = {}
    for (model, window), frames in paired.items():
        combined = pd.concat(frames, ignore_index=True)
        aggregates[f"{model}_{window}"] = {"model": model, "window": window,
                                             **summarize(combined, settings.random_state)}
    report = {"plan": plan, "data": provenance, "results": results, "aggregates": aggregates, "folds": selection}
    (settings.backtests_dir / "recent_history_study.json").write_text(json.dumps(report, indent=2))
    write_report(settings, report)
    if freeze:
        freeze_confirmation(settings, histories, provenance)
    return report


def write_report(settings, report):
    lines = ["# Does recent training history improve direction prediction?", "",
             "Historical development comparison: previously inspected dates, no untouched-data claim. Model defaults and website are unchanged.", "",
             "All variants use the same 14 compact features, regularized model settings, companies, validation/test dates and label gaps. Training includes all available history, up to four calendar years, or up to two years ending at the last training date. The four-year window can equal full history in early folds. Features retain their normal warm-up history.", "",
             "Each window uses the unchanged validation-only cutoff rule (seven fixed cutoffs, at least 20 down calls, always-up fallback). The primary policy selects a window by validation correct count, fewer down calls, then Brier/log loss, with full/four/two-year order as the last tie-break. The three fixed-window rows are diagnostics, not candidates selected by test results.", "",
             "| Model | Training policy | Fixed 0.5 accuracy | Selected-rule accuracy | Always-up | Difference (pp) | Correct / incorrect down | Brier | 95% difference interval (pp) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"].values():
        low, high = row["uncertainty"]["interval_95"]
        lines.append(f"| {row['model']} | {row['window']} | {row['fixed_cutoff_accuracy']:.2%} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {100*row['accuracy_gain']:+.2f} | {row['correct_down']} / {row['incorrect_down']} | {row['brier_score']:.4f} | [{100*low:+.2f}, {100*high:+.2f}] |")
    lines += ["", "## Validation-selected policy by company", "", "| Company | Model | Accuracy | Always-up | Extra correct | Down calls |", "|---|---|---:|---:|---:|---:|"]
    for row in report["results"]:
        if row["window"] == "validation_selected":
            lines.append(f"| {row['ticker']} | {row['model']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['extra_correct_vs_always_up']:+d} | {row['down_calls']} |")
    first = report["results"][0]
    lines += ["", f"Each company has {first['rows']} test sessions, {first['execution_start']} through {first['execution_end']}. Aggregate accuracy weights asset-days equally.", "",
              "Descriptive intervals resample 20-session blocks 2,000 times, keeping companies on a date together. They do not adjust for repeated historical research or multiple comparisons. Cutoff and window selection reuse validation and can overfit it. No test rows are omitted. A match achieved by predicting up every day does not demonstrate skill.", "",
              "The matched full-history control should reproduce the previous separate-company compact model. Classification cutoffs do not define a trading strategy; no profit improvement is claimed.", "",
              "A separate confirmation plan freezes choices using only known outcomes before newer prices are retrieved. Later confirmation is retrospective until predictions are actually timestamped before their outcomes. See confirmation_plan.md and, once available, confirmation.md."]
    (settings.backtests_dir / "recent_history_study.md").write_text("\n".join(lines) + "\n")
