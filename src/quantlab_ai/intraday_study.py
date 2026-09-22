"""Offline, matched feature ablation on the already-used historical period."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .direction_study import (CUTOFFS, MIN_VALIDATION_DOWN_CALLS, MODEL_NAMES,
                              apply_direction_rule, down_call_audit, select_direction_rule)
from .features.profiles import feature_columns
from .models.base import combine_fold_predictions
from .models.classical import ClassicalModelTrainer
from .models.evaluator import evaluate_classifier

PROTOCOL = "intraday_study_v10"
TICKERS = ("AAPL", "MSFT", "NVDA")
VARIANTS = ("compact_control", "compact_intraday")
EXTRA_COLUMNS = ["intraday_return", "intraday_return_mean_5", "intraday_up_share_20"]
HISTORICAL_END = pd.Timestamp("2026-05-22")


def add_intraday_features(history, raw):
    """At signal-day close, today's open/close is known; tomorrow's is not."""
    history, raw = history.copy(), raw.copy()
    for frame in (history, raw):
        frame["date"] = pd.to_datetime(frame.date)
        if frame.empty or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
            raise ValueError("Expected unique chronological dates")
    raw = raw.loc[raw.date <= history.date.max()].copy()
    values = raw[["open", "close"]].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive")
    raw["intraday_return"] = raw.close / raw.open - 1
    raw["intraday_return_mean_5"] = raw.intraday_return.rolling(5, min_periods=5).mean()
    raw["intraday_up_share_20"] = (raw.close > raw.open).rolling(20, min_periods=20).mean()
    # Join source prices too: do not combine incompatible cached price snapshots.
    merged = history.merge(raw[["date", "open", "close"] + EXTRA_COLUMNS], on="date",
                           how="left", validate="one_to_one", suffixes=("", "_source"))
    for column in ("open", "close"):
        if not np.allclose(merged[column], merged[column + "_source"], rtol=1e-12, atol=1e-12):
            raise ValueError("Raw prices disagree with historical features")
    if not np.isfinite(merged[EXTRA_COLUMNS].to_numpy()).all():
        raise ValueError("Missing source sessions or insufficient warm-up; cannot drop scored rows")
    return merged.drop(columns=["open_source", "close_source"])


def evaluate_fold(settings, train, validation, test, model_name, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown feature variant")
    if (train.empty or validation.empty or test.empty
            or train.execution_date.max() >= validation.date.min()
            or validation.execution_date.max() >= test.date.min()):
        raise ValueError("Training and validation outcomes must precede later signal dates")
    trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
    trainer.feature_columns = feature_columns("compact") + (EXTRA_COLUMNS if variant == "compact_intraday" else [])
    model = trainer.fit_frame(train)
    probabilities = trainer.predict_frame(model, validation, validation)
    selection = select_direction_rule(validation.target, probabilities)
    # Freeze the rule before accessing test probabilities; test labels are only scored later.
    probabilities = trainer.predict_frame(model, test, test)
    frame = test[["date", "execution_date", "ticker", "target"]].copy()
    frame["prob_up"] = probabilities
    frame["fixed_prediction"] = (probabilities >= .5).astype(int)
    frame["prediction"] = apply_direction_rule(probabilities, selection["selected"])
    frame["training_prevalence"] = float(train.target.mean())
    frame["variant"] = variant
    frame["direction_rule"] = selection["selected"]["rule"]
    frame["direction_cutoff"] = selection["selected"]["cutoff"]
    audit = {"selection": selection, "feature_columns": trainer.feature_columns}
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        audit[name] = {"rows": len(part), "start": str(part.date.min()), "end": str(part.date.max()),
                       "last_outcome": str(part.execution_date.max())}
    return frame, audit, {"model": model, "feature_columns": trainer.feature_columns,
                          "direction_rule": selection["selected"], "protocol": PROTOCOL,
                          "variant": variant, "classification_only": True,
                          "trained_through": audit["train"]["last_outcome"],
                          "validation_through": audit["validation"]["last_outcome"]}


def paired_gain(candidate, control, seed=42, repetitions=2000):
    keys = ["ticker", "execution_date"]
    for frame in (candidate, control):
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate company/session predictions")
    pair = candidate[keys + ["target", "prediction"]].merge(
        control[keys + ["target", "prediction"]], on=keys, how="outer",
        suffixes=("_candidate", "_control"), validate="one_to_one", indicator=True)
    if not (pair._merge == "both").all() or not (pair.target_candidate == pair.target_control).all():
        raise ValueError("Comparisons require identical company/session outcomes")
    pair["difference"] = ((pair.prediction_candidate == pair.target_candidate).astype(int)
                          - (pair.prediction_control == pair.target_control).astype(int))
    daily = pair.groupby("execution_date", sort=True).difference.mean().to_numpy()
    n = len(daily)
    if not n:
        raise ValueError("Empty comparison")
    block = min(20, n)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        draws.append(daily[indices].mean())
    return {"gain": float(pair.difference.mean()), "interval_95": np.quantile(draws, [.025, .975]).tolist(),
            "sessions": n, "block_sessions": block, "repetitions": repetitions,
            "interpretation": "Descriptive historical interval, not adjusted for repeated research."}


def metrics(frame):
    result = evaluate_classifier(frame.target.to_numpy(), frame.prediction.to_numpy(), frame.prob_up.to_numpy()).to_dict()
    down = frame.target == 0
    return {**result, **down_call_audit(frame.target, frame.prediction),
            "down_recall": float((frame.loc[down, "prediction"] == 0).mean()),
            "fixed_accuracy": float((frame.fixed_prediction == frame.target).mean()),
            "always_down_accuracy": float(1 - frame.target.mean()),
            "training_majority_accuracy": float(((frame.training_prevalence >= .5).astype(int) == frame.target).mean()),
            "training_prevalence_brier": float(((frame.training_prevalence - frame.target) ** 2).mean())}


def verify_control(frame, old):
    columns = ["date", "execution_date", "ticker", "target", "prediction", "fixed_prediction", "fold"]
    for data in (frame, old):
        for col in ("date", "execution_date"):
            data[col] = pd.to_datetime(data[col])
    pd.testing.assert_frame_equal(frame[columns].reset_index(drop=True), old[columns].reset_index(drop=True))
    # XGBoost emits float32 and pandas writes its shortest round-trip decimal.
    # Restore that precision before comparing with a CSV parsed as float64.
    restored = old.prob_up.to_numpy().astype(frame.prob_up.dtype)
    np.testing.assert_allclose(frame.prob_up, restored, rtol=0, atol=1e-12)


def run(settings):
    settings = replace(settings, protocol=PROTOCOL, use_cached_data=True)
    settings.ensure_directories()
    root = settings.project_root
    histories, sources = {}, {}
    for ticker in TICKERS:
        history_path = root / "data/processed/recent_history_study_v8" / f"{ticker}_features.csv"
        raw_path = root / "data/raw" / f"{ticker}_raw.csv"
        history = pd.read_csv(history_path, parse_dates=["date", "execution_date"])
        if history.execution_date.max() > HISTORICAL_END:
            raise ValueError("Development inputs include reserved newer dates")
        histories[ticker] = add_intraday_features(history, pd.read_csv(raw_path))
        sources[ticker] = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in (history_path, raw_path)}
    plan = {"protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
            "historical_only": True, "latest_allowed_outcome": str(HISTORICAL_END.date()),
            "tickers": TICKERS, "model_families": MODEL_NAMES, "variants": VARIANTS,
            "features_added_as_one_group": EXTRA_COLUMNS, "training_window": "full expanding history",
            "cutoffs": CUTOFFS, "minimum_validation_down_calls": MIN_VALIDATION_DOWN_CALLS,
            "selection": "Same validation-only direction rule as v8; no feature or family selection by test scores.",
            "comparison": "Each added-feature model versus matched compact control, always-up and always-down.",
            "source_hash": sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": sources,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
            "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")}}
    # Record the bounded plan before fitting any models or inspecting new study scores.
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    collected, audits, rows = {}, {}, []
    for ticker, history in histories.items():
        for model_name in MODEL_NAMES:
            trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
            for variant in VARIANTS:
                frames, folds = [], []
                for fold, (train, validation, test) in enumerate(trainer.walk_forward_splits(history), 1):
                    frame, audit, artifact = evaluate_fold(settings, train, validation, test, model_name, variant)
                    frame["fold"], audit["fold"] = fold, fold
                    frames.append(frame)
                    folds.append(audit)
                combined = combine_fold_predictions(frames)
                key = f"{ticker.lower()}_{model_name}_{variant}"
                if variant == "compact_control":
                    old_path = root / "backtesting/recent_history_study_v8" / f"{ticker.lower()}_{model_name}_full_predictions.csv"
                    verify_control(combined, pd.read_csv(old_path))
                combined.to_csv(settings.backtests_dir / f"{key}_predictions.csv", index=False)
                trainer.registry.save_joblib(key, artifact)
                collected[ticker, model_name, variant] = combined
                audits[key] = folds
                rows.append({"ticker": ticker, "model": model_name, "variant": variant, **metrics(combined)})
            print(f"Completed matched intraday comparison: {ticker} / {model_name}", flush=True)
    aggregates, comparisons = [], {}
    for model_name in MODEL_NAMES:
        by_variant = {variant: pd.concat([collected[t, model_name, variant] for t in TICKERS], ignore_index=True)
                      for variant in VARIANTS}
        for variant, frame in by_variant.items():
            aggregates.append({"model": model_name, "variant": variant, **metrics(frame)})
        comparisons[model_name] = paired_gain(by_variant[VARIANTS[1]], by_variant[VARIANTS[0]], settings.random_state)
    report = {"plan": plan, "results": rows, "aggregates": aggregates,
              "paired_comparisons": comparisons, "folds": audits, "controls_reproduced": 6}
    (settings.backtests_dir / "intraday_study.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(settings, report)
    return report


def write_report(settings, report):
    lines = ["# Intraday feature-group experiment", "",
             "Historical development only: previously inspected outcomes, January 24, 2023–May 22, 2026. "
             "Three companies, 836 sessions each (2,508 company-days). No new prices were downloaded, "
             "and the May–September confirmation and prospective ledger were not used.", "",
             "The 14-feature compact control is compared with the same model plus completed-session "
             "open-to-close return, its trailing five-session arithmetic mean, and trailing 20-session "
             "share of up intraday sessions. All include the completed signal day. Rolling inputs use "
             "prior cached prices for warm-up; no test days are dropped. Flat sessions count as down.", "",
             "Both regularized model families retain full expanding training history, validation/test "
             "boundaries and one-session label gaps. Cutoffs and the always-up fallback are selected "
             "only on validation using the existing rule. There is no new cutoff grid, window search "
             "or feature-subset search. All six controls reproduce the prior full-history predictions "
             "and probabilities (restoring native numeric precision after CSV parsing; absolute tolerance 1e-12).", "",
             "| Model | Inputs | Accuracy | Fixed 0.5 accuracy | Balanced accuracy | Down recall | Brier | ROC AUC |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['model']} | {row['variant']} | {row['accuracy']:.2%} | {row['fixed_accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['down_recall']:.2%} | {row['brier_score']:.4f} | {row['roc_auc']:.4f} |")
    first = report["aggregates"][0]
    lines += ["", f"Same-date baselines: always-up **{first['always_up_accuracy']:.2%}**, always-down **{first['always_down_accuracy']:.2%}**, training-majority **{first['training_majority_accuracy']:.2%}**. "
              f"Training-prevalence Brier: {first['training_prevalence_brier']:.4f}; lower is better.", "",
              "## Added features versus the matched control", ""]
    for model, result in report["paired_comparisons"].items():
        low, high = result["interval_95"]
        lines.append(f"- {model}: {100 * result['gain']:+.2f} percentage points; descriptive 95% interval [{100 * low:+.2f}, {100 * high:+.2f}].")
    lines += ["", "Intervals use 2,000 samples of 20-session blocks with companies on a date kept together. "
              "They are not adjusted for repeated historical research or multiple comparisons. "
              "They are not confirmation of a reliable advantage.", "", "## Every company and model", "",
              "| Company | Model | Inputs | Accuracy | Always-up | Correct / incorrect down calls |",
              "|---|---|---|---:|---:|---:|"]
    for row in report["results"]:
        lines.append(f"| {row['ticker']} | {row['model']} | {row['variant']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['correct_down']} / {row['incorrect_down']} |")
    lines += ["", "## Reproduce", "", "Run from the project root with the existing local v8 inputs:", "", "```sh",
              'PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \\',
              '  .venv/bin/python -m quantlab_ai.intraday_study', "```", "",
              "Local JSON records settings, packages, input/source hashes and every fold's validation "
              "rule. Model artifacts and predictions are isolated under intraday_study_v10 and excluded "
              "from Git. This command never fetches market data. It requires existing v8 historical "
              "features, raw prices and control predictions; a fresh Git clone alone is insufficient.", "",
              "The website, default models, earlier frozen confirmation artifacts and prospective "
              "forecasts are unchanged. Classification scores do not establish profitability after costs.", ""]
    (settings.project_root / "docs/intraday_study.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(Settings())
