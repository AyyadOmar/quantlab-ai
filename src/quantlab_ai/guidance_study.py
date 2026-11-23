"""Matched, offline guidance-content study; never consumes fresh outcome data."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data.guidance import CONTENT_COLUMNS, EVENT_COLUMNS, build_event_features, load_documents
from .direction_study import CUTOFFS, MIN_VALIDATION_DOWN_CALLS, MODEL_NAMES, apply_direction_rule, select_direction_rule
from .features.profiles import feature_columns
from .intraday_study import HISTORICAL_END, TICKERS, metrics, paired_gain, verify_control
from .models.base import combine_fold_predictions
from .models.classical import ClassicalModelTrainer

PROTOCOL = "guidance_study_v11"
VARIANTS = {"price_control": [], "release_timing": ["earnings_release_recent"], "guidance_content": EVENT_COLUMNS}


def evaluate_fold(settings, train, validation, test, model_name, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown guidance variant")
    if (train.empty or validation.empty or test.empty or train.execution_date.max() >= validation.date.min()
            or validation.execution_date.max() >= test.date.min()):
        raise ValueError("Outcome dates overlap later signal dates")
    trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
    trainer.feature_columns = feature_columns("compact") + VARIANTS[variant]
    fitted = trainer.fit_frame(train)
    selection = select_direction_rule(validation.target, trainer.predict_frame(fitted, validation, validation))
    probabilities = trainer.predict_frame(fitted, test, test)
    frame = test[["date", "execution_date", "ticker", "target", "earnings_release_recent"]].copy()
    frame["prob_up"] = probabilities
    frame["fixed_prediction"] = (probabilities >= .5).astype(int)
    frame["prediction"] = apply_direction_rule(probabilities, selection["selected"])
    frame["training_prevalence"] = float(train.target.mean())
    frame["variant"] = variant
    audit = {"selection": selection, "feature_columns": trainer.feature_columns}
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        audit[name] = {"start": str(part.date.min()), "end": str(part.date.max()),
                       "last_outcome": str(part.execution_date.max()), "rows": len(part)}
    artifact = {"model": fitted, "feature_columns": trainer.feature_columns,
                "direction_rule": selection["selected"], "protocol": PROTOCOL, "variant": variant,
                "trained_through": audit["train"]["last_outcome"], "classification_only": True}
    return frame, audit, artifact


def run(settings):
    settings = replace(settings, protocol=PROTOCOL, use_cached_data=True)
    settings.ensure_directories()
    root = settings.project_root
    documents, manifest = load_documents(root)
    histories, inputs = {}, {}
    for ticker in TICKERS:
        path = root / "data/processed/recent_history_study_v8" / f"{ticker}_features.csv"
        history = pd.read_csv(path, parse_dates=["date", "execution_date"])
        if history.execution_date.max() > HISTORICAL_END:
            raise ValueError("Reserved newer outcomes cannot enter development")
        histories[ticker] = build_event_features(history, documents)
        inputs[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
    inputs["data/guidance_v11/source_manifest.json"] = sha256((root / "data/guidance_v11/source_manifest.json").read_bytes()).hexdigest()
    plan = {"protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
            "historical_only": True, "outcomes_through": "2026-05-22", "tickers": TICKERS,
            "models": MODEL_NAMES, "variants": VARIANTS, "event_window_signal_sessions": 5,
            "available_from": "Calendar day after SEC filing; no same-day announcement use",
            "cutoffs": CUTOFFS, "minimum_validation_down_calls": MIN_VALIDATION_DOWN_CALLS,
            "selection": "Unchanged validation-only direction rule; no test-based feature/family selection",
            "training_window": "full expanding history", "inputs": inputs,
            "document_count": len(documents), "announcement_coverage": "All indexed 8-K item 2.02 exhibits; earnings calls and 8-K amendments are outside scope",
            "source_hashes": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), Path(__file__).parent / "data/guidance.py"]},
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
            "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost")}}
    # Record design before fitting or inspecting this experiment's prediction scores.
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (root / "data/guidance_v11/extracted_events.json").write_text(json.dumps(documents, indent=2) + "\n")
    frames, audits, rows = {}, {}, []
    for ticker, history in histories.items():
        for model_name in MODEL_NAMES:
            trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
            for variant in VARIANTS:
                predictions, folds = [], []
                for fold, (train, validation, test) in enumerate(trainer.walk_forward_splits(history), 1):
                    prediction, audit, artifact = evaluate_fold(settings, train, validation, test, model_name, variant)
                    prediction["fold"], audit["fold"] = fold, fold
                    predictions.append(prediction)
                    folds.append(audit)
                combined = combine_fold_predictions(predictions)
                stem = f"{ticker.lower()}_{model_name}_{variant}"
                if variant == "price_control":
                    old = root / "backtesting/recent_history_study_v8" / f"{ticker.lower()}_{model_name}_full_predictions.csv"
                    verify_control(combined, pd.read_csv(old))
                combined.to_csv(settings.backtests_dir / f"{stem}_predictions.csv", index=False)
                trainer.registry.save_joblib(stem, artifact)
                frames[ticker, model_name, variant] = combined
                audits[stem] = folds
                rows.append({"ticker": ticker, "model": model_name, "variant": variant, **metrics(combined)})
            print(f"Completed guidance comparison: {ticker} / {model_name}", flush=True)
    aggregates, comparisons, event_scores = [], {}, []
    for model_name in MODEL_NAMES:
        combined = {variant: pd.concat([frames[t, model_name, variant] for t in TICKERS], ignore_index=True) for variant in VARIANTS}
        for variant, frame in combined.items():
            aggregates.append({"model": model_name, "variant": variant, **metrics(frame)})
            subset = frame.loc[frame.earnings_release_recent == 1]
            event_scores.append({"model": model_name, "variant": variant, **metrics(subset)})
        comparisons[model_name] = {reference: paired_gain(combined["guidance_content"], combined[reference], settings.random_state)
                                   for reference in ("price_control", "release_timing")}
        comparisons[model_name]["always_up"] = paired_gain(combined["guidance_content"], combined["guidance_content"].assign(prediction=1), settings.random_state)
    coverage = {ticker: {"documents": sum(r["ticker"] == ticker for r in documents),
                         **{col: sum(r[col] for r in documents if r["ticker"] == ticker) for col in CONTENT_COLUMNS + ["guidance_forward_growth_known"]}}
                for ticker in TICKERS}
    report = {"plan": plan, "coverage": coverage, "results": rows, "aggregates": aggregates,
              "event_window_scores": event_scores, "comparisons": comparisons, "folds": audits,
              "controls_reproduced": 6}
    (settings.backtests_dir / "guidance_study.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(root, report)
    return report


def write_report(root, report):
    lines = ["# Earnings announcement and guidance-content study", "",
             "Historical development on already-inspected January 24, 2023–May 22, 2026 outcomes. "
             "All 2,508 company-days are scored (836 each for AAPL, MSFT and NVDA). "
             "The newer confirmation period and prospective ledger were not used.", "",
             "**Interpretation:** the subsequent [zero-content diagnostic](guidance_placebo.md) "
             "reproduced all 2,508 XGBoost direction calls with the guidance values removed. "
             "Its two-prediction lead over always-up is not evidence that guidance improved accuracy. "
             "Logistic regression worsened. Neither model is promoted.", "",
             "## Matched comparisons", "",
             "Price control uses the existing 14 compact inputs. Release timing adds a five-session "
             "earnings-release indicator. Guidance content adds four literal flags: numerical revenue "
             "outlook, explicit guidance raise, reduction or withdrawal. It also adds the next-quarter "
             "revenue outlook midpoint divided by the just-reported quarter's revenue, minus one, "
             "with a separate availability flag. Missing values use zero plus that flag. "
             "This is sequential revenue growth in management's outlook, not analyst surprise. Timing is a separate control "
             "so an event-date effect is not mistaken for useful announcement content. All models use "
             "the same full expanding training history, regularized settings and validation-only cutoff "
             "selection. All six price controls reproduce the prior predictions and native-precision probabilities.", "",
             "| Model | Inputs | Accuracy | Balanced accuracy | Down recall | Brier | ROC AUC |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['model']} | {row['variant']} | {row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['down_recall']:.2%} | {row['brier_score']:.4f} | {row['roc_auc']:.4f} |")
    first = report["aggregates"][0]
    lines += ["", f"Identical-date baselines: always-up **{first['always_up_accuracy']:.2%}**, always-down **{first['always_down_accuracy']:.2%}**. "
              f"Training-prevalence Brier: {first['training_prevalence_brier']:.4f} (lower is better).", "",
              "## Guidance content versus controls and always-up", ""]
    for model, comparisons in report["comparisons"].items():
        for reference, result in comparisons.items():
            low, high = result["interval_95"]
            lines.append(f"- {model} versus {reference}: {100*result['gain']:+.2f} percentage points; descriptive 95% interval [{100*low:+.2f}, {100*high:+.2f}].")
    lines += ["", "Intervals resample 20-session blocks 2,000 times, keeping companies on a date together. "
              "These are exploratory intervals, without correction for repeated research or multiple comparisons.", "",
              "## Source coverage and extraction", "",
              "| Company | Indexed announcements | Numeric revenue outlook | Explicit raise | Explicit reduction | Explicit withdrawal | Comparable quarterly outlook |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for ticker, row in report["coverage"].items():
        lines.append(f"| {ticker} | {row['documents']} | " + " | ".join(str(row[c]) for c in CONTENT_COLUMNS + ["guidance_forward_growth_known"]) + " |")
    lines += ["", "Coverage is relative to the fixed 2018–May 2026 SEC 8-K item 2.02 index, not all company news. "
              "The public SEC earnings exhibits were retrieved as readable narrative text, linked to the "
              "indexed accession, and hashed locally. Legal disclaimers and appended financial tables "
              "are outside the narrative extraction where a disclaimer boundary exists. Missing or partial "
              "narratives stop the study rather than becoming zero-news observations. Source files, URLs, "
              "collection times and extraction evidence remain in the local research dataset.", "",
              "An unflagged announcement means no supported literal phrase was detected in that document. "
              "It does not establish that management gave no guidance in an earnings call. The extractor "
              "does not infer analyst surprise, guidance revisions from a different forecast period, "
              "or sentiment from optimistic prose. Negated and hypothetical revision statements are excluded. "
              "Sparse flags limit what this experiment can establish about richer news models.", "",
              "## Availability and limitations", "",
              "Announcement features begin on the calendar day after SEC filing and last five signal "
              "sessions. Signals are still made after a completed session for the next session's "
              "open-to-close outcome. This deliberately delayed rule can miss the first market reaction. "
              "Current downloads of archived filings support retrospective development, not a live "
              "record. Filing amendments, conference-call-only guidance and independently reported news "
              "are outside this first study. No fresh prices were downloaded or new defaults promoted.", "",
              "## Event-window diagnostic", "",
              "This subset is reported alongside all-day accuracy, never as a replacement score. "
              "Each model is compared on the same event-window company-days.", "",
              "| Model | Inputs | Rows | Accuracy | Same-row always-up | Same-row always-down |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["event_window_scores"]:
        lines.append(f"| {row['model']} | {row['variant']} | {row['rows']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['always_down_accuracy']:.2%} |")
    lines += ["", "## Every company", "", "| Company | Model | Inputs | Accuracy | Always-up |", "|---|---|---|---:|---:|"]
    for row in report["results"]:
        lines.append(f"| {row['ticker']} | {row['model']} | {row['variant']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} |")
    lines += ["", "## Reproduce offline", "", "```sh",
              'PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \\',
              '  .venv/bin/python -m quantlab_ai.guidance_study', "```", "",
              "Requires the existing hashed SEC index, announcement snapshots under data/guidance_v11, "
              "v8 historical feature files and saved control predictions. These local datasets are "
              "excluded from Git; a fresh clone alone cannot reproduce training. The study writes its "
              "plan before fitting and saves fold audits, prediction files and isolated model artifacts. "
              "The website is unchanged. No profitability claim follows from these classification scores.", ""]
    (root / "docs/guidance_study.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(Settings())
