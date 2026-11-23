"""Post-hoc audit: preserve feature layout while removing guidance information."""
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.data.guidance import EVENT_COLUMNS, build_event_features, load_documents
from quantlab_ai.guidance_study import evaluate_fold
from quantlab_ai.intraday_study import HISTORICAL_END, TICKERS, metrics, paired_gain, verify_control
from quantlab_ai.models.base import combine_fold_predictions
from quantlab_ai.models.classical import ClassicalModelTrainer


def remove_content(history):
    """Keep all rows, price inputs, timing and column order; zero only content."""
    result = history.copy()
    for column in EVENT_COLUMNS:
        if column != "earnings_release_recent":
            result[column] = 0.0
    return result


def run(settings):
    settings = replace(settings, protocol="guidance_placebo_v11", use_cached_data=True)
    settings.ensure_directories()
    root = settings.project_root
    documents, _ = load_documents(root)
    original = root / "backtesting/guidance_study_v11"
    histories, inputs = {}, {}
    for ticker in TICKERS:
        path = root / "data/processed/recent_history_study_v8" / f"{ticker}_features.csv"
        history = pd.read_csv(path, parse_dates=["date", "execution_date"])
        if history.execution_date.max() > HISTORICAL_END:
            raise ValueError("Reserved newer outcomes cannot enter diagnostic")
        histories[ticker] = build_event_features(history, documents)
        inputs[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
    for path in [root / "data/guidance_v11/source_manifest.json", original / "guidance_study.json"]:
        inputs[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(), "post_hoc": True,
        "reason": "MSFT has no extracted content, but adding constant columns changed XGBoost predictions.",
        "design": "One zero-content placebo per company, identical 21 columns/order, timing retained; unchanged XGBoost settings and validation-only rule.",
        "interpretation": "Diagnostic after inspecting v11 results; not an independently preregistered efficacy test or model-selection candidate.",
        "outcomes_through": str(HISTORICAL_END.date()), "inputs": inputs,
        "source_hashes": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), root / "src/quantlab_ai/guidance_study.py",
                                       root / "src/quantlab_ai/data/guidance.py"]},
    }
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frames = {variant: [] for variant in ["price_control", "release_timing", "guidance_content", "zero_content_placebo"]}
    audits, rows = {}, []
    for ticker, history in histories.items():
        empty = remove_content(history)
        trainer = ClassicalModelTrainer(settings, "xgboost", "compact", "regularized")
        predictions, folds = [], []
        for fold, (train, validation, test) in enumerate(trainer.walk_forward_splits(empty), 1):
            frame, audit, _ = evaluate_fold(settings, train, validation, test, "xgboost", "guidance_content")
            frame["fold"], audit["fold"] = fold, fold
            predictions.append(frame)
            folds.append(audit)
        placebo = combine_fold_predictions(predictions)
        # MSFT's actual content is entirely zero, so native-precision outputs must reproduce.
        if ticker == "MSFT":
            content = [c for c in EVENT_COLUMNS if c != "earnings_release_recent"]
            if np.any(history[content].to_numpy() != 0):
                raise ValueError("MSFT zero-content premise no longer holds")
            verify_control(placebo, pd.read_csv(original / "msft_xgboost_guidance_content_predictions.csv"))
        placebo["variant"] = "zero_content_placebo"
        placebo.to_csv(settings.backtests_dir / f"{ticker.lower()}_placebo_predictions.csv", index=False)
        audits[ticker] = folds
        for variant in frames:
            frame = placebo if variant == "zero_content_placebo" else pd.read_csv(
                original / f"{ticker.lower()}_xgboost_{variant}_predictions.csv", parse_dates=["date", "execution_date"])
            frames[variant].append(frame)
            rows.append({"ticker": ticker, "variant": variant, **metrics(frame)})
        print(f"Completed zero-content diagnostic: {ticker}", flush=True)
    combined = {variant: pd.concat(parts, ignore_index=True) for variant, parts in frames.items()}
    direction_changes = int((combined["guidance_content"].prediction.to_numpy()
                             != combined["zero_content_placebo"].prediction.to_numpy()).sum())
    report = {"plan": plan, "results": rows, "folds": audits, "msft_content_reproduced": True,
              "changed_direction_calls": direction_changes,
              "aggregates": [{"variant": variant, **metrics(frame)} for variant, frame in combined.items()],
              "content_vs_placebo": paired_gain(combined["guidance_content"], combined["zero_content_placebo"])}
    (settings.backtests_dir / "placebo_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Guidance feature-layout diagnostic", "",
             "Post-hoc follow-up to the [guidance study](guidance_study.md). Microsoft had no extracted "
             "guidance content, yet adding six all-zero columns changed its XGBoost predictions. "
             "XGBoost samples 80% of columns per tree: column count and layout can change the sampled "
             "price inputs even when the added columns contain no information.", "",
             "This diagnostic preserves the 21-column content-model layout and release timing, "
             "but zeros all six guidance-content columns. It uses the same historical dates, settings "
             "and validation-only selection. The plan was recorded before fitting this diagnostic, "
             "after inspecting the original study. It is not a new model candidate or fresh confirmation.", "",
             "| XGBoost inputs | Accuracy | Brier |", "|---|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['variant']} | {row['accuracy']:.2%} | {row['brier_score']:.4f} |")
    comparison = report["content_vs_placebo"]
    low, high = comparison["interval_95"]
    lines += ["", f"Same-date always-up: {report['aggregates'][0]['always_up_accuracy']:.2%}. "
              f"Real content minus zero-content placebo: {100*comparison['gain']:+.2f} percentage points; "
              f"descriptive 95% block-bootstrap interval [{100*low:+.2f}, {100*high:+.2f}]. "
              "Intervals do not adjust for repeated research.", "",
              f"The two variants differ on **{direction_changes} of 2,508 direction calls**. "
              "A zero-width accuracy interval here reflects identical historical decisions, "
              "not certainty about future performance. Some NVIDIA probabilities changed without "
              "changing a direction call.", "",
              "Microsoft's zero-content placebo reproduces all actual-content predictions and "
              "native-precision probabilities. Its score change therefore cannot be attributed "
              "to guidance information. The original aggregate lead is not evidence that news helped.", "",
              "| Company | Inputs | Accuracy |", "|---|---|---:|"]
    for row in rows:
        lines.append(f"| {row['ticker']} | {row['variant']} | {row['accuracy']:.2%} |")
    lines += ["", "## Reproduce", "", "```sh",
              'PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \\',
              '  .venv/bin/python scripts/check_guidance_placebo.py', "```", "",
              "Requires the local ignored guidance dataset, v8 features and completed v11 study. "
              "No newer outcomes, website updates or default-model changes are included. "
              "Future feature studies should retain this placebo control or disable column sampling "
              "consistently across every compared variant before evaluating fresh dates.", ""]
    (root / "docs/guidance_placebo.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    run(Settings())
