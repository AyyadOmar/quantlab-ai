"""Compare filing activity and a stock-specific volatility forecast on matched sessions."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd

from .config import Settings
from .data.filing_news import FILING_NEWS_COLUMNS, add_filing_news_features, load_filing_events
from .direction_study import CUTOFFS, MIN_VALIDATION_DOWN_CALLS, MODEL_NAMES
from .direction_study import apply_direction_rule, select_direction_rule
from .features.profiles import feature_columns
from .features.volatility_forecast import (
    BASELINE_COLUMN, VOLATILITY_COLUMNS, add_volatility_features, score_volatility, score_volatility_gain,
)
from .intraday_study import HISTORICAL_END, TICKERS, metrics, paired_gain, verify_control
from .models.base import combine_fold_predictions
from .models.classical import ClassicalModelTrainer

PROTOCOL = "news_volatility_v12"
PRICE = feature_columns("compact")
EXTRAS = [*FILING_NEWS_COLUMNS, *VOLATILITY_COLUMNS]
# XGBoost samples columns. Keep even the inactive columns in each ablation arm.
ACTIVE = {"native_price": [], "matched_price": [], "filings": FILING_NEWS_COLUMNS,
          "volatility": VOLATILITY_COLUMNS, "combined": EXTRAS}


def inputs_for_variant(history, variant):
    active = ACTIVE[variant]
    if variant == "native_price":
        return history, PRICE
    masked = {col: 0.0 for col in EXTRAS if col not in active}
    return history.assign(**masked), PRICE + EXTRAS


def evaluate(settings, model_name, history, variant):
    data, columns = inputs_for_variant(history, variant)
    trainer = ClassicalModelTrainer(settings, model_name, "compact", "regularized")
    trainer.feature_columns = columns
    folds, predictions = [], []
    for fold, (train, validation, test) in enumerate(trainer.walk_forward_splits(data), 1):
        fitted = trainer.fit_frame(train)
        validation_prob = trainer.predict_frame(fitted, validation, validation)
        rule = select_direction_rule(validation.target, validation_prob)["selected"]
        prob = trainer.predict_frame(fitted, test, test)
        frame = test[["ticker", "date", "execution_date", "target"]].copy()
        predictions.append(frame.assign(fold=fold, prob_up=prob, prediction=apply_direction_rule(prob, rule),
                                        fixed_prediction=(prob >= .5).astype(int),
                                        training_prevalence=float(train.target.mean())))
        folds.append({"fold": fold, "rule": rule, "train_last_outcome": str(train.execution_date.max()),
                      "validation_start": str(validation.date.min()), "validation_last_outcome": str(validation.execution_date.max()),
                      "test_start": str(test.date.min()), "test_last_outcome": str(test.execution_date.max())})
    return combine_fold_predictions(predictions), folds


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run(settings):
    settings = replace(settings, protocol=PROTOCOL, use_cached_data=True)
    root = settings.project_root
    output = settings.backtests_dir
    output.mkdir(parents=True, exist_ok=True)
    histories, inputs, provenance = {}, {}, {}
    for ticker in TICKERS:
        features = root / "data/processed/recent_history_study_v8" / f"{ticker.lower()}_features.csv"
        raw = root / "data/raw" / f"{ticker.lower()}_raw.csv"
        history = pd.read_csv(features, parse_dates=["date", "execution_date"])
        if history.execution_date.max() > HISTORICAL_END or set(history.ticker) != {ticker}:
            raise ValueError(f"Unexpected development history for {ticker}")
        for path in [features, raw]:
            inputs[str(path.relative_to(root))] = digest(path)
        events, provenance[ticker] = load_filing_events(root, ticker)
        enriched = add_volatility_features(history, pd.read_csv(raw))
        histories[ticker] = add_filing_news_features(enriched, events)
        pd.testing.assert_frame_equal(histories[ticker][history.columns], history)

    plan = {"protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(), "tickers": TICKERS,
            "models": MODEL_NAMES, "historical_outcomes_through": str(HISTORICAL_END.date()),
            "primary_comparisons": ["combined vs matched_price", "combined vs always_up"],
            "active_extras": ACTIVE, "compact_columns": PRICE, "matched_columns": PRICE + EXTRAS,
            "selection": "Full expanding history, fixed regularization, validation-only cutoff; score all test sessions.",
            "cutoffs": CUTOFFS, "minimum_validation_down_calls": MIN_VALIDATION_DOWN_CALLS,
            "settings": asdict(settings), "inputs": inputs, "sec_provenance": provenance,
            "code_hashes": {str(path.relative_to(root)): digest(path) for path in [
                Path(__file__), Path(__file__).parent / "data/filing_news.py",
                Path(__file__).parent / "features/volatility_forecast.py"]},
            "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost"]}}
    plan["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in plan["settings"].items()}
    save_json(output / "study_plan.json", plan)

    scores, comparisons, audits, risk_frames = [], {}, {}, []
    for model_name in MODEL_NAMES:
        by_variant = {variant: [] for variant in ACTIVE}
        for ticker, history in histories.items():
            for variant in ACTIVE:
                frame, folds = evaluate(settings, model_name, history, variant)
                key = f"{ticker.lower()}_{model_name}_{variant}"
                if variant == "native_price":
                    ref = root / "backtesting/recent_history_study_v8" / f"{ticker.lower()}_{model_name}_full_predictions.csv"
                    verify_control(frame, pd.read_csv(ref))
                    if model_name == MODEL_NAMES[0]:
                        risk = frame[["date", "ticker", "execution_date"]].merge(
                            history, on=["date", "ticker", "execution_date"], validate="one_to_one")
                        if len(risk) != len(frame):
                            raise ValueError("Volatility scores must cover the direction test rows")
                        risk_frames.append(risk)
                frame.to_csv(output / f"{key}.csv", index=False)
                audits[key] = folds
                scores.append({"ticker": ticker, "model": model_name, "inputs": variant, **metrics(frame)})
                by_variant[variant].append(frame)
            print(f"Scored {ticker} / {model_name}", flush=True)
        combined = {name: pd.concat(parts, ignore_index=True) for name, parts in by_variant.items()}
        for variant, frame in combined.items():
            scores.append({"ticker": "All", "model": model_name, "inputs": variant, **metrics(frame)})
        candidate = combined["combined"]
        comparisons[model_name] = {k: paired_gain(candidate, frame, settings.random_state)
                                   for k, frame in {"matched_price": combined["matched_price"],
                                                    "always_up": candidate.assign(prediction=1)}.items()}

    risk = pd.concat(risk_frames, ignore_index=True)
    risk[["ticker", "date", "execution_date", *VOLATILITY_COLUMNS,
          BASELINE_COLUMN, "next_day_return"]].to_csv(output / "volatility_predictions.csv", index=False)
    volatility = {ticker: score_volatility(part) for ticker, part in risk.groupby("ticker")}
    volatility["All"] = score_volatility(risk)
    report = {"date_range": [str(risk.execution_date.min().date()), str(risk.execution_date.max().date())],
              "direction_scores": scores, "combined_paired_accuracy": comparisons,
              "volatility_scores": volatility, "volatility_gain": score_volatility_gain(risk, settings.random_state),
              "folds": audits, "native_controls_reproduced": len(TICKERS)*len(MODEL_NAMES)}
    save_json(output / "results.json", report)
    return report


if __name__ == "__main__":
    run(Settings())
