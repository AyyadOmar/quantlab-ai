"""One frozen confirmation snapshot; optional append-only, genuinely future forecasts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .data.loader import MarketDataLoader
from .features.builder import FeatureBuilder
from .recent_history_study import PROTOCOL, MODEL_NAMES, predict_artifact, summarize
from .direction_study import apply_direction_rule


def stitch_frozen_history(frozen, fresh, cutoff):
    """Keep original development prices; reject revised price units across the join."""
    cutoff = pd.Timestamp(cutoff)
    old = frozen.loc[frozen.date <= cutoff].copy()
    if old.empty or old.date.max() != cutoff:
        raise ValueError("Frozen history does not reach the specified cutoff.")
    joined = old.merge(fresh, on="date", suffixes=("_old", "_new"))
    if joined.empty or joined.date.max() != cutoff:
        raise ValueError("Fresh source must overlap the frozen cutoff for a price-unit check.")
    for column in ["open", "high", "low", "close"]:
        if not np.allclose(joined[column + "_old"], joined[column + "_new"], rtol=1e-5, atol=1e-4):
            raise ValueError("Historical OHLC revisions detected. Review price units before confirming; do not mix snapshots.")
    new = fresh.loc[fresh.date > cutoff]
    if new.empty:
        raise ValueError("No confirmation sessions after the frozen cutoff.")
    return pd.concat([old, new], ignore_index=True).sort_values("date").reset_index(drop=True)


def confirmation_rows(features, cutoff):
    cutoff = pd.Timestamp(cutoff)
    # The signal at the cutoff's close may predict the first new session's outcome.
    result = features.loc[(features.date >= cutoff) & (features.execution_date > cutoff)].copy()
    if result.empty or result.date.duplicated().any():
        raise ValueError("Confirmation requires new, unique outcome sessions.")
    return result


def append_forecasts(path: Path, records: list[dict]):
    """Do not overwrite an existing prediction, even if a later run disagrees."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = {json.loads(line)["forecast_id"] for line in handle if line.strip()}
        written = 0
        for record in records:
            if record["forecast_id"] not in existing:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                existing.add(record["forecast_id"])
                written += 1
        handle.flush()
        return written


def prospective_date_is_safe(as_of, now=None):
    now = now or datetime.now(timezone.utc)
    local_day = pd.Timestamp(now.astimezone(ZoneInfo("America/New_York")).date())
    as_of = pd.Timestamp(as_of).normalize()
    # Conservative: don't backfill a prospective claim when a later weekday has begun.
    return as_of <= local_day and as_of + pd.offsets.BDay(1) > local_day


def run_confirmation(settings, end: str):
    settings = replace(settings, protocol=PROTOCOL)
    settings.ensure_directories()
    freeze_path = settings.backtests_dir / "confirmation_freeze.json"
    freeze_bytes = freeze_path.read_bytes()  # Required before any network access.
    freeze = json.loads(freeze_bytes)
    cutoff = pd.Timestamp(freeze["known_through"])
    if pd.Timestamp(end) <= cutoff:
        raise ValueError("Confirmation end must be after the frozen outcome cutoff.")
    for name, digest in freeze["source_hashes"].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Selection code changed after the freeze; preserve the original protocol.")
    loaded = {}
    for key, entry in freeze["artifacts"].items():
        loaded[key] = {}
        for window, artifact in entry["files"].items():
            path = settings.project_root / artifact["path"]
            if sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("Frozen model hash mismatch.")
            loaded[key][window] = joblib.load(path)
    data_dir = settings.data_dir / "recent_history_v8"
    snapshot_dir = data_dir / "confirmation_prices"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "manifest.json"
    existing = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    if existing and (existing["end_exclusive"] != end or existing["freeze_sha256"] != sha256(freeze_bytes).hexdigest()):
        raise ValueError("Confirmation snapshot is immutable. Reproduce with its original end date and freeze.")
    loader = MarketDataLoader(replace(settings, use_cached_data=False))
    snapshots, sources = {}, {}
    # All downloads complete before any model scores are computed. No partial-universe results.
    for ticker in [*freeze["companies"], "SPY"]:
        path = snapshot_dir / f"{ticker.lower()}.csv"
        if existing:
            if sha256(path.read_bytes()).hexdigest() != existing["sources"][ticker]["sha256"]:
                raise ValueError("Confirmation price hash mismatch.")
            fresh = pd.read_csv(path, parse_dates=["date"])
        else:
            fresh = loader.download(ticker, "2018-01-01", end)
            fresh.to_csv(path, index=False)
        sources[ticker] = {"sha256": sha256(path.read_bytes()).hexdigest(), "rows": len(fresh),
                           "start": str(fresh.date.min()), "end": str(fresh.date.max())}
        frozen_loader = MarketDataLoader(replace(settings, use_cached_data=True))
        old = frozen_loader.download(ticker, "2018-01-01", (cutoff + pd.Timedelta(days=1)).date().isoformat())
        expected_hash = (freeze["raw_hashes"][freeze["companies"][0]]["market_sha256"] if ticker == "SPY"
                         else freeze["raw_hashes"][ticker]["raw_sha256"])
        if sha256(old.to_csv(index=False).encode()).hexdigest() != expected_hash:
            raise ValueError("Development price cache changed after the freeze.")
        snapshots[ticker] = stitch_frozen_history(old, fresh, cutoff)
    if existing is None:
        existing = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "end_exclusive": end,
                    "freeze_sha256": sha256(freeze_bytes).hexdigest(), "sources": sources}
        with snapshot_path.open("x") as handle:
            json.dump(existing, handle, indent=2)
    feature_settings = replace(settings)
    feature_settings.processed_data_dir = settings.processed_data_dir / "confirmation"
    feature_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    builder = FeatureBuilder(feature_settings)
    features = {ticker: builder.build(snapshots[ticker], ticker, snapshots["SPY"])
                for ticker in freeze["companies"]}
    tests = {ticker: confirmation_rows(frame, cutoff) for ticker, frame in features.items()}
    anchor = next(iter(tests.values()))
    if any(not frame[["date", "execution_date"]].reset_index(drop=True).equals(
            anchor[["date", "execution_date"]].reset_index(drop=True)) for frame in tests.values()):
        raise ValueError("Confirmation calendars differ across companies.")
    results, frames = [], {(model, scope): [] for model in MODEL_NAMES for scope in ["full", "validation_selected"]}
    for key, entry in freeze["artifacts"].items():
        for scope in ["full", "validation_selected"]:
            window = "full" if scope == "full" else entry["selected_window"]
            pred = predict_artifact(loaded[key][window], tests[entry["ticker"]])
            pred.to_csv(settings.backtests_dir / f"confirmation_{key}_{scope}_predictions.csv", index=False)
            frames[entry["model"], scope].append(pred)
            results.append({"ticker": entry["ticker"], "model": entry["model"], "scope": scope,
                            "selected_window": window, **summarize(pred, settings.random_state)})
    aggregates = {}
    for (model, scope), collected in frames.items():
        row = summarize(pd.concat(collected, ignore_index=True), settings.random_state)
        row["uncertainty"]["interpretation"] = "Descriptive frozen-model retrospective confirmation; short period, correlated companies, multiple reported families. Not a prospective trading record."
        aggregates[f"{model}_{scope}"] = {"model": model, "scope": scope, **row}
    # Only the last still-unresolved session may enter the prospective ledger.
    records = []
    as_of = snapshots["SPY"].date.max()
    if all(frame.date.max() == as_of for frame in snapshots.values()) and prospective_date_is_safe(as_of):
        for key, entry in freeze["artifacts"].items():
            artifact = loaded[key][entry["selected_window"]]
            latest = builder.build_for_inference(snapshots[entry["ticker"]], snapshots["SPY"]).iloc[[-1]]
            if latest.date.iloc[0] != as_of:
                raise ValueError("Latest feature row is stale; no prospective record may be backfilled.")
            probability = float(artifact["model"].predict_proba(latest[artifact["feature_columns"]])[:, 1][0])
            prediction = int(apply_direction_rule([probability], artifact["direction_rule"])[0])
            forecast_id = f"{key}:{as_of.date().isoformat()}:{existing['freeze_sha256']}"
            records.append({"forecast_id": forecast_id, "created_at": datetime.now(timezone.utc).isoformat(),
                            "as_of_session": as_of.date().isoformat(), "target": "next actual trading session open-to-close",
                            "ticker": entry["ticker"], "model": entry["model"], "window": entry["selected_window"],
                            "prob_up": probability, "prediction": prediction, "status": "unresolved",
                            "freeze_sha256": existing["freeze_sha256"],
                            "price_snapshot_sha256": sources[entry["ticker"]]["sha256"]})
    written = append_forecasts(data_dir / "prospective_predictions.jsonl", records) if records else 0
    report = {"freeze_sha256": existing["freeze_sha256"], "snapshot": existing, "results": results,
              "aggregates": aggregates, "prospective_records_available": len(records), "new_prospective_records": written}
    (settings.backtests_dir / "confirmation.json").write_text(json.dumps(report, indent=2))
    write_report(settings, report)
    return report


def write_report(settings, report):
    lines = ["# Frozen-model confirmation on newer dates", "",
             "The window and cutoff choices were saved before this price snapshot was downloaded. These outcomes were not used to refit models or choose a winning family. This is retrospective confirmation on newer dates, not a record of forecasts made before those historical events.", "",
             "| Model | Frozen policy | Accuracy | Always-up | Difference (pp) | Correct / incorrect down | Brier | 95% difference interval (pp) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"].values():
        low, high = row["uncertainty"]["interval_95"]
        lines.append(f"| {row['model']} | {row['scope']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {100*row['accuracy_gain']:+.2f} | {row['correct_down']} / {row['incorrect_down']} | {row['brier_score']:.4f} | [{100*low:+.2f}, {100*high:+.2f}] |")
    first = report["results"][0]
    lines += ["", "## Selected policy by company", "",
              "| Company | Model | Frozen window | Accuracy | Always-up | Correct / incorrect down |",
              "|---|---|---|---:|---:|---:|"]
    for row in report["results"]:
        if row["scope"] == "validation_selected":
            lines.append(f"| {row['ticker']} | {row['model']} | {row['selected_window']} | {row['accuracy']:.2%} | {row['always_up_accuracy']:.2%} | {row['correct_down']} / {row['incorrect_down']} |")
    lines += ["", "Inspect the company-level counts before generalizing an aggregate improvement. Matching always-up without any down calls is not evidence of learned direction skill."]
    lines += ["", f"{first['rows']} sessions per company, from {first['execution_start']} through {first['execution_end']}. Apple, Microsoft and NVIDIA share identical dates. Models remain frozen throughout.", "",
              "The comparison uses all available sessions in the declared snapshot, not a period picked because it scores well. Intervals use 20-session blocks, keeping all companies on each date together. A short period may have very few effective independent blocks and cannot establish a reliable edge. Models and prices can still be affected by historical selection or data revisions. No trading returns are claimed.", "",
              f"Prospective records eligible at this run: {report['prospective_records_available']}; newly appended: {report['new_prospective_records']}. The local ledger only records the latest session if a later weekday has not begun, and never replaces a prior prediction. Its target is the next actual market session, with outcomes initially unresolved.", "",
              "No scheduled collection is configured. Repeating this confirmation command reproduces its immutable snapshot; it does not silently refresh or retune models. Continued prospective collection and later outcome resolution are separate work. Website and default models remain unchanged."]
    (settings.backtests_dir / "confirmation.md").write_text("\n".join(lines) + "\n")
