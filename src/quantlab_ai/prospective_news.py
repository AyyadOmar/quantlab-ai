"""After-close, expanding-window comparison on news observed before the signal close."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .data.loader import MarketDataLoader
from .data.news_sentiment import SCORER, sentiment_at, sentiment_window
from .data.press_news import DEFAULT_FOLDER, HEADLINES, headline_activity_at, healthy_batches

START = "2026-11-23"
MIN_TRAIN_SESSIONS = 20
PRICE = ["return_1d", "return_5d", "intraday", "rms_20"]
COUNTS = ["headline_urls_1h", "headline_urls_delta_1h", "eligible_urls",
          "article_tone_scored", "articles_with_scored_quotes"]
TEXT = ["headline_balance", "article_tone_mean", "quote_balance"]
ARMS = {"price": PRICE, "counts": PRICE + COUNTS, "wording": PRICE + COUNTS + TEXT}
MORNING = ["morning_" + k for k in ["eligible_urls", "article_tone_scored", "articles_with_scored_quotes", *TEXT]]
ARMS["morning_and_late"] = ARMS["wording"] + MORNING
DB = DEFAULT_FOLDER / "prospective_direction_v2.sqlite3"
NY = ZoneInfo("America/New_York")


def data_ready_after(cal, day):
    # MarketDataLoader regards the current day's Yahoo daily bar as final at 16:15 ET,
    # including early closes. News always uses the actual exchange close.
    vendor_ready = datetime.fromisoformat(str(pd.Timestamp(day).date()) + "T16:15:00").replace(tzinfo=NY)
    return max(vendor_ready, cal.session_close(day) + timedelta(minutes=15))


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS signals (day TEXT PRIMARY KEY, target_day TEXT UNIQUE, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS outcomes (day TEXT PRIMARY KEY, observed TEXT NOT NULL, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY, settings TEXT NOT NULL)")
    plan = json.dumps({"start": START, "arms": ARMS, "tickers": list(HEADLINES), "min_sessions": MIN_TRAIN_SESSIONS,
                       "model": "StandardScaler + LogisticRegression(liblinear,C=.1,seed=42)",
                       "threshold": .5, "target": "next trading session close > open", "scorer": SCORER,
                       "calendar": "XNYS", "required_late_batches": 8,
                       "morning": "session open to earlier of noon NY or two hours before close; full coverage"}, sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO protocol VALUES (1,?)", (plan,))
        if db.execute("SELECT settings FROM protocol WHERE id=1").fetchone()[0] != plan:
            raise ValueError("Protocol changed. Use a new versioned ledger for a new experiment.")
    return db


def resolved_rows(db, before=None):
    rows = []
    for day, signal, prices in db.execute(
        "SELECT s.day,s.payload,o.payload FROM signals s JOIN outcomes o ON s.target_day=o.day "
        "WHERE (? IS NULL OR o.day<=?) ORDER BY s.day", (before, before)
    ):
        record, settled = json.loads(signal), json.loads(prices)
        for ticker, features in record["features"].items():
            price = settled[ticker]
            rows.append({**features, "day": day, "ticker": ticker,
                         "up": int(price["close"] > price["open"]),
                         "predictions": record["predictions"].get(ticker)})
    return rows


def forecasts(history, today):
    train = pd.DataFrame(history)
    if train.empty or train.day.nunique() < MIN_TRAIN_SESSIONS or train.up.nunique() < 2:
        return {}
    current = pd.DataFrame.from_dict(today, orient="index")
    output = {t: {} for t in today}
    for name, cols in ARMS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="liblinear", random_state=42))
        model.fit(train[cols], train.up)
        for ticker, prob in zip(current.index, model.predict_proba(current[cols])[:, 1]):
            output[ticker][name] = float(prob)
    return output


def news_features(cutoff, folder):
    activity = headline_activity_at(cutoff, folder)
    coverage = {k: activity[k] for k in ("recent_batches", "previous_batches", "expected_batches_per_hour")}
    if min(coverage["recent_batches"], coverage["previous_batches"]) < coverage["expected_batches_per_hour"]:
        return {}, coverage
    scores = sentiment_at(cutoff, folder, activity)
    if any(s["headline_scored"] != s["eligible_urls"] for s in scores.values()):
        return {}, {**coverage, "incomplete_headline_scores": True}
    result = {}
    for ticker, s in scores.items():
        fields = {**activity["tickers"][ticker], **s}
        # The count paired with each balance distinguishes absent text from neutral wording.
        result[ticker] = {key: fields[key] for key in COUNTS}
        result[ticker].update({key: float(fields[key] or 0) for key in TEXT})
    # Freeze the morning summary at its own cutoff so afternoon edits cannot enter it.
    close = pd.Timestamp(cutoff).tz_convert(NY)
    opening = xc.get_calendar("XNYS").session_open(close.date().isoformat())
    morning_end = min(close.normalize() + pd.Timedelta(hours=12), close - pd.Timedelta(hours=2))
    slots = pd.date_range(opening, morning_end.tz_convert("UTC"), freq="15min", inclusive="left")
    if slots.empty:
        raise ValueError("Exchange session has no distinct morning window")
    available = healthy_batches(folder, morning_end)
    coverage["morning_batches"] = sum(stamp in available for stamp in slots)
    coverage["expected_morning_batches"] = len(slots)
    if coverage["morning_batches"] != len(slots):
        return {}, coverage
    morning = sentiment_window(folder, opening, morning_end, complete=True)
    for ticker, fields in morning.items():
        if fields["headline_scored"] != fields["eligible_urls"]:
            return {}, {**coverage, "incomplete_morning_scores": True}
        result[ticker].update({"morning_" + key: float(fields[key] or 0)
                               for key in ["eligible_urls", "article_tone_scored", "articles_with_scored_quotes", *TEXT]})
    return result, coverage


def price_frame(raw, ticker, expected):
    if set(raw.ticker) != {ticker}:
        raise ValueError(f"Unexpected ticker in {ticker} prices")
    data = raw.set_index("date").sort_index()
    if data.index.has_duplicates:
        raise ValueError(f"Duplicate price dates for {ticker}")
    bars = data.reindex(expected)[["open", "close"]].astype(float)
    if not np.isfinite(bars.to_numpy()).all() or (bars <= 0).any().any():
        raise ValueError(f"Missing or invalid session prices for {ticker}")
    return bars


def price_inputs(bars):
    close = bars.close
    intraday = close / bars.open - 1
    return {"return_1d": float(close.iloc[-1] / close.iloc[-2] - 1),
            "return_5d": float(close.iloc[-1] / close.iloc[-6] - 1),
            "intraday": float(intraday.iloc[-1]), "rms_20": float(np.sqrt(np.mean(intraday.tail(20)**2)))}


def capture(db, folder=DEFAULT_FOLDER, clock=lambda: datetime.now(timezone.utc), download=None):
    now = clock()
    day = now.astimezone(NY).date().isoformat()
    cal = xc.get_calendar("XNYS")
    if day < START or not cal.is_session(day) or now < data_ready_after(cal, day):
        return {"status": "outside_after_close_window"}
    cutoff = cal.session_close(day).isoformat()
    target = cal.next_session(day).date().isoformat()
    # Serialize retries. A failure in price collection rolls back both outcomes and signals.
    with db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT payload FROM signals WHERE day=?", (day,)).fetchone()
        if existing:
            return {"status": "already_recorded", "day": day}
        pending = [row[0] for row in db.execute(
            "SELECT target_day FROM signals WHERE target_day<=? AND target_day NOT IN (SELECT day FROM outcomes) "
            "AND json_extract(payload,'$.status') != 'missing_coverage' ORDER BY target_day", (day,))]
        news, coverage = news_features(cutoff, folder)
        bars = {}
        if news or pending:
            window = cal.sessions_window(pd.Timestamp(day), -21)[-20:]
            start = min([str(window[0].date()), *pending])
            expected = window.union(pd.DatetimeIndex(pending, dtype="datetime64[ns]")).sort_values()
            fetch = download or MarketDataLoader(Settings()).download
            for ticker in HEADLINES:
                raw = fetch(ticker, start, str(pd.Timestamp(day).date() + timedelta(days=1)))
                bars[ticker] = price_frame(raw, ticker, expected)
            for due in pending:
                prices = {t: frame.loc[due].to_dict() for t, frame in bars.items()}
                db.execute("INSERT INTO outcomes VALUES (?,?,?)", (due, clock().isoformat(), json.dumps(prices)))
        features = {t: {**price_inputs(bars[t].tail(20)), **fields} for t, fields in news.items()}
        train = resolved_rows(db, before=day)
        preds = forecasts(train, features) if features else {}
        status = "forecast" if preds else ("training" if features else "missing_coverage")
        created = clock()
        if created.astimezone(NY).date().isoformat() != day or created >= cal.session_open(target):
            raise ValueError("Forecast window expired before recording")
        record = {"day": day, "target_day": target, "cutoff": cutoff, "created": created.isoformat(),
                  "status": status, "coverage": coverage, "training_rows": len(train),
                  "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "sklearn": version("scikit-learn"),
                  "features": features, "predictions": preds}
        db.execute("INSERT INTO signals VALUES (?,?,?)", (day, target, json.dumps(record, allow_nan=False)))
        return {"status": status, "day": day, "target_day": target, "coverage": coverage,
                "training_rows": len(train), "outcome_sessions_added": len(pending)}


def report(db, now=None):
    now = now or datetime.now(timezone.utc)
    cal = xc.get_calendar("XNYS")
    last = now.astimezone(NY).date().isoformat()
    expected = [] if last < START else [str(d.date()) for d in cal.sessions_in_range(START, last)
                                        if data_ready_after(cal, d) <= now]
    logged = {day: json.loads(r)["status"] for day, r in db.execute("SELECT day,payload FROM signals ORDER BY day")}
    resolved = resolved_rows(db)
    scored = [row for row in resolved if row["predictions"]]
    metrics = {}
    if scored:
        actual = np.array([row["up"] for row in scored])
        for arm in [*ARMS, "always_up", "always_down"]:
            if arm in ARMS:
                prob = np.array([r["predictions"][arm] for r in scored])
            else:
                prob = np.full(len(actual), int(arm == "always_up"))
            metrics[arm] = {"correct": int(np.sum((prob >= .5) == actual)), "accuracy": float(np.mean((prob >= .5) == actual)),
                            "brier": float(np.mean((prob-actual)**2))}
    morning_comparison = None
    changed = None
    if scored:
        correct_with_text = np.array([r["predictions"]["wording"] >= .5 for r in scored]) == actual
        correct_with_counts = np.array([r["predictions"]["counts"] >= .5 for r in scored]) == actual
        correct_with_morning = np.array([r["predictions"]["morning_and_late"] >= .5 for r in scored]) == actual
        morning_comparison = {"calls_fixed_by_morning": int(np.sum(correct_with_morning & ~correct_with_text)),
                              "additional_errors_with_morning": int(np.sum(~correct_with_morning & correct_with_text))}
        changed = {"calls_fixed_by_wording": int(np.sum(correct_with_text & ~correct_with_counts)),
                   "additional_errors_with_wording": int(np.sum(~correct_with_text & correct_with_counts))}
    return {"first_eligible_session": START, "sessions_recorded": len(logged),
            "missing_runs": [d for d in expected if d not in logged],
            "incomplete_news_sessions": [d for d, status in logged.items() if status == "missing_coverage"],
            "training_sessions": sum(s == "training" for s in logged.values()),
            "forecast_sessions": sum(s == "forecast" for s in logged.values()),
            "resolved_complete_sessions": len({r["day"] for r in resolved}),
            "scored_forecasts_per_arm": len(scored), "wording_vs_counts": changed, "morning_vs_late": morning_comparison, "results": metrics}


if __name__ == "__main__":
    import argparse
    from contextlib import closing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Record today's after-close inputs and settle prior sessions")
    args = parser.parse_args()
    with closing(connect(DB)) as connection:
        print(json.dumps(capture(connection) if args.capture else report(connection), indent=2))
