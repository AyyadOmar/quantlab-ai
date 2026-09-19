"""Ablations for market context and historically available earnings filings."""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtesting.engine import BacktestEngine
from .config import Settings
from .data.context import COMPANY_CIKS, MARKET_SYMBOLS
from .data.loader import MarketDataLoader
from .feature_study import CANDIDATES, Candidate, choose_candidate, evaluate_fold, paired_brier_interval
from .features.builder import FeatureBuilder
from .features.context import add_market_context, add_earnings_context
from .features.profiles import feature_columns
from .models.base import aggregate_classification_metrics, classification_baselines, combine_fold_predictions
from .models.classical import ClassicalModelTrainer

CONTEXT_CANDIDATES = CANDIDATES + tuple(
    Candidate(f"{base}_{context}", "regularized")
    for base in ["relative", "compact"] for context in ["market", "earnings", "context"])
BASE_NAMES = {candidate.name for candidate in CANDIDATES}
SCOPES = {
    "price_only": BASE_NAMES,
    "with_market": BASE_NAMES | {f"{base}_market_regularized" for base in ["relative", "compact"]},
    "with_earnings": BASE_NAMES | {f"{base}_earnings_regularized" for base in ["relative", "compact"]},
    "with_both": {candidate.name for candidate in CONTEXT_CANDIDATES},
}


def candidates_for_ticker(ticker: str) -> tuple[Candidate, ...]:
    if ticker in COMPANY_CIKS:
        return CONTEXT_CANDIDATES
    if ticker in {"SPY", "QQQ"}:
        # Even constant extra columns can change tree column-sampling behavior.
        # Do not treat that randomness as an earnings effect for an ETF.
        return tuple(c for c in CONTEXT_CANDIDATES
                     if not c.feature_set.endswith(("_earnings", "_context")))
    raise ValueError(f"No declared earnings applicability for {ticker}.")


def load_context(root: Path) -> tuple[dict, dict, dict]:
    folder = root / "data" / "context_v4"
    manifest = json.loads((folder / "manifest.json").read_text())
    for source in manifest["sources"]:
        path = folder / source["file"]
        if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Cached context differs from the source manifest: {path.name}")
    histories = {name: pd.read_csv(folder / f"{name}.csv", parse_dates=["date"]) for name in MARKET_SYMBOLS}
    earnings = {ticker: pd.read_csv(folder / f"{ticker.lower()}_earnings_filings.csv") for ticker in COMPANY_CIKS}
    return histories, earnings, manifest


def run_context_study(settings: Settings, tickers: list[str], start: str, end: str) -> dict:
    settings = replace(settings, protocol="context_study_v4", use_cached_data=True)
    settings.ensure_directories()
    histories, earnings, manifest = load_context(settings.project_root)
    loader, builder, engine = MarketDataLoader(settings), FeatureBuilder(settings), BacktestEngine(settings)
    market = loader.download("SPY", start, end)
    plan = {"protocol": settings.protocol, "execution_protocol": "open_to_close_v2", "start": start, "end_exclusive": end,
            "tickers": tickers, "candidates": [asdict(c) for c in CONTEXT_CANDIDATES],
            "feature_columns": {c.feature_set: feature_columns(c.feature_set) for c in CONTEXT_CANDIDATES},
            "scopes": {name: sorted(candidates) for name, candidates in SCOPES.items()},
            "selection": "Minimize validation Brier then log loss; each scope can keep a price-only model.",
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
            "sources": manifest,
            "earnings_limit": "Past SEC 8-K item 2.02 filings only. Not future earnings dates, actual EPS or analyst surprises.",
            "timing": "Market observations strictly before the signal date, maximum age 7 calendar days. Earnings usable from calendar day after filing.",
            "default_models_changed": False, "historical_research_only": True}
    (settings.backtests_dir / "study_plan.json").write_text(json.dumps(plan, indent=2))
    rows, selections, coverage, data_hashes, diagnostics = [], {}, {}, {}, []
    differences = {(model, scope): [] for model in ["logistic_regression", "xgboost"] for scope in SCOPES if scope != "price_only"}
    for ticker in tickers:
        candidates = candidates_for_ticker(ticker)
        raw = loader.download(ticker, start, end)
        features = builder.build(raw, ticker, None if ticker == "SPY" else market)
        original_dates = features.date.copy()
        features = add_market_context(features, histories)
        features = add_earnings_context(features, earnings.get(ticker))
        if not original_dates.reset_index(drop=True).equals(features.date.reset_index(drop=True)):
            raise ValueError("Context joins changed evaluation dates.")
        columns = sorted({column for candidate in CONTEXT_CANDIDATES for column in feature_columns(candidate.feature_set)})
        if not np.isfinite(features[columns].to_numpy()).all():
            raise ValueError("Context feature values must be finite on the entire baseline sample.")
        features.to_csv(settings.processed_data_dir / f"{ticker.lower()}_features.csv", index=False)
        data_hashes[ticker] = sha256(raw.to_csv(index=False).encode()).hexdigest()
        coverage[ticker] = {"feature_rows": len(features), "earnings_applicable": ticker in COMPANY_CIKS,
                            "earnings_known_fraction": float(features.earnings_history_known.mean()),
                            "recent_earnings_rows": int(features.earnings_recent_5d.sum()),
                            "max_market_age_days": max(int((features.date-features[f"{name}_source_date"]).dt.days.max()) for name in MARKET_SYMBOLS)}
        for model_name in ["logistic_regression", "xgboost"]:
            splitter = ClassicalModelTrainer(settings, model_name)
            grouped = {name: [] for name in SCOPES}
            static = {c.name: [] for c in candidates if c.feature_set.startswith("compact") and c.model_profile == "regularized"}
            folds = []
            for fold, (train, validation, test) in enumerate(splitter.walk_forward_splits(features), 1):
                predictions, report, artifact = evaluate_fold(settings, features, train, validation, test, model_name, candidates)
                report["fold"] = fold
                report["scope_selections"] = {}
                for scope, names in SCOPES.items():
                    chosen = choose_candidate([score for score in report["validation_scores"] if score["candidate"] in names])
                    report["scope_selections"][scope] = chosen
                    pred = predictions[chosen].copy()
                    pred["fold"] = fold
                    grouped[scope].append(pred)
                for name in static:
                    pred = predictions[name].copy()
                    pred["fold"] = fold
                    static[name].append(pred)
                folds.append(report)
            path = splitter.registry.save_joblib(f"{model_name}_{ticker.lower()}_context_selected", artifact)
            selections[f"{ticker}_{model_name}"] = {"folds": folds, "selected_artifact": path}
            combined = {scope: combine_fold_predictions(frames) for scope, frames in grouped.items()}
            for scope, pred in combined.items():
                metrics = aggregate_classification_metrics(pred)
                baseline = classification_baselines(pred)
                backtest = engine.run_with_threshold(pred, model_name, ticker, None, False)
                row = {"ticker": ticker, "model": model_name, "scope": scope, **metrics,
                       "always_up_accuracy": baseline["always_up"]["accuracy"],
                       "prior_brier": baseline["training_prevalence"]["brier_score"],
                       "net_return": backtest.metrics["total_return"], "trades": backtest.metrics["trade_count"],
                       "max_drawdown": backtest.metrics["max_drawdown"], "sharpe_ratio": backtest.metrics["sharpe_ratio"],
                       "execution_start": str(pred.execution_date.iloc[0]), "execution_end": str(pred.execution_date.iloc[-1])}
                rows.append(row)
                pred.to_csv(settings.backtests_dir / f"{ticker.lower()}_{model_name}_{scope}_predictions.csv", index=False)
                backtest.equity_curve.to_csv(settings.backtests_dir / f"{ticker.lower()}_{model_name}_{scope}_equity.csv", index=False)
                if scope != "price_only":
                    base = combined["price_only"]
                    paired = pred[["execution_date"]].copy()
                    paired["loss_difference"] = (pred.prob_up-pred.target)**2 - (base.prob_up-base.target)**2
                    differences[(model_name, scope)].append(paired)
            for name, frames in static.items():
                pred = combine_fold_predictions(frames)
                diagnostics.append({"ticker": ticker, "model": model_name, "candidate": name,
                                    **aggregate_classification_metrics(pred)})
            print(f"Completed context study: {ticker} / {model_name}", flush=True)
    intervals = {f"{model}/{scope}": paired_brier_interval(frames, settings.random_state)
                 for (model, scope), frames in differences.items()}
    report = {"plan": plan, "raw_data_hashes": data_hashes, "coverage": coverage, "results": rows,
              "selection": selections, "paired_uncertainty": intervals, "fixed_compact_diagnostics": diagnostics}
    (settings.backtests_dir / "context_study.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(settings.backtests_dir / "context_study.csv", index=False)
    write_context_report(settings, report)
    return report


def write_context_report(settings: Settings, report: dict) -> None:
    data = pd.DataFrame(report["results"])
    lines = ["# Market and earnings-context study", "",
             "The added inputs are technology-sector performance (XLK), market volatility (VIX), long/short yield proxies (TNX/IRX), and recency of earnings-related SEC filings for AAPL, MSFT and NVDA. ETFs have no company earnings event features.", "",
             "Market data uses only previous available sessions. Earnings features become usable the calendar day after filing. Filing dates can lag actual earnings announcements; this conservative first version contains no upcoming earnings schedule, actual EPS, or analyst surprise data.", "",
             "Each comparison selects its model and trading threshold using earlier validation only, on exactly the same dates and costs as the preceding feature study. Every context selection can retain a price-only candidate. These remain retrospective experiments on previously examined dates; defaults and public demo results are unchanged.", "",
             "## Mean results across five tickers", "",
             "| Model | Available inputs | Accuracy | AUC | Brier ↓ | Log loss ↓ | Mean ticker net return |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (model, scope), group in data.groupby(["model", "scope"], sort=False):
        lines.append(f"| {model} | {scope} | {group.accuracy.mean():.2%} | {group.roc_auc.mean():.3f} | {group.brier_score.mean():.4f} | {group.log_loss.mean():.4f} | {group.net_return.mean():.2%} |")
    lines += ["", "Mean ticker returns are not a simulated portfolio. Both-side fees and slippage total approximately 14 bps per round trip. The earnings comparisons contain three companies and two ETFs; company-only results follow.", "",
              "## Earnings context: companies only", "", "| Model | Available inputs | Accuracy | Brier ↓ |", "|---|---|---:|---:|"]
    for (model, scope), group in data.loc[data.ticker.isin(COMPANY_CIKS)].groupby(["model", "scope"], sort=False):
        lines.append(f"| {model} | {scope} | {group.accuracy.mean():.2%} | {group.brier_score.mean():.4f} |")
    lines += ["", "## Combined context versus price-only selection", "",
              "| Ticker | Model | Price accuracy | Context accuracy | Always up | Price Brier | Context Brier | Prior Brier | Context net return | Trades |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (ticker, model), group in data.groupby(["ticker", "model"], sort=False):
        base = group.loc[group.scope.eq("price_only")].iloc[0]
        full = group.loc[group.scope.eq("with_both")].iloc[0]
        lines.append(f"| {ticker} | {model} | {base.accuracy:.2%} | {full.accuracy:.2%} | {full.always_up_accuracy:.2%} | {base.brier_score:.4f} | {full.brier_score:.4f} | {full.prior_brier:.4f} | {full.net_return:.2%} | {full.trades} |")
    lines += ["", "## Paired probability-error uncertainty", "", "Context minus price-only Brier; negative is better. Descriptive 95% intervals from 20-session block resampling after averaging same-date differences across tickers. These are not corrected for repeated research or multiple comparisons.", ""]
    for name, interval in report["paired_uncertainty"].items():
        low, high = interval["block_bootstrap_95_percent_interval"]
        lines.append(f"- {name}: {interval['selected_minus_original_brier']:.4f}, interval [{low:.4f}, {high:.4f}].")
    lines += ["", "## Coverage and reproducibility", ""]
    for ticker, coverage in report["coverage"].items():
        lines.append(f"- {ticker}: {coverage['feature_rows']} feature rows; earnings applicable: {coverage['earnings_applicable']}; past filing known on {coverage['earnings_known_fraction']:.1%} of rows; maximum market age {coverage['max_market_age_days']} calendar days.")
    first = data.iloc[0]
    lines += ["", f"Test executions: {first.execution_start} through {first.execution_end}; {first.evaluation_rows} observations per experiment.", "",
              "The JSON report records validation selections, fixed compact-feature diagnostics, data hashes and cost assumptions. Context files have an integrity-checked manifest. Selected model artifacts store the actual feature list and threshold. Observations with missing or stale market context raise an error rather than silently changing comparison dates.", "",
              "Public sources: [SEC submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [Form 8-K item 2.02](https://www.sec.gov/files/form8-k.pdf), and Yahoo Finance through yfinance. Yield curve is a quoted-index difference, not an official constant-maturity spread. Current historical market downloads can contain corrections; these are not archived point-in-time market vintages."]
    (settings.backtests_dir / "context_study.md").write_text("\n".join(lines)+"\n")
