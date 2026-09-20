from __future__ import annotations

import argparse
import json
from datetime import date

from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantLab AI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the end-to-end ML pipeline")
    run_parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    run_parser.add_argument("--start", required=True, help="Historical start date YYYY-MM-DD")
    run_parser.add_argument("--end", required=True, help="Historical end date YYYY-MM-DD")
    run_parser.add_argument(
        "--model",
        required=True,
        choices=["logistic_regression", "random_forest", "xgboost", "lstm"],
        help="Model to train",
    )

    batch_parser = subparsers.add_parser("run-batch", help="Run the same experiment across multiple tickers")
    batch_parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Space-separated ticker symbols, e.g. AAPL MSFT NVDA SPY",
    )
    batch_parser.add_argument("--start", required=True, help="Historical start date YYYY-MM-DD")
    batch_parser.add_argument("--end", required=True, help="Historical end date YYYY-MM-DD")
    batch_parser.add_argument(
        "--model",
        required=True,
        choices=["logistic_regression", "random_forest", "xgboost", "lstm"],
        help="Model to train for every ticker",
    )

    predict_parser = subparsers.add_parser("predict-latest", help="Generate and store the latest live prediction")
    predict_parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    predict_parser.add_argument("--start", default="2018-01-01", help="Historical training start date YYYY-MM-DD")
    predict_parser.add_argument("--end", default=date.today().isoformat(), help="End date for latest available data")
    predict_parser.add_argument(
        "--model",
        required=True,
        choices=["logistic_regression", "random_forest", "xgboost", "lstm", "all"],
        help="Model to use, or 'all' to generate predictions from every model",
    )

    resolve_parser = subparsers.add_parser("resolve-live", help="Resolve pending live predictions against actual next-day outcomes")
    resolve_parser.add_argument("--ticker", help="Optional ticker filter")

    report_parser = subparsers.add_parser("report-live", help="Show live prediction accuracy summary")
    report_parser.add_argument("--ticker", help="Optional ticker filter")

    baseline_parser = subparsers.add_parser("baseline", help="Compare logistic regression and XGBoost with simple baselines")
    baseline_parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    baseline_parser.add_argument("--start", default="2018-01-01")
    baseline_parser.add_argument("--end", required=True)
    study_parser = subparsers.add_parser("feature-study", help="Test relative features and regularization with validation-only selection")
    study_parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    study_parser.add_argument("--start", default="2018-01-01")
    study_parser.add_argument("--end", required=True)
    direction_parser = subparsers.add_parser("direction-study", help="Test validation-selected direction cutoffs against always-up using cached prices")
    direction_parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    direction_parser.add_argument("--start", default="2018-01-01")
    direction_parser.add_argument("--end", required=True)
    pooled_parser = subparsers.add_parser("pooled-direction-study", help="Compare separate and shared-company models using matched cached dates")
    pooled_parser.add_argument("--start", default="2018-01-01")
    pooled_parser.add_argument("--end", required=True)
    recent_parser = subparsers.add_parser("recent-history-study", help="Compare full, four-year and two-year training, then freeze confirmation models")
    recent_parser.add_argument("--start", default="2018-01-01")
    recent_parser.add_argument("--end", default="2026-05-25")
    recent_parser.add_argument("--historical-only", action="store_true", help="Reproduce development results without changing the confirmation freeze")
    confirmation_parser = subparsers.add_parser("recent-history-confirm", help="Evaluate frozen recent-history models on a separate public-price snapshot")
    confirmation_parser.add_argument("--end", required=True, help="Exclusive end date for the one-time confirmation snapshot")
    context_parser = subparsers.add_parser("context-study", help="Compare cached market and earnings filing context")
    context_parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    context_parser.add_argument("--start", default="2018-01-01")
    context_parser.add_argument("--end", required=True)
    earnings_parser = subparsers.add_parser("earnings-study", help="Exploratory public earnings-surprise and dated schedule comparison")
    earnings_parser.add_argument("--start", default="2018-01-01")
    earnings_parser.add_argument("--end", required=True)
    earnings_parser.add_argument("--allow-retrospective-snapshot", action="store_true")
    fetch_parser = subparsers.add_parser("fetch-context", help="Download public market and SEC context")
    fetch_parser.add_argument("--start", default="2017-01-01")
    fetch_parser.add_argument("--end", required=True)
    for command_parser in [run_parser, batch_parser, baseline_parser, study_parser, context_parser]:
        command_parser.add_argument("--cached", action="store_true", help="Use saved raw prices without network access")
        command_parser.add_argument("--fee-bps", type=float, default=5.0, help="Fee per side in basis points")
        command_parser.add_argument("--slippage-bps", type=float, default=2.0, help="Adverse fill slippage per side")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings(use_cached_data=getattr(args, "cached", False),
                        trading_fee_bps=getattr(args, "fee_bps", 5.0),
                        slippage_bps=getattr(args, "slippage_bps", 2.0))
    settings.ensure_directories()

    if args.command == "recent-history-study":
        from .recent_history_study import run_recent_history_study
        result = run_recent_history_study(settings, args.start, args.end, freeze=not args.historical_only)
        print(json.dumps(result["aggregates"], indent=2))
    elif args.command == "recent-history-confirm":
        from .recent_history_confirmation import run_confirmation
        result = run_confirmation(settings, args.end)
        print(json.dumps(result["aggregates"], indent=2))
    elif args.command == "pooled-direction-study":
        from .pooled_direction_study import run_pooled_direction_study
        result = run_pooled_direction_study(settings, args.start, args.end)
        print(json.dumps(result["aggregates"], indent=2))
    elif args.command == "direction-study":
        from .direction_study import run_direction_study
        result = run_direction_study(settings, [ticker.upper() for ticker in args.tickers], args.start, args.end)
        print(json.dumps(result["aggregates"], indent=2))
    elif args.command == "earnings-study":
        from .earnings_study import run_earnings_study
        result = run_earnings_study(settings, args.start, args.end, allow_retrospective_snapshot=args.allow_retrospective_snapshot)
        print(json.dumps(result["coverage"], indent=2))
    elif args.command == "fetch-context":
        from .data.context import fetch_context
        fetch_context(settings.project_root, args.start, args.end)
    elif args.command == "context-study":
        from .context_study import run_context_study
        result = run_context_study(settings, [ticker.upper() for ticker in args.tickers], args.start, args.end)
        print(json.dumps(result["paired_uncertainty"], indent=2))
    elif args.command == "feature-study":
        from .feature_study import run_feature_study
        result = run_feature_study(settings, [ticker.upper() for ticker in args.tickers], args.start, args.end)
        print(json.dumps(result["paired_uncertainty"], indent=2))
    elif args.command == "baseline":
        from .pipeline import PipelineRunner
        result = PipelineRunner(settings).run_baselines([ticker.upper() for ticker in args.tickers], args.start, args.end)
        print(json.dumps(result, indent=2))
        if result["failures"]:
            raise SystemExit(1)
    elif args.command == "run":
        from .pipeline import PipelineRunner

        runner = PipelineRunner(settings=settings)
        runner.run(
            ticker=args.ticker.upper(),
            start_date=args.start,
            end_date=args.end,
            model_name=args.model,
        )
    elif args.command == "run-batch":
        from .pipeline import PipelineRunner

        runner = PipelineRunner(settings=settings)
        result = runner.run_batch(
            tickers=[ticker.upper() for ticker in args.tickers],
            start_date=args.start,
            end_date=args.end,
            model_name=args.model,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "predict-latest":
        from .pipeline import PipelineRunner

        runner = PipelineRunner(settings=settings)
        result = runner.predict_latest(
            ticker=args.ticker.upper(),
            start_date=args.start,
            end_date=args.end,
            model_name=args.model,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "resolve-live":
        from .pipeline import PipelineRunner

        runner = PipelineRunner(settings=settings)
        result = runner.resolve_live_predictions(ticker=args.ticker.upper() if args.ticker else None)
        print(json.dumps(result, indent=2))
    elif args.command == "report-live":
        from .pipeline import PipelineRunner

        runner = PipelineRunner(settings=settings)
        result = runner.live_prediction_summary(ticker=args.ticker.upper() if args.ticker else None)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
