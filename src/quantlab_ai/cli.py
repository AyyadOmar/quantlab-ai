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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()
    settings.ensure_directories()

    if args.command == "run":
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
