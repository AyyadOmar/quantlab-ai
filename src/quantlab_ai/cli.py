from __future__ import annotations

import argparse

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


if __name__ == "__main__":
    main()
