# QuantLab AI

QuantLab AI is a production-style machine learning platform for equity data ingestion, quantitative feature engineering, predictive modeling, and backtesting systematic trading strategies. The project is designed to look and feel like a serious ML engineering codebase suitable for a software engineering or machine learning internship portfolio.

## Why this project stands out

- End-to-end pipeline from market data download to backtest evaluation
- Multiple model families: classical ML and deep learning
- Modular Python package with clean separation of concerns
- Reproducible configuration and model persistence
- Walk-forward validation instead of a single naive holdout split
- Benchmark-aware backtesting against buy-and-hold, always-long, and momentum baselines
- Notebook support for experiments without turning the repo into a notebook-only project

## Architecture

```text
quantlab-ai/
├── api/                    # Future Flask API surface
├── backtesting/            # Strategy simulation and trade analytics
├── data/
│   ├── processed/          # Feature-engineered datasets
│   └── raw/                # Downloaded market data snapshots
├── models/                 # Serialized trained model artifacts
├── notebooks/              # Research and exploratory analysis only
├── src/
│   └── quantlab_ai/
│       ├── data/           # Data ingestion, storage, validation
│       ├── features/       # Technical indicators and dataset builders
│       ├── models/         # Trainers, interfaces, and prediction services
│       ├── backtesting/    # Portfolio simulation engine
│       ├── visualization/  # Charts and evaluation plots
│       ├── utils/          # Shared helpers
│       ├── config.py       # Runtime configuration
│       └── pipeline.py     # Orchestrates training and evaluation
├── visualizations/         # Exported charts and analysis images
├── requirements.txt
└── pyproject.toml
```

## Folder and file roles

### Root folders

- `data/raw/`: locally cached OHLCV price history pulled from Yahoo Finance
- `data/processed/`: model-ready tabular datasets after feature engineering
- `models/`: persisted sklearn pipelines, PyTorch checkpoints, and metadata
- `notebooks/`: experiments, diagnostics, and feature research
- `backtesting/`: exported trade logs and summary reports
- `visualizations/`: saved charts such as equity curves and confusion matrices
- `api/`: reserved for a later Flask or FastAPI serving layer

### Source package

- `src/quantlab_ai/config.py`: central settings object, paths, and defaults
- `src/quantlab_ai/pipeline.py`: main training workflow orchestration
- `src/quantlab_ai/data/loader.py`: pulls historical data from `yfinance`
- `src/quantlab_ai/data/repository.py`: stores raw data and experiment metadata in SQLite
- `src/quantlab_ai/features/indicators.py`: technical indicators like RSI and volatility
- `src/quantlab_ai/features/builder.py`: feature table creation and supervised labels
- `src/quantlab_ai/models/base.py`: shared model interface
- `src/quantlab_ai/models/classical.py`: logistic regression, random forest, optional XGBoost
- `src/quantlab_ai/models/lstm.py`: PyTorch sequence model for next-day movement
- `src/quantlab_ai/models/evaluator.py`: metrics and confusion matrix generation
- `src/quantlab_ai/models/registry.py`: save and load model artifacts
- `src/quantlab_ai/backtesting/engine.py`: strategy simulation, equity tracking, summary stats
- `src/quantlab_ai/visualization/plots.py`: candlestick, equity, and evaluation charts
- `src/quantlab_ai/utils/logging.py`: consistent application logging
- `src/quantlab_ai/cli.py`: command-line entrypoint for the platform

## ML pipeline

1. Download OHLCV data for a selected ticker and date range
2. Persist the raw dataset to disk and SQLite metadata storage
3. Engineer quantitative features:
   - simple and exponential moving averages
   - RSI
   - rolling volatility
   - momentum
   - lagged returns
   - volume trend indicators
4. Generate next-day movement labels
5. Train multiple models with walk-forward expanding-window validation
6. Evaluate precision, recall, F1, ROC AUC, and confusion matrix
7. Convert probabilities into trading signals
8. Backtest the signal-driven strategy against benchmark strategies
9. Save artifacts, plots, and experiment outputs

## Supported models

- Logistic Regression
- Random Forest
- XGBoost when the dependency is available
- PyTorch LSTM classifier

## How to run

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
PYTHONPATH=src python3 -m quantlab_ai.cli run \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --model random_forest
```

### 3. Train the LSTM

```bash
PYTHONPATH=src python3 -m quantlab_ai.cli run \
  --ticker MSFT \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --model lstm
```

## Outputs

- Raw CSV cache in `data/raw/`
- Processed feature dataset in `data/processed/`
- SQLite experiment database in `data/quantlab.db`
- Saved model artifact in `models/`
- Backtest metrics and trade log in `backtesting/`
- Charts in `visualizations/`

## Baseline experiments

Initial experiments were run on `AAPL` daily data from `2018-01-01` to `2024-12-31` using both classical machine learning models and a PyTorch LSTM. This section is intentionally honest: the goal is not to claim unrealistic alpha, but to show a full quantitative ML workflow that surfaces what is and is not working.

### Evaluation design

- Expanding-window walk-forward validation was used to generate out-of-sample predictions across time rather than relying on a single static split.
- Trading signals were evaluated against multiple baselines:
  - buy-and-hold
  - always-long exposure
  - a simple momentum rule based on prior-day price direction
- This setup makes the repo more representative of how a real quant research workflow evaluates predictive models under temporal constraints.

### AAPL model comparison

| Model | Strategy Return | Benchmark Return | Max Drawdown | Sharpe Ratio | Win Rate | Trade Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `-6.76%` | `+41.85%` | `-7.04%` | `-2.00` | `22.22%` | `9` |
| Random Forest | `-3.45%` | `+41.85%` | `-6.67%` | `-0.90` | `54.17%` | `24` |
| XGBoost | `+0.33%` | `+41.85%` | `-7.55%` | `-0.08` | `55.36%` | `56` |
| LSTM | `-4.62%` | `+41.54%` | `-8.05%` | `-0.80` | `46.15%` | `26` |

### Key observations

- The pipeline successfully executed a full research loop: data ingestion, feature engineering, supervised learning, probability generation, signal creation, backtesting, and visualization.
- `XGBoost` was the strongest baseline of the three, producing a slightly positive strategy return and the best win rate, though it still materially underperformed buy-and-hold.
- `Random Forest` generated fewer trades and a modestly positive win rate, but its signals were not strong enough to overcome missed upside exposure.
- `Logistic Regression` underperformed most clearly, which suggests the current feature set is not linearly separable enough for a simple classifier.
- The `LSTM` completed successfully and added architectural depth to the platform, but on this baseline setup it did not outperform the tree-based models.
- Across all four models, the main lesson is that getting some directional classification signal is easier than producing a robust trading edge after thresholding and transaction costs.
- The benchmark comparison makes the project more credible because it highlights whether a model is truly adding value over naive exposure rules.
- This is exactly the kind of result worth showing recruiters: the platform is working, the evaluation is realistic, and the next iteration path is data-driven rather than hand-wavy.

### Visual outputs

#### Random Forest equity curve

![AAPL Random Forest Equity Curve](visualizations/aapl_random_forest_equity_curve.png)

#### Random Forest confusion matrix

![AAPL Random Forest Confusion Matrix](visualizations/aapl_random_forest_confusion_matrix.png)

#### Random Forest probability distribution

![AAPL Random Forest Probability Distribution](visualizations/aapl_random_forest_probabilities.png)

#### XGBoost equity curve

![AAPL XGBoost Equity Curve](visualizations/aapl_xgboost_equity_curve.png)

#### LSTM equity curve

![AAPL LSTM Equity Curve](visualizations/aapl_lstm_equity_curve.png)

#### Logistic Regression confusion matrix

![AAPL Logistic Regression Confusion Matrix](visualizations/aapl_logistic_regression_confusion_matrix.png)

#### Candlestick chart

![AAPL Candlestick Chart](visualizations/aapl_candlestick.png)

## Roadmap

### Short-term improvements

- Expand feature engineering with MACD, Bollinger Bands, ATR, OBV, and market regime indicators
- Tune the trading threshold instead of using a fixed cutoff
- Add unit tests for indicators, dataset building, and backtesting correctness

### Production-oriented upgrades

- Add a Flask or FastAPI inference API for latest-signal serving
- Add Docker support and GitHub Actions for reproducibility
- Store experiment runs with richer metadata for comparison and tracking
- Extend the backtester to support position sizing, stop-loss rules, and transaction-cost sensitivity
- Add a dashboard layer for model diagnostics and strategy monitoring

## Suggested additions

### Additional ML features

- MACD, Bollinger Bands, ATR, OBV, VWAP, and sector-relative strength
- Regime detection features based on volatility clustering
- Fundamental or macroeconomic features merged with price history
- Cross-asset features using indices, yields, or ETF benchmarks
- Feature importance monitoring across rolling retrains

### Reinforcement learning extensions

- Formulate trading as a sequential decision process with reward shaping
- Add a Gymnasium environment for portfolio allocation
- Train DQN, PPO, or SAC agents on transaction-cost-aware simulations
- Compare RL strategies against supervised signal models in the same backtester

### Deployment ideas

- Flask or FastAPI inference API for latest prediction serving
- Scheduled retraining pipeline via GitHub Actions or Airflow
- Docker image with reproducible environment
- Streamlit dashboard for research review and strategy performance monitoring
- PostgreSQL migration plus object storage for enterprise-style artifact tracking

### Resume-ready bullets

- Built a modular Python quantitative research platform for stock prediction, feature engineering, and backtesting across classical ML and PyTorch models
- Designed a reproducible ML pipeline using pandas, scikit-learn, SQLite, and artifact versioning for end-to-end experiment tracking
- Implemented quantitative indicators, probability-based signal generation, and trading analytics including Sharpe ratio, drawdown, and win rate
- Developed a production-style package architecture with CLI workflows, persistence layers, visualization outputs, and notebook-independent training flows

### How to impress recruiters

- Add automated tests for feature engineering and backtesting correctness
- Include benchmark comparisons against buy-and-hold and naive momentum
- Show time-aware validation and avoid leakage in the README discussion
- Publish screenshots of charts and a short architecture diagram in GitHub
- Add API serving and Docker support to show software engineering depth
- Document trade-offs, limitations, and future extensions like RL and portfolio optimization

## Notes on professionalism

This repo intentionally treats notebooks as optional research tools. The main product is the Python application itself: configurable, modular, testable, and ready to extend into a service or dashboard.
