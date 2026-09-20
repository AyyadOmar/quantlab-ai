# Relative features and smaller-model study

Historical research on the same cached dates as the corrected baseline. No new prospective data was used and no defaults were automatically promoted.

Six predeclared candidates per model family: original (24), relative (24), and compact (14) features, each with baseline or regularized settings. Models and trading thresholds are selected using earlier validation only, before test predictions.

Regularized logistic regression uses C=0.01 instead of 0.1. Regularized XGBoost uses 100 depth-2 trees instead of 300 depth-5 trees, with stronger penalties and minimum leaf weight. See study_plan.json and model code for exact settings.

The 24-feature relative set replaces four price-level moving averages with close/average minus one, and divides three MACD values by close. The compact set removes repeated moving averages, momentum and volatility proxies; it is fixed in advance, not selected using outcomes.

## Aggregate comparisons

Equal-weight means across tickers; net returns are mean ticker returns, not a simulated portfolio. Fit time totals cover the fixed candidate's folds. Lower Brier and log loss are better. Static candidate rows are diagnostics, not a ranking used for selection.

| Model | Candidate | Accuracy | AUC | Brier | Log loss | Mean net return | Mean fit seconds/ticker |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | original_baseline | 50.02% | 0.481 | 0.2614 | 0.7199 | -2.03% | 0.022 |
| logistic_regression | original_regularized | 49.78% | 0.484 | 0.2571 | 0.7086 | -12.52% | 0.019 |
| logistic_regression | relative_baseline | 50.86% | 0.487 | 0.2536 | 0.7008 | 15.79% | 0.022 |
| logistic_regression | relative_regularized | 51.65% | 0.487 | 0.2515 | 0.6964 | 9.98% | 0.019 |
| logistic_regression | compact_baseline | 52.15% | 0.494 | 0.2519 | 0.6973 | 10.52% | 0.016 |
| logistic_regression | compact_regularized | 51.99% | 0.491 | 0.2509 | 0.6950 | 12.91% | 0.015 |
| logistic_regression | validation_selected | 52.11% | 0.488 | 0.2514 | 0.6961 | 10.04% | 0.017 |
| xgboost | original_baseline | 48.40% | 0.494 | 0.3091 | 0.8488 | -9.62% | 0.998 |
| xgboost | original_regularized | 51.39% | 0.499 | 0.2510 | 0.6952 | -2.89% | 0.142 |
| xgboost | relative_baseline | 49.69% | 0.494 | 0.2865 | 0.7851 | -16.35% | 0.996 |
| xgboost | relative_regularized | 51.82% | 0.495 | 0.2513 | 0.6959 | 2.42% | 0.141 |
| xgboost | compact_baseline | 50.19% | 0.498 | 0.2842 | 0.7815 | -22.50% | 0.745 |
| xgboost | compact_regularized | 53.21% | 0.507 | 0.2503 | 0.6939 | -7.19% | 0.114 |
| xgboost | validation_selected | 52.08% | 0.497 | 0.2510 | 0.6953 | -0.13% | 0.131 |

Selection evaluates all six candidates, so its actual search cost is their combined fit cost, not the selected-model fit time shown above. Timings are approximate local wall times.

## Validation-selected model versus original baseline

| Ticker | Model | Original accuracy | Selected accuracy | Always up | Original Brier | Selected Brier | Training-prior Brier | Selected net return | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | logistic_regression | 49.16% | 52.75% | 55.38% | 0.2547 | 0.2499 | 0.2475 | -10.34% | 44 |
| AAPL | xgboost | 47.97% | 53.83% | 55.38% | 0.3066 | 0.2474 | 0.2475 | 9.80% | 104 |
| MSFT | logistic_regression | 50.24% | 50.96% | 52.15% | 0.2591 | 0.2551 | 0.2497 | -2.97% | 46 |
| MSFT | xgboost | 46.41% | 52.27% | 52.15% | 0.3304 | 0.2527 | 0.2497 | -8.30% | 60 |
| NVDA | logistic_regression | 47.85% | 51.44% | 53.83% | 0.2895 | 0.2509 | 0.2494 | 60.83% | 35 |
| NVDA | xgboost | 49.28% | 50.72% | 53.83% | 0.2964 | 0.2547 | 0.2494 | -1.69% | 443 |
| SPY | logistic_regression | 53.95% | 54.43% | 54.78% | 0.2498 | 0.2490 | 0.2478 | 0.00% | 0 |
| SPY | xgboost | 50.00% | 52.15% | 54.78% | 0.2919 | 0.2505 | 0.2478 | -2.76% | 17 |
| QQQ | logistic_regression | 48.92% | 50.96% | 55.38% | 0.2538 | 0.2521 | 0.2473 | 2.69% | 14 |
| QQQ | xgboost | 48.33% | 51.44% | 55.38% | 0.3202 | 0.2498 | 0.2473 | 2.30% | 24 |

## Paired uncertainty

Selected minus original Brier: negative values mean better probability predictions. Intervals resample 20-session blocks after averaging same-date loss differences across tickers. They are descriptive because these historical dates have already been inspected.
- logistic_regression: -0.0100; 95% interval [-0.0133, -0.0071].
- xgboost: -0.0580; 95% interval [-0.0676, -0.0479].

## Interpretation and limits

Evaluation: 2023-01-24 00:00:00 through 2026-05-22 00:00:00, 836 sessions per candidate/ticker in this run. The final partial fold is retained.

Costs remain 5 bps fees plus 2 bps slippage per side; cash earns zero. Accuracy uses 0.5; trades use validation-selected thresholds. Improved probability scores need not improve trading returns or beat always-up accuracy.

Each ticker/model's selected artifact stores its actual feature columns, model, threshold and cutoffs. Predictions, equity curves, full validation scores, data hashes and package versions are saved beside this report. Existing baseline and demo files are preserved.

Next confirmation should use newly collected dates with this candidate list and selection rule frozen. Do not select the best static candidate by the test table.
