# Intraday feature-group experiment

Historical development only: previously inspected outcomes, January 24, 2023–May 22, 2026. Three companies, 836 sessions each (2,508 company-days). No new prices were downloaded, and the May–September confirmation and prospective ledger were not used.

The 14-feature compact control is compared with the same model plus completed-session open-to-close return, its trailing five-session arithmetic mean, and trailing 20-session share of up intraday sessions. All include the completed signal day. Rolling inputs use prior cached prices for warm-up; no test days are dropped. Flat sessions count as down.

Both regularized model families retain full expanding training history, validation/test boundaries and one-session label gaps. Cutoffs and the always-up fallback are selected only on validation using the existing rule. There is no new cutoff grid, window search or feature-subset search. All six controls reproduce the prior full-history predictions and probabilities (restoring native numeric precision after CSV parsing; absolute tolerance 1e-12).

| Model | Inputs | Accuracy | Fixed 0.5 accuracy | Balanced accuracy | Down recall | Brier | ROC AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | compact_control | 52.71% | 51.52% | 50.36% | 19.33% | 0.2516 | 0.4950 |
| logistic_regression | compact_intraday | 52.63% | 50.84% | 49.91% | 13.98% | 0.2522 | 0.4926 |
| xgboost | compact_control | 53.67% | 53.15% | 50.15% | 3.71% | 0.2510 | 0.5112 |
| xgboost | compact_intraday | 53.03% | 52.67% | 49.51% | 3.11% | 0.2516 | 0.5083 |

Same-date baselines: always-up **53.79%**, always-down **46.21%**, training-majority **53.79%**. Training-prevalence Brier: 0.2488; lower is better.

## Added features versus the matched control

- logistic_regression: -0.08 percentage points; descriptive 95% interval [-1.16, +0.96].
- xgboost: -0.64 percentage points; descriptive 95% interval [-1.60, +0.32].

Intervals use 2,000 samples of 20-session blocks with companies on a date kept together. They are not adjusted for repeated historical research or multiple comparisons. They are not confirmation of a reliable advantage.

## Every company and model

| Company | Model | Inputs | Accuracy | Always-up | Correct / incorrect down calls |
|---|---|---|---:|---:|---:|
| AAPL | logistic_regression | compact_control | 55.38% | 55.38% | 0 / 0 |
| AAPL | logistic_regression | compact_intraday | 55.38% | 55.38% | 0 / 0 |
| AAPL | xgboost | compact_control | 55.38% | 55.38% | 1 / 1 |
| AAPL | xgboost | compact_intraday | 55.02% | 55.38% | 0 / 3 |
| MSFT | logistic_regression | compact_control | 49.64% | 52.15% | 128 / 149 |
| MSFT | logistic_regression | compact_intraday | 50.00% | 52.15% | 129 / 147 |
| MSFT | xgboost | compact_control | 51.91% | 52.15% | 15 / 17 |
| MSFT | xgboost | compact_intraday | 52.03% | 52.15% | 11 / 12 |
| NVDA | logistic_regression | compact_control | 53.11% | 53.83% | 96 / 102 |
| NVDA | logistic_regression | compact_intraday | 52.51% | 53.83% | 33 / 44 |
| NVDA | xgboost | compact_control | 53.71% | 53.83% | 27 / 28 |
| NVDA | xgboost | compact_intraday | 52.03% | 53.83% | 25 / 40 |

## Reproduce

Run from the project root with the existing local v8 inputs:

```sh
PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
  .venv/bin/python -m quantlab_ai.intraday_study
```

Local JSON records settings, packages, input/source hashes and every fold's validation rule. Model artifacts and predictions are isolated under intraday_study_v10 and excluded from Git. This command never fetches market data. It requires existing v8 historical features, raw prices and control predictions; a fresh Git clone alone is insufficient.

The website, default models, earlier frozen confirmation artifacts and prospective forecasts are unchanged. Classification scores do not establish profitability after costs.
