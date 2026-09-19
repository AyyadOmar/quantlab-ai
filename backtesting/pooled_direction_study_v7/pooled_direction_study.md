# Shared-company models versus always-up

Follow-up after the v6 cutoff experiment failed to beat always-up. All dates remain historical development data already examined in prior studies; no new holdout was used.

Both approaches use the same fixed 14 compact features and regularized settings. One trains on each company separately; the other fits once per fold on Apple, Microsoft and NVIDIA together, with equal observations per company and no company identifier. Only historical training rows are pooled. Every company shares the same date boundaries and label gaps. Scaling and imputation are fitted only on training data.

Each company separately selects its direction cutoff on its own later validation data using the unchanged v6 rule: seven fixed cutoffs, at least 20 validation down calls, always-up fallback, ties favor fewer down calls. Model families and settings are fixed before scoring. All test days count. No earnings snapshot features or fresh data are used.

## Matched aggregate comparisons

These results cover three companies; compare scopes within this table. The earlier v6 aggregate covers five assets and has a different always-up rate.

| Model | Training | Fixed 0.5 | Selected rule | Always-up | Gain (pp) | Correct / incorrect down | Brier | 95% gain interval (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | per_company | 51.52% | 52.71% | 53.79% | -1.08 | 224 / 251 | 0.2516 | [-2.71, +0.52] |
| logistic_regression | shared_companies | 51.48% | 52.51% | 53.79% | -1.28 | 153 / 185 | 0.2512 | [-2.79, +0.24] |
| xgboost | per_company | 53.15% | 53.67% | 53.79% | -0.12 | 43 / 46 | 0.2510 | [-0.96, +0.64] |
| xgboost | shared_companies | 52.55% | 53.51% | 53.79% | -0.28 | 11 / 18 | 0.2502 | [-0.64, +0.00] |

## Per company

| Company | Model | Training | Selected rule | Always-up | Extra correct | Down calls |
|---|---|---|---:|---:|---:|---:|
| AAPL | logistic_regression | per_company | 55.38% | 55.38% | +0 | 0 |
| MSFT | logistic_regression | per_company | 49.64% | 52.15% | -21 | 277 |
| NVDA | logistic_regression | per_company | 53.11% | 53.83% | -6 | 198 |
| AAPL | logistic_regression | shared_companies | 55.38% | 55.38% | +0 | 0 |
| MSFT | logistic_regression | shared_companies | 52.15% | 52.15% | +0 | 0 |
| NVDA | logistic_regression | shared_companies | 50.00% | 53.83% | -32 | 338 |
| AAPL | xgboost | per_company | 55.38% | 55.38% | +0 | 2 |
| MSFT | xgboost | per_company | 51.91% | 52.15% | -2 | 32 |
| NVDA | xgboost | per_company | 53.71% | 53.83% | -1 | 55 |
| AAPL | xgboost | shared_companies | 55.38% | 55.38% | +0 | 0 |
| MSFT | xgboost | shared_companies | 52.15% | 52.15% | +0 | 0 |
| NVDA | xgboost | shared_companies | 52.99% | 53.83% | -7 | 29 |

## Limits and next decision

There are 836 test days per company, from 2023-01-24 00:00:00 to 2026-05-22 00:00:00. The shared model gets three times as many rows, but correlated companies do not provide three times as much independent information.

The descriptive intervals resample 20-session blocks, keeping companies on a date together, with 2,000 draws. They do not account for choosing this follow-up after earlier results or for multiple comparisons. A match achieved by predicting up every day does not demonstrate skill. No model is selected or promoted using this test table.

This classification experiment changes neither default trading rules nor the website. Probability accuracy and profit are different questions. If neither approach establishes useful down-day discrimination, stop adjusting cutoffs on these dates. The next hypothesis should change the information available at prediction time and be specified before collecting a new confirmation period.
