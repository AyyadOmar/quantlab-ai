# Does recent training history improve direction prediction?

Historical development comparison: previously inspected dates, no untouched-data claim. Model defaults and website are unchanged.

All variants use the same 14 compact features, regularized model settings, companies, validation/test dates and label gaps. Training includes all available history, up to four calendar years, or up to two years ending at the last training date. The four-year window can equal full history in early folds. Features retain their normal warm-up history.

Each window uses the unchanged validation-only cutoff rule (seven fixed cutoffs, at least 20 down calls, always-up fallback). The primary policy selects a window by validation correct count, fewer down calls, then Brier/log loss, with full/four/two-year order as the last tie-break. The three fixed-window rows are diagnostics, not candidates selected by test results.

| Model | Training policy | Fixed 0.5 accuracy | Selected-rule accuracy | Always-up | Difference (pp) | Correct / incorrect down | Brier | 95% difference interval (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | full | 51.52% | 52.71% | 53.79% | -1.08 | 224 / 251 | 0.2516 | [-2.71, +0.52] |
| logistic_regression | four_years | 51.24% | 52.71% | 53.79% | -1.08 | 198 / 225 | 0.2516 | [-2.55, +0.28] |
| logistic_regression | two_years | 51.36% | 52.47% | 53.79% | -1.32 | 147 / 180 | 0.2521 | [-2.87, +0.12] |
| logistic_regression | validation_selected | 51.56% | 52.43% | 53.79% | -1.36 | 279 / 313 | 0.2520 | [-3.19, +0.44] |
| xgboost | full | 53.15% | 53.67% | 53.79% | -0.12 | 43 / 46 | 0.2510 | [-0.96, +0.64] |
| xgboost | four_years | 51.44% | 53.39% | 53.79% | -0.40 | 38 / 48 | 0.2518 | [-1.28, +0.36] |
| xgboost | two_years | 51.04% | 51.16% | 53.79% | -2.63 | 150 / 216 | 0.2539 | [-4.35, -1.00] |
| xgboost | validation_selected | 52.71% | 53.51% | 53.79% | -0.28 | 110 / 117 | 0.2515 | [-1.52, +0.84] |

## Validation-selected policy by company

| Company | Model | Accuracy | Always-up | Extra correct | Down calls |
|---|---|---:|---:|---:|---:|
| AAPL | logistic_regression | 55.02% | 55.38% | -3 | 35 |
| AAPL | xgboost | 55.38% | 55.38% | +0 | 34 |
| MSFT | logistic_regression | 49.28% | 52.15% | -24 | 290 |
| MSFT | xgboost | 51.91% | 52.15% | -2 | 22 |
| NVDA | logistic_regression | 52.99% | 53.83% | -7 | 267 |
| NVDA | xgboost | 53.23% | 53.83% | -5 | 171 |

Each company has 836 test sessions, 2023-01-24 00:00:00 through 2026-05-22 00:00:00. Aggregate accuracy weights asset-days equally.

Descriptive intervals resample 20-session blocks 2,000 times, keeping companies on a date together. They do not adjust for repeated historical research or multiple comparisons. Cutoff and window selection reuse validation and can overfit it. No test rows are omitted. A match achieved by predicting up every day does not demonstrate skill.

The matched full-history control should reproduce the previous separate-company compact model. Classification cutoffs do not define a trading strategy; no profit improvement is claimed.

A separate confirmation plan freezes choices using only known outcomes before newer prices are retrieved. Later confirmation is retrospective until predictions are actually timestamped before their outcomes. See confirmation_plan.md and, once available, confirmation.md.
