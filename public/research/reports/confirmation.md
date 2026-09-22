# Frozen-model confirmation on newer dates

The window and cutoff choices were saved before this price snapshot was downloaded. These outcomes were not used to refit models or choose a winning family. This is retrospective confirmation on newer dates, not a record of forecasts made before those historical events.

| Model | Frozen policy | Accuracy | Always-up | Difference (pp) | Correct / incorrect down | Brier | 95% difference interval (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | full | 47.33% | 47.74% | -0.41 | 10 / 11 | 0.2626 | [-2.47, +1.65] |
| logistic_regression | validation_selected | 51.85% | 47.74% | +4.12 | 11 / 1 | 0.2539 | [+1.65, +7.82] |
| xgboost | full | 47.74% | 47.74% | +0.00 | 0 / 0 | 0.2553 | [+0.00, +0.00] |
| xgboost | validation_selected | 48.15% | 47.74% | +0.41 | 6 / 5 | 0.2559 | [-1.65, +2.47] |

## Selected policy by company

| Company | Model | Frozen window | Accuracy | Always-up | Correct / incorrect down |
|---|---|---|---:|---:|---:|
| AAPL | logistic_regression | two_years | 53.09% | 53.09% | 0 / 0 |
| AAPL | xgboost | full | 53.09% | 53.09% | 0 / 0 |
| MSFT | logistic_regression | two_years | 46.91% | 46.91% | 0 / 0 |
| MSFT | xgboost | four_years | 48.15% | 46.91% | 6 / 5 |
| NVDA | logistic_regression | two_years | 55.56% | 43.21% | 11 / 1 |
| NVDA | xgboost | full | 43.21% | 43.21% | 0 / 0 |

Inspect the company-level counts before generalizing an aggregate improvement. Matching always-up without any down calls is not evidence of learned direction skill.

81 sessions per company, from 2026-05-26 00:00:00 through 2026-09-18 00:00:00. Apple, Microsoft and NVIDIA share identical dates. Models remain frozen throughout.

The comparison uses all available sessions in the declared snapshot, not a period picked because it scores well. Intervals use 20-session blocks, keeping all companies on each date together. A short period may have very few effective independent blocks and cannot establish a reliable edge. Models and prices can still be affected by historical selection or data revisions. No trading returns are claimed.

Prospective records eligible at this run: 6; newly appended: 6. The local ledger only records the latest session if a later weekday has not begun, and never replaces a prior prediction. Its target is the next actual market session, with outcomes initially unresolved.

No scheduled collection is configured. Repeating this confirmation command reproduces its immutable snapshot; it does not silently refresh or retune models. Continued prospective collection and later outcome resolution are separate work. Website and default models remain unchanged.
