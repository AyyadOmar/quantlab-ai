# Predicting down versus always-up

Historical development experiment on previously inspected dates. No fresh holdout was downloaded or evaluated. Model defaults and website results were not changed.

Models retain the six price-feature/regularization candidates and validation Brier selection from v3. Only the classification decision changes: choose among seven predeclared probability cutoffs (0.35 to 0.65, step 0.05) and always-up using validation accuracy. Cutoffs require at least 20 validation down calls; ties prefer fewer down calls, then a lower cutoff. Always-up wins accuracy ties against any eligible cutoff.

An up prediction means the next session closes above its open. A down prediction includes flat sessions. Every test session is scored; no abstentions or discarded low-confidence days. The test result does not choose the rule. Earlier test dates may enter later training once historical, as in the existing expanding-window protocol.

## Aggregate results

Pooled asset-day accuracy on identical dates, with both prespecified model families reported. Correct down calls add successes versus always-up; incorrect down calls remove successes.

| Model | Fixed 0.5 | Selected rule | Always-up | Gain (percentage points) | Correct / incorrect down | 95% gain interval (pp) |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 52.11% | 52.78% | 54.31% | -1.53 | 238 / 302 | [-2.68, -0.33] |
| xgboost | 52.08% | 52.18% | 54.31% | -2.13 | 184 / 273 | [-3.47, -0.96] |

## Per asset

| Asset | Model | Selected rule | Always-up | Extra correct | Down calls | Positive folds / all folds | Always-up fallback folds |
|---|---|---:|---:|---:|---:|---:|---:|
| AAPL | logistic_regression | 55.38% | 55.38% | +0 | 0 | 0 / 5 | 5 |
| AAPL | xgboost | 54.78% | 55.38% | -5 | 31 | 0 / 5 | 3 |
| MSFT | logistic_regression | 49.64% | 52.15% | -21 | 105 | 0 / 5 | 4 |
| MSFT | xgboost | 49.16% | 52.15% | -25 | 97 | 0 / 5 | 4 |
| NVDA | logistic_regression | 53.47% | 53.83% | -3 | 193 | 1 / 5 | 2 |
| NVDA | xgboost | 52.03% | 53.83% | -15 | 123 | 0 / 5 | 2 |
| SPY | logistic_regression | 54.78% | 54.78% | +0 | 0 | 0 / 5 | 5 |
| SPY | xgboost | 53.71% | 54.78% | -9 | 15 | 0 / 5 | 3 |
| QQQ | logistic_regression | 50.60% | 55.38% | -40 | 242 | 1 / 5 | 2 |
| QQQ | xgboost | 51.20% | 55.38% | -35 | 191 | 0 / 5 | 4 |

## Interpretation

Evaluation outcomes span 2023-01-24 00:00:00 to 2026-05-22 00:00:00. See JSON for each asset's row count, full cutoff scores, fold boundaries, and data hashes.

Intervals use 2,000 circular bootstrap samples of 20-session blocks, keeping all assets on a date together. They are descriptive, not proof of an edge: the dates have been examined in earlier research, validation is reused for model and cutoff selection, and intervals do not correct for multiple experiments. Assets are correlated and are not independent replications.

A cutoff changes classification decisions, not probabilities, their Brier scores, or ranking ability. An always-up fallback can match the benchmark without finding any down-day signal. A validation win does not guarantee a test win. Default trading thresholds, costs, trading signals, and website figures remain unchanged; no profit claim follows from this classification study.

If the selected rules fail to improve, keep the baseline and test a separately specified shared model across assets. Any eventual candidate needs a frozen rule and confirmation on untouched dates before promotion.
