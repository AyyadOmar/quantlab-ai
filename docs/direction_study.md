# Testing direction rules against always-up

These are historical development experiments, not evidence from an untouched final test. They use cached prices from 2018-01-01 through the exclusive end date 2026-05-25. Features describe completed session T; the label is whether session T+1 closes above its open. Flat sessions are included in the non-up class.

## Recorded outcome

Neither experiment beat always-up in its aggregate comparison. In v6, the selected direction rule scored 52.78% for logistic regression and 52.18% for XGBoost versus 54.31% always-up across five assets. In v7's matched three-company comparison, regularized compact XGBoost scored 53.67% when trained separately and 53.51% when trained jointly, versus 53.79% always-up. Shared logistic regression also remained below its benchmark. No default model was promoted.

The small deficit for compact XGBoost is not evidence of an advantage: its down calls corrected 43 always-up errors but introduced 46 errors. Shared XGBoost corrected 11 and introduced 18. Predicting up more often brings the score nearer the baseline without demonstrating useful down-day discrimination.

Validation completed with 43 passing tests. Saved final-fold artifacts reproduced their probabilities and decisions. The matched separate-company controls also reproduced the prior compact model's outputs exactly.

## 1. Classification cutoffs (`direction_study_v6`)

The prior study selected among six feature/regularization candidates by validation Brier score, then log loss. Keep that selection unchanged. The old classifier predicts up at probability 0.5; the new rule selects among cutoffs 0.35, 0.40, 0.45, 0.50, 0.55, 0.60 and 0.65 using validation accuracy. Always-up is an explicit candidate.

Cutoffs must produce at least 20 down calls on validation to be eligible. This predeclared count prevents selecting a cutoff based on just a handful of examples; it is not a statistical significance guarantee. Ties prefer fewer down calls, then the lower cutoff, so an accuracy tie with always-up keeps always-up. No test outcomes select the candidate or cutoff. A validation win can still lose on later data.

Logistic regression and XGBoost are both reported for AAPL, MSFT, NVDA, SPY and QQQ. Every test day counts. The report shows the prior fixed cutoff, selected rule, always-up, correct/incorrect down calls, fold results and uncertainty. One correct down call adds one success relative to always-up; one incorrect down call removes one. Probabilities remain unchanged: changing a cutoff cannot improve their ranking ability or Brier score.

The previous study's probabilities and fixed-cutoff labels were reproduced exactly for all ten ticker/model controls. [Results](../backtesting/direction_study_v6/direction_study.md).

## 2. Matched shared-company training (`pooled_direction_study_v7`)

After v6 failed to beat always-up, test whether shared training helps. Before evaluating this follow-up, fix the 14 compact features, regularized model settings, company list (AAPL, MSFT, NVDA), and unchanged v6 cutoff rule. Do not search feature sets or model settings in this experiment.

Compare each company trained separately with a model fitted to all three companies' training rows. The shared model is fitted once per model family and fold. Scaling and imputation use only training rows; no company identifier or absolute price-level feature is added. Both scopes use identical dates and model settings, so their difference reflects sharing training observations. Each company selects its own cutoff using its own validation rows, after model fitting. All companies use synchronized calendar windows with a gap between label outcomes and the next window. Missing/mismatched calendars fail rather than silently change the sample.

Both scopes and both model families are reported; none is chosen because it tops the test table. These results cover three companies, whereas v6 covers five assets. The extra company rows are correlated observations, not independent extra days. [Results](../backtesting/pooled_direction_study_v7/pooled_direction_study.md).

## Evaluation and uncertainty

Both studies retain the expanding chronological windows and one-session label gaps in [the research protocol](research_protocol.md). Initial training, validation, and test proportions are unchanged. Earlier test observations may enter training in later folds once their outcomes are historical. Validation is reused for model selection and classification cutoffs in v6; v7 fixes model settings and uses validation only for the cutoff. No random split mixes later and earlier observations.

The confidence intervals resample 20-session blocks of daily accuracy differences, with all assets for a day kept together, for 2,000 seeded repetitions. They are descriptive and do not correct for repeated research or multiple comparisons. A positive result on these already inspected dates would still require confirmation on fresh dates under a frozen rule. Matching always-up by issuing no down calls is not evidence of a learned advantage.

These studies evaluate classification, not trading. They do not change trade thresholds, costs, live prediction behavior, website snapshots or model defaults. Their saved last-fold models are research artifacts, not models refitted on all available observations.

## Reproduce

From the project root, with the existing raw-price cache:

```sh
PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli direction-study \
  --start 2018-01-01 --end 2026-05-25

PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli pooled-direction-study \
  --start 2018-01-01 --end 2026-05-25

PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantlab-matplotlib \
.venv/bin/python -m unittest discover -s tests -v
```

Both commands force cached-only loading, regardless of the default loader setting. They never download or evaluate fresh holdout dates automatically. The library path above is specific to this Mac's existing XGBoost/OpenMP setup.

Each versioned output directory contains a plan written before fitting, source/data hashes, package versions, JSON fold audits, row-level predictions and a readable Markdown report. Model files and raw output CSV/JSON files remain excluded from Git; code, tests, documentation and Markdown summaries are intended for review.
