# Recent training history and frozen confirmation

## What changed

This experiment holds the prediction target, three companies (AAPL, MSFT, NVDA), 14 compact inputs, regularized model settings and historical evaluation dates fixed. It compares all available training rows with up to four or two calendar years. Windows end at the last training signal date, before the separate validation window. They do not reach forward into validation or test data. The four-year variant can equal full history in early folds when fewer years are available.

Both logistic regression and XGBoost are reported. Each window uses the existing seven classification cutoffs and always-up fallback, chosen on validation only. The window-selection policy ranks validation correct counts, then fewer down calls, Brier score, log loss, and finally full/four/two-year order. Fixed-window test scores are diagnostic only. No model family is chosen by the test table.

## Historical result

The familiar evaluation period covers 836 sessions per company, January 24, 2023 through May 22, 2026. The full-history controls exactly reproduce the previous six separate-company outputs.

| XGBoost training policy | Accuracy | Always-up |
|---|---:|---:|
| Full history | 53.67% | 53.79% |
| Up to four years | 53.39% | 53.79% |
| Up to two years | 51.16% | 53.79% |
| Validation-selected window | 53.51% | 53.79% |

Selected logistic regression scored 52.43% versus 53.79% always-up. These development results do not support a general claim that discarding older training data helps. [Full historical report](../backtesting/recent_history_study_v8/recent_history_study.md).

## Newer-period confirmation

Before any newer quotes were downloaded, the workflow fitted and saved a separate confirmation model for each company/family using outcomes known by May 22, 2026. The final validation window ends at that cutoff; model fitting ends on February 21, 2025, preserving the existing long validation period. There is no refit on newer prices. [Frozen choices](../backtesting/recent_history_study_v8/confirmation_plan.md).

The public Yahoo snapshot covers confirmation outcomes from May 26 through September 18, 2026: 81 sessions per company, 243 company-days. The first signal is made from the May 22 close, for the next actual trading session. Model choices, artifact hashes and freeze time are recorded before downloading; all required sources must be available before scoring. The original development price cache is preserved, historical price-unit consistency is checked, and confirmation sources have their own immutable hashes.

| Frozen policy | Accuracy | Always-up | Additional correct predictions |
|---|---:|---:|---:|
| Logistic regression, full history | 47.33% | 47.74% | −1 |
| Logistic regression, validation-selected history | **51.85%** | 47.74% | **+10** |
| XGBoost, full history | 47.74% | 47.74% | 0 |
| XGBoost, validation-selected history | 48.15% | 47.74% | +1 |

This is a promising, limited result. All ten additional correct logistic predictions came from NVIDIA (11 correct down calls, one incorrect). Apple and Microsoft simply matched always-up. The period has only 81 market sessions; company-days are correlated, and there are few independent 20-session blocks. The descriptive intervals are not adjusted for the larger history of experiments. The models still have weak probability scores, and no trading-cost or profit improvement has been established.

This period was newly retrieved and unscored in this experiment, but the predictions were produced retrospectively. It is not a record of forecasts made before those historical events, and it is no longer an untouched period after this run. Keep both model families visible and do not retune on this confirmation table. [Complete confirmation report](../backtesting/recent_history_study_v8/confirmation.md).

## Starting the prospective record

Six forecasts were appended on September 20, 2026 (UTC), using the completed September 18 session and the frozen models. Their target is the next actual trading session's open-to-close direction. Outcomes were unknown when recorded. The local ledger is `data/recent_history_v8/prospective_predictions.jsonl`.

Each record includes creation time, input date, probability, direction, chosen window, source hash and freeze hash. Existing predictions cannot be replaced by rerunning the command. A conservative weekday guard prevents labeling an old prediction as prospective after a later weekday has begun. These are frozen-model research forecasts, not automatically refreshed live models or a trading recommendation.

No scheduled collection or automatic outcome resolution is enabled. The initial six records start the prospective track; further collection and resolution need a separate workflow. Repeating confirmation uses the same saved snapshot and does not silently download new dates.

## Reproduce

From the project root, using the existing environment and original raw-price cache:

```sh
# First run: development comparison plus a write-once confirmation freeze.
PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli recent-history-study \
  --start 2018-01-01 --end 2026-05-25

# Once frozen: retrieve and evaluate the one-time confirmation snapshot.
# On this workspace the snapshot already exists, so this reproduces it offline.
PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli recent-history-confirm --end 2026-09-21

# Reproduce historical comparisons without replacing the existing freeze.
PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli recent-history-study --historical-only

PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantlab-matplotlib \
.venv/bin/python -m unittest discover -s tests -v
```

Model files, the freeze JSON, downloaded price snapshots and forecast records remain local and excluded from Git. Commit the code, tests, documentation, historical report, frozen-choice Markdown, and confirmation report. The current local freeze cannot be overwritten, and a confirmation end date cannot be silently changed after its snapshot is recorded. Do not delete these safeguards to search for a more attractive confirmation result.

The website and default prediction/trading models are unchanged. A future website entry can describe this as preliminary retrospective confirmation, with its date range and concentration in NVIDIA. A general claim of beating always-up should wait for broader prospective evidence.
