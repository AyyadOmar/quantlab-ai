# Earnings announcement and guidance-content study

Historical development on already-inspected January 24, 2023–May 22, 2026 outcomes. All 2,508 company-days are scored (836 each for AAPL, MSFT and NVDA). The newer confirmation period and prospective ledger were not used.

**Interpretation:** the subsequent [zero-content diagnostic](guidance_placebo.md) reproduced all 2,508 XGBoost direction calls with the guidance values removed. Its two-prediction lead over always-up is not evidence that guidance improved accuracy. Logistic regression worsened. Neither model is promoted.

## Matched comparisons

Price control uses the existing 14 compact inputs. Release timing adds a five-session earnings-release indicator. Guidance content adds four literal flags: numerical revenue outlook, explicit guidance raise, reduction or withdrawal. It also adds the next-quarter revenue outlook midpoint divided by the just-reported quarter's revenue, minus one, with a separate availability flag. Missing values use zero plus that flag. This is sequential revenue growth in management's outlook, not analyst surprise. Timing is a separate control so an event-date effect is not mistaken for useful announcement content. All models use the same full expanding training history, regularized settings and validation-only cutoff selection. All six price controls reproduce the prior predictions and native-precision probabilities.

| Model | Inputs | Accuracy | Balanced accuracy | Down recall | Brier | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | price_control | 52.71% | 50.36% | 19.33% | 0.2516 | 0.4950 |
| logistic_regression | release_timing | 52.27% | 50.04% | 20.62% | 0.2519 | 0.4942 |
| logistic_regression | guidance_content | 51.48% | 49.67% | 25.88% | 0.2518 | 0.4952 |
| xgboost | price_control | 53.67% | 50.15% | 3.71% | 0.2510 | 0.5112 |
| xgboost | release_timing | 52.87% | 49.66% | 7.33% | 0.2513 | 0.5085 |
| xgboost | guidance_content | 53.87% | 51.09% | 14.50% | 0.2512 | 0.5087 |

Identical-date baselines: always-up **53.79%**, always-down **46.21%**. Training-prevalence Brier: 0.2488 (lower is better).

## Guidance content versus controls and always-up

- logistic_regression versus price_control: -1.24 percentage points; descriptive 95% interval [-2.59, +0.04].
- logistic_regression versus release_timing: -0.80 percentage points; descriptive 95% interval [-1.80, +0.12].
- logistic_regression versus always_up: -2.31 percentage points; descriptive 95% interval [-4.63, -0.04].
- xgboost versus price_control: +0.20 percentage points; descriptive 95% interval [-0.80, +1.16].
- xgboost versus release_timing: +1.00 percentage points; descriptive 95% interval [-0.44, +2.51].
- xgboost versus always_up: +0.08 percentage points; descriptive 95% interval [-1.08, +1.16].

Intervals resample 20-session blocks 2,000 times, keeping companies on a date together. These are exploratory intervals, without correction for repeated research or multiple comparisons.

## Source coverage and extraction

| Company | Indexed announcements | Numeric revenue outlook | Explicit raise | Explicit reduction | Explicit withdrawal | Comparable quarterly outlook |
|---|---:|---:|---:|---:|---:|---:|
| AAPL | 35 | 10 | 0 | 1 | 0 | 9 |
| MSFT | 34 | 0 | 0 | 0 | 0 | 0 |
| NVDA | 36 | 36 | 0 | 1 | 0 | 34 |

Coverage is relative to the fixed 2018–May 2026 SEC 8-K item 2.02 index, not all company news. The public SEC earnings exhibits were retrieved as readable narrative text, linked to the indexed accession, and hashed locally. Legal disclaimers and appended financial tables are outside the narrative extraction where a disclaimer boundary exists. Missing or partial narratives stop the study rather than becoming zero-news observations. Source files, URLs, collection times and extraction evidence remain in the local research dataset.

An unflagged announcement means no supported literal phrase was detected in that document. It does not establish that management gave no guidance in an earnings call. The extractor does not infer analyst surprise, guidance revisions from a different forecast period, or sentiment from optimistic prose. Negated and hypothetical revision statements are excluded. Sparse flags limit what this experiment can establish about richer news models.

## Availability and limitations

Announcement features begin on the calendar day after SEC filing and last five signal sessions. Signals are still made after a completed session for the next session's open-to-close outcome. This deliberately delayed rule can miss the first market reaction. Current downloads of archived filings support retrospective development, not a live record. Filing amendments, conference-call-only guidance and independently reported news are outside this first study. No fresh prices were downloaded or new defaults promoted.

## Event-window diagnostic

This subset is reported alongside all-day accuracy, never as a replacement score. Each model is compared on the same event-window company-days.

| Model | Inputs | Rows | Accuracy | Same-row always-up | Same-row always-down |
|---|---|---:|---:|---:|---:|
| logistic_regression | price_control | 206 | 50.00% | 49.03% | 50.97% |
| logistic_regression | release_timing | 206 | 50.97% | 49.03% | 50.97% |
| logistic_regression | guidance_content | 206 | 51.46% | 49.03% | 50.97% |
| xgboost | price_control | 206 | 51.46% | 49.03% | 50.97% |
| xgboost | release_timing | 206 | 50.00% | 49.03% | 50.97% |
| xgboost | guidance_content | 206 | 53.40% | 49.03% | 50.97% |

## Every company

| Company | Model | Inputs | Accuracy | Always-up |
|---|---|---|---:|---:|
| AAPL | logistic_regression | price_control | 55.38% | 55.38% |
| AAPL | logistic_regression | release_timing | 53.11% | 55.38% |
| AAPL | logistic_regression | guidance_content | 52.99% | 55.38% |
| AAPL | xgboost | price_control | 55.38% | 55.38% |
| AAPL | xgboost | release_timing | 55.02% | 55.38% |
| AAPL | xgboost | guidance_content | 55.14% | 55.38% |
| MSFT | logistic_regression | price_control | 49.64% | 52.15% |
| MSFT | logistic_regression | release_timing | 49.88% | 52.15% |
| MSFT | logistic_regression | guidance_content | 49.88% | 52.15% |
| MSFT | xgboost | price_control | 51.91% | 52.15% |
| MSFT | xgboost | release_timing | 50.72% | 52.15% |
| MSFT | xgboost | guidance_content | 52.75% | 52.15% |
| NVDA | logistic_regression | price_control | 53.11% | 53.83% |
| NVDA | logistic_regression | release_timing | 53.83% | 53.83% |
| NVDA | logistic_regression | guidance_content | 51.56% | 53.83% |
| NVDA | xgboost | price_control | 53.71% | 53.83% |
| NVDA | xgboost | release_timing | 52.87% | 53.83% |
| NVDA | xgboost | guidance_content | 53.71% | 53.83% |

## Reproduce offline

```sh
PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
  .venv/bin/python -m quantlab_ai.guidance_study
```

Requires the existing hashed SEC index, announcement snapshots under data/guidance_v11, v8 historical feature files and saved control predictions. These local datasets are excluded from Git; a fresh clone alone cannot reproduce training. The study writes its plan before fitting and saves fold audits, prediction files and isolated model artifacts. The website is unchanged. No profitability claim follows from these classification scores.
