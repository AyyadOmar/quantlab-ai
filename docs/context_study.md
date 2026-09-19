# Market and earnings context

This experiment adds external information to the existing price-feature/model candidates. Read the [results](../backtesting/context_study_v4/context_study.md). The new candidates remain research options; defaults and public demo snapshots are unchanged.

## Inputs and availability

- **Technology-sector context:** XLK one- and five-session returns, plus the stock's five-session performance relative to the latest available XLK five-session return. XLK is a technology exposure proxy for this technology-heavy basket, not a full sector-classification system for arbitrary stocks.
- **Volatility:** VIX level and five-session percentage change.
- **Interest rates:** TNX level and five-session change; TNX minus IRX as a yield-curve proxy. These are quoted Yahoo index values. IRX is a bill yield measure, so the difference is not an official constant-maturity Treasury spread.
- **Earnings-related context:** calendar days since the last SEC Form 8-K item 2.02 filing, capped at 180; a recent-filing indicator (within five calendar days); and an indicator that past filing history is known.

Market observations must have dates **strictly earlier than the signal date**; values over seven calendar days old are rejected. Features are calculated on source histories before the backward join. No forward-looking joins or backward filling are allowed. The stock's current completed-session features remain available under the established after-close prediction schedule.

An earnings filing is usable beginning the calendar day after its SEC filing date. The model never uses a later filing's event date to infer what was known earlier. This is deliberately conservative and can miss the initial reaction: the SEC filing can follow the actual company announcement. It is **not** an upcoming earnings calendar, an EPS surprise dataset, or a measure of earnings magnitude. Those additions would require reliable release timestamps and historically available analyst expectations.

Only AAPL, MSFT and NVDA use earnings features. SPY and QQQ exclude earnings candidates altogether. Adding all-zero columns can change XGBoost's random column sampling, so merely setting ETF earnings inputs to zero would confound the experiment.

Official references: [SEC submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) describes recent and archived filing histories; [Form 8-K](https://www.sec.gov/files/form8-k.pdf) describes item 2.02, Results of Operations and Financial Condition. Market data comes from Yahoo Finance through yfinance.

## Download and reproduce

The completed run downloaded context from January 2017 through May 22, 2026, supplying a warmup period before the existing 2018 training history. Raw stock/ETF prices remain the existing cache, so the comparison's labels and execution prices are unchanged.

```sh
PYTHONPATH=src .venv/bin/python -m quantlab_ai.cli fetch-context \
  --start 2017-01-01 --end 2026-05-25

PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli context-study \
  --cached --start 2018-01-01 --end 2026-05-25
```

Only the first command needs network access; it refreshes `data/context_v4/`. Preserve that directory and its manifest to reproduce this exact snapshot. Downloads include current and relevant historical SEC submissions archives and select unamended 8-K item 2.02 records, deduplicated by accession. Extra financial-results filings may occur outside quarterly earnings. The downloaded histories contain 39 AAPL, 38 MSFT and 40 NVDA qualifying filings. These counts are coverage information, not quarterly-event counts.

Market histories downloaded today may contain corrections; they are not archived point-in-time market-data vintages. The filing-based availability rules reduce future-information leakage without pretending that the entire vendor dataset is vintage-perfect.

## Comparisons

The original six feature/model candidates remain available. New candidates append market, earnings, or both types of context to relative/compact price features, using the fixed regularized model settings from the previous study. ETFs omit earnings variants.

Within each existing chronological fold, four separate selection policies use only earlier validation scores:

1. Price-only: the preceding study's six candidates.
2. Market available: price-only plus market candidates.
3. Earnings available: price-only plus earnings candidates (identical to price-only for ETFs).
4. Both available: all applicable candidates.

Each policy minimizes validation Brier score, then log loss; each uses the chosen model's validation-only trading threshold. It may retain a price-only model. Model hyperparameters, execution costs and prediction dates remain unchanged. All test dates must match; missing/stale market observations raise errors rather than silently reducing the sample.

There are 40 policy/ticker/model comparisons. The report also contains fixed compact-model diagnostics and paired uncertainty intervals. The intervals are descriptive, not corrected for repeated research or multiple comparisons. The entire period was already examined in earlier work; new dates are still needed for prospective confirmation.

## Outputs

- `data/context_v4/`: market CSVs, raw SEC JSON, filing-event CSVs, timestamped source manifest and hashes.
- `data/processed/context_study_v4/`: combined features and source/availability dates for audit.
- `backtesting/context_study_v4/`: study plan, full results, validation choices, predictions, equity curves and readable report.
- `models/context_study_v4/`: the last fold's validation-selected models, actual feature columns, trading thresholds and cutoffs.

Saved models require the same context joins at inference time. Do not supply zeros when live context is missing or stale. This study does not wire the new candidates into the demo or make them the default strategy.

## Checks

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantlab-matplotlib \
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover same-day/future market exclusion, stale inputs, next-day filing availability, SEC event filtering, ETF applicability and all prior selection/execution protections. Price-only predictions and signals are also compared exactly with the preceding feature-study files.
