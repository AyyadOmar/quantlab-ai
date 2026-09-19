# Public earnings surprises and advance-notice schedules

Read the [completed exploratory report](../backtesting/earnings_study_v5/earnings_study.md). Public sources were used; no paid data or account credentials were needed. The default models and website demo were not changed.

## What was obtained

Yahoo's public tables supplied complete EPS estimate, reported EPS and percentage-surprise values for **38 dates per company** (AAPL, MSFT and NVDA) from January 2017 through the May 2026 study cutoff. The saved tables extend beyond this interval, but joins never admit future events. Estimate-only upcoming rows are discarded from result features.

These are **current snapshots**, not archived historical versions. The provider's percentage surprise is used directly because displayed EPS values may be rounded or adjusted. The extractor does not recompute surprise by dividing rounded EPS numbers, or silently assume current figures were exactly what traders saw at the time. Snapshot inputs require explicit opt-in to retrospective exploration. They cannot establish a leakage-free prospective improvement.

NVIDIA's public newsroom archive supplied **34 dated advance earnings-call notices** during the same period. Seventy archive pages were traversed back into 2016; one abbreviated summary was resolved using its linked article. The records include publication date, conservative known date, scheduled call date, source URL and source hash. Four report dates had no matching advance notice: 2017-08-10, 2018-02-08, 2019-02-14 and 2022-08-24. They remain unknown; no publication dates were invented. Current archived pages may have been updated, and this collection is not an independently preserved vintage archive.

Schedules are tested for **NVDA only**. Microsoft's archive returned HTTP 403 to the downloader, and comparable historical Apple advance-notice coverage was not obtained. Neither was given synthetic calendar features. The generic schedule join supports event revisions when a stable event ID and publication history are supplied; this public extraction is not a complete revision/cancellation audit.

## Features and timing

Surprise inputs:

- Last reported percentage surprise, represented as a fraction and clipped to [-1, 1].
- Surprise decayed with a fixed 20-calendar-day time constant.
- Days since the earnings report, capped at 180.
- Whether a report no more than 180 days old is known.

No surprise is usable on its report date; it becomes available on the following calendar day. If that day is not a trading day, the next available signal session receives it. Reports older than 180 days become unknown and their surprise features become zero. With after-close signals and next-open entries, this conservative lag can miss the first reaction session.

Upcoming-call inputs:

- Days until the next known earnings call, capped at 90; unknown also maps to 90 and has a separate flag.
- Whether a known call is within five calendar days.
- Whether an upcoming call is known.

A dated advance notice becomes usable on the day after publication. At each signal date, the join considers only then-known records, keeps the latest known revision of each event, and selects the nearest upcoming date. These are call dates, not exact results-release timestamps. A retrospectively listed earnings date does not count as advance notice.

## Evaluation

The previous six price-only candidates remain the controls. Relative and compact price features can be augmented with surprises, schedules, or both, using the same fixed regularized models. Each scope selects candidates and trading thresholds on earlier validation data. The test dates, target, costs and split boundaries remain unchanged.

- AAPL and MSFT: price-only versus surprise-snapshot selection.
- NVDA: price-only, surprise-snapshot, dated-schedule and combined selection.
- Two model families: logistic regression and XGBoost.
- **16 policy/ticker/model comparisons**, with 836 test sessions per comparison.

All six price-only company/model predictions and signals reproduce the preceding feature study exactly. Small accuracy changes should not be confused with reliable probability estimates or profitable trading. Thousands of daily feature rows repeat a small number of quarterly events; they do not provide thousands of independent earnings observations. All historical dates have already appeared in earlier research.

## Reproduce

From the project root, fetch or refresh the public snapshots:

```sh
.venv/bin/python scripts/fetch_earnings_public.py
.venv/bin/python scripts/fetch_nvidia_schedules.py
```

These commands require network access. The schedule collector reuses saved archive pages; preserve the entire `data/earnings_v5/` folder for exact reproduction. The Yahoo snapshot fetch refreshes its files, so preserve an old snapshot before refreshing it if you need exact old results.

Run the cached experiment:

```sh
PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli earnings-study \
  --start 2018-01-01 --end 2026-05-25 --allow-retrospective-snapshot
```

Without the final flag, the command refuses to present unverified snapshots as a historical experiment. The source library also rejects them by default. This flag is a research-mode label, not a claim that vintage problems have been solved.

Run the tests:

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantlab-matplotlib \
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover report-day/future exclusion, missing actuals, zero EPS estimates, stale surprise values, advance-publication timing, schedule revisions, and prior validation/backtest protections.

## Outputs

The `earnings_study_v5` subdirectories under `data/processed`, `backtesting` and `models` keep new experiments separate. The full report includes raw-data hashes, source manifests, model selections, thresholds, package versions and diagnostics. Model artifacts are explicitly marked exploratory and must be supplied with their saved feature columns and availability rules.

Public references: [Yahoo earnings calendar](https://finance.yahoo.com/calendar/earnings/), [NVIDIA newsroom](https://nvidianews.nvidia.com/news). Each extracted schedule record points to the individual official announcement. The `lxml` parser was installed in the existing project environment and added to requirements for reproducible Yahoo-table parsing.
