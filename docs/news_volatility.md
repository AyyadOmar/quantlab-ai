# Do company disclosures and changes in volatility help predict stock direction?

Across 2,508 historical stock sessions, the combined inputs brought logistic regression to **53.63%**, four correct predictions short of always-up. XGBoost scored **52.79%**. The volatility forecast had a more promising result on its own task. Its error in predicting next-session variance was **4.76% lower** than the trailing 20-session baseline.

## What the models could know

We made predictions after the close for the following session's open-to-close return. The direction target was up when the next close exceeded the next open, with down or flat as the other outcome. Apple, Microsoft and NVIDIA each contributed 836 scored sessions from January 24, 2023 through May 22, 2026. The training windows reached back to 2018. These dates had already been used in research.

- **Company disclosures.** The SEC submissions archive included 265 original Form 8-K filings across the full source window from January 2017 through May 24, 2026 (Apple 86, Microsoft 89, NVIDIA 90). For each prediction we counted filings over seven calendar days, including separate counts for earnings (2.02), Regulation FD or other events (7.01 or 8.01), and governance (section 5). A single filing can appear in more than one category. The filing-volume delta subtracts the previous week's count from this week's count. We checked archived source hashes and delayed availability to the day after both the filing date and the acceptance date in New York. This permits after-hours and weekend filings to enter at a later signal close, at the cost of missing an immediate reaction. [SEC submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) · [Form 8-K item definitions](https://www.sec.gov/files/form8-k.pdf)
- **Stock volatility.** We used the existing Yahoo Finance daily open and close snapshots. For each stock, the estimate of next-session variance is 94% of the previous estimate plus 6% of the latest completed session's squared open-to-close return. It starts at the first squared return in the price history. Taking its square root gives a daily return scale. The model sees that estimate and its change since the previous session. For example, a forecast that moves from 1.5% to 2.0% has a delta of +0.5 percentage points. The existing price inputs include broader market context. This study's new volatility forecast is specific to each stock. [Yahoo Finance historical prices](https://finance.yahoo.com/quote/AAPL/history/) · [Market-level VIX data from Cboe](https://www.cboe.com/tradable-products/vix/vix-historical-data)

## Direction results

Each regularized model was fitted separately per stock using an expanding training window. The cutoff was selected using the earlier validation period, with a one-session gap between windows and the same scorekeeping for every variant. The original 14-column price controls reproduced the earlier saved predictions and probabilities for all six company/model combinations. For the four matched comparisons we kept all 21 columns in the same order and filled each inactive feature group with zeros. This matters because XGBoost samples columns when building trees. Simply adding empty columns changed its score.

| Inputs | Logistic accuracy | XGBoost accuracy | Logistic Brier | XGBoost Brier |
|---|---:|---:|---:|---:|
| Original price control, 14 columns | 52.71% | 53.67% | 0.2516 | 0.2510 |
| Matched price control, seven zero columns | 52.71% | 53.87% | 0.2516 | 0.2512 |
| Matched, filings active | 52.19% | 53.55% | 0.2518 | 0.2512 |
| Matched, volatility active | 52.35% | 52.71% | 0.2517 | 0.2510 |
| Matched, both groups active | 53.63% | 52.79% | 0.2519 | 0.2511 |

Always-up scored 53.79% and always-down scored 46.21% on these same rows. Lower Brier scores mean smaller probability errors. With both groups active, logistic regression recognized 6.56% of down/flat sessions and XGBoost recognized 11.48%.

| Company | Always-up | Combined logistic | Combined XGBoost |
|---|---:|---:|---:|
| Apple | 55.38% | 55.38% | 55.50% |
| Microsoft | 52.15% | 50.84% | 49.88% |
| NVIDIA | 53.83% | 54.67% | 52.99% |

Compared with the matched price control, logistic accuracy rose 0.92 percentage points (descriptive 95% interval −0.32 to +2.11) and XGBoost fell 1.08 points (−2.47 to +0.24). Compared with always-up, the differences were −0.16 points (−1.28 to +0.84) and −1.00 points (−2.55 to +0.40). The intervals use 2,000 draws of 20-session blocks and keep all three stocks together by date. Repeated experiments on these historical dates limit what the intervals establish.

## Volatility results

We compared the forecast **variance** with the next session's squared open-to-close return, a noisy daily measure of move size. The baseline takes the mean squared return over the past 20 sessions. For each forecast we computed mean squared error using returns in percentage points, so the error units are percentage points to the fourth power. Large moves, particularly for NVIDIA, have a large influence. This test scores every direction-test row.

| Company | Weighted variance error | 20-session baseline error |
|---|---:|---:|
| Apple | 86.17 | 89.79 |
| Microsoft | 19.37 | 19.96 |
| NVIDIA | 192.57 | 203.26 |
| All | 99.37 | 104.34 |

The pooled relative error reduction was 4.76%. A descriptive 95% interval from jointly resampling 20-session blocks across the three stocks was +1.55% to +12.21%. This reflects uncertainty within the historical sample.

## External headlines for future experiments

The press collector reads the latest GDELT 2.1 Global Knowledge Graph batch. It checks the archive against the public index, matches company names or selected products in headlines, and records the publisher domain, article URL, headline, optional publisher timestamp, GDELT batch time and the time QuantLab first observed it. A headline becomes eligible at first observation. Each run records source health and coverage. Sampling the latest batch leaves gaps between runs, and the English name rules can miss relevant stories or match incidental mentions. [GDELT data](https://www.gdeltproject.org/data.html)

The November 22, 2026 17:00 UTC batch contained 699 web records and yielded nine distinct matched articles from seven publishers (Apple 2, Microsoft 0, NVIDIA 7). Seven carried a publisher timestamp. Collecting the same batch again added zero duplicate versions. A zero in one batch says little about daily news volume. The direction results above use SEC filing activity. This headline collection started after the evaluation period.

```sh
PYTHONPATH=src .venv/bin/python scripts/collect_press_news.py
PYTHONPATH=src .venv/bin/python scripts/collect_press_news.py --as-of 2026-11-22T17:15:00Z
```

The first command collects one batch. The second reads the saved ledger and reports each stock's count of newly observed article URLs and its change relative to the preceding hour. It uses the batch time for window membership and requires collection by the specified cutoff. A count requires all four quarter-hour batches, and a delta requires both hours to be complete. A revised title at the same URL counts once. At the example cutoff only one batch was present, so the count and delta are unavailable. Similar stories at different URLs can still be counted twice. The commands use the local directory `data/news_v9/gdelt/`. The next direction test needs regular collection through signal close, an audit of matched headlines, and forecasts recorded before their outcomes. [Original sample archive](https://data.gdeltproject.org/gdeltv2/20261122170000.gkg.csv.zip)

## Run the experiment

From the project root with dependencies installed, the following command exercises the full feature-to-prediction path on synthetic inputs. It runs offline on a fresh checkout.

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_news_volatility_study.py
```

To reproduce the historical evaluation, supply the original local price files from `data/raw/`, SEC manifest and archive files from `data/context_v4/`, v8 features from `data/processed/recent_history_study_v8/`, and v8 control predictions from `backtesting/recent_history_study_v8/`. These caches are excluded from Git. The [context study](context_study.md) and [training history study](recent_history_study.md) describe how those inputs were produced. Later provider snapshots can differ.

```sh
PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
  .venv/bin/python -m quantlab_ai.news_volatility_study
```

The runner saves the plan, source digests, software versions, fold boundaries, direction probabilities, variance forecasts and per-company scores under `backtesting/news_volatility_v12/`. An employer can inspect how each input becomes available, check the fixed comparisons and run the offline tests. A portfolio can report the historical variance result with its baseline. A claim about future accuracy requires forecasts recorded before their outcomes and scored on fresh sessions.
