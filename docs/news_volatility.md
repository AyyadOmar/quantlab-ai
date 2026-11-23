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

## Collect news before the close and measure its tone

QuantLab samples GDELT every 15 minutes throughout the regular US stock session, from the 9:30 a.m. opening bell through the exchange close. The local Codex automation uses the Toronto time zone, and the collector checks the exchange calendar for holidays, daylight saving time and early closes. This requires the local scheduler, workspace and network to be available. Missed batches remain visible when inspecting a cutoff. The first live collection and scoring check ran manually on November 22, 2026. The first scheduled trading session is November 23. A checkout on another machine can run the same command with its own scheduler.

```sh
PYTHONPATH=src .venv/bin/python scripts/collect_press_news.py --scheduled
PYTHONPATH=src .venv/bin/python scripts/collect_press_news.py --as-of 2026-11-23T21:00:00Z
```

The first command downloads a pinned quantized FinBERT model on first use during a collection window. It runs inference locally, saves positive, negative and neutral wording scores per headline version, and timestamps the scores when inference finishes. Stocks mentioned together in a headline are excluded from company sentiment summaries when two or more tracked tickers match. Mixed-company stories involving other issuers, syndicated copies, product promotions and non-English headlines still need review. The full stock universe for this experiment is AAPL, MSFT and NVDA. [FinBERT model card](https://huggingface.co/ProsusAI/finbert) · [Quantized model revision](https://huggingface.co/Xenova/finbert/tree/8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20)

The cutoff report includes three text measurements and their coverage:

- Mean FinBERT positive score minus negative score across scored headlines.
- Mean FinBERT balance for company-specific quotations extracted from the article, when GDELT supplies them. Up to three distinct quotes explicitly mentioning the matched company or a covered product are used. Quotes get equal weight within an article and each article gets equal weight in the final mean.
- Mean GDELT whole-document tone when available. This measures word use throughout the story and can differ from the headline or the company's news. [GDELT field definitions](https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf)

The report uses only the version observed by the cutoff and its scores already available then. Missing observations or scores have null values. Previously captured headlines can be scored today, with today's scoring time. Raw article text and open-ended article scraping are outside this pipeline. We store article URLs, metadata, headlines, selected quotations and numerical scores locally.

Across the first two distinct batches we scored 13 headlines. Four records included full-document tone after that field was added. None included a qualifying company quotation. In a separate wording check, FinBERT distinguished a raised forecast, a cut forecast and an annual meeting announcement. Those checks assess text interpretation. To evaluate price direction, compare price-only, price plus news volume, and price plus volume and text scores on identical future stock sessions with complete coverage. Freeze model settings before evaluating and report missing sessions, both constant baselines and probability error. A positive story can already be reflected in the price.

## Prospective direction comparison

Starting November 23, 2026, the after-close runner snapshots features for Apple, Microsoft and NVIDIA and assigns the **next exchange session's open-to-close direction** as their target. Morning inputs freeze at noon New York time (11 a.m. on a 1 p.m. early close so the two windows stay disjoint). Late-day inputs freeze at the official close. The collector also saves intervening midday batches. Premarket and overnight batches are outside the scheduled window. A capture can run from 4:15 p.m. New York time through that evening, allowing the existing daily price loader to accept the completed bar even on early-close days. Prices come from the project's Yahoo Finance loader. The final 20 signal sessions must all have valid daily open and close prices.

We chose one small pooled logistic regression per input set. Each uses the same three stocks, training dates, standardized inputs, fixed regularization (C = 0.1) and a 0.5 decision threshold:

| Comparison | Available inputs |
|---|---|
| Price | One- and five-session price changes, today's open-to-close return, trailing 20-session RMS intraday return |
| Price and counts | Price inputs plus hourly headline count, count change from the previous hour, number of eligible stories, and how many have document tone or scored company quotations |
| Price, counts and wording | All prior inputs plus average FinBERT headline balance, GDELT article tone and average FinBERT quotation balance |
| Morning and late news | The late-day wording model plus morning story counts, source coverage counts and mean headline, document and quotation scores |

Including the text-coverage counts in both news models makes the last comparison about the numerical wording scores. If a fully observed window has no qualifying articles, the counts and corresponding text inputs are zero. All four models use the same dates. A session qualifies when the morning window and the final two hours have complete batch coverage, and all eligible headlines used in sentiment have a score by their window cutoff. A regular morning requires ten quarter-hour batches and an early-close morning requires six. The report lists coverage failures and missed runs.

The first **20 complete sessions with resolved outcomes** supply initial training examples (60 stock-days). This is a pilot design choice. A fit also needs both up and down examples. After that, each eligible evening the runner fits on previously captured rows with known outcomes and records four probabilities per ticker for the next trading day. All four models continue learning on the same expanding history. A closed session can resolve yesterday's target before today's forecasts are built. The scorecard reports accuracy, correct counts and Brier error for all four arms plus always-up and always-down on identical dates. Small early samples and correlated stocks limit any improvement claim. Compare late-day wording versus counts, then compare morning plus late-day wording versus late-day wording alone. The expanded experiment uses the v2 ledger.

```sh
# Capture today's inputs and settle previous forecasts after close.
PYTHONPATH=src .venv/bin/python -m quantlab_ai.prospective_news --capture
# Read current coverage and the scorecard, offline.
PYTHONPATH=src .venv/bin/python -m quantlab_ai.prospective_news
```

The local weekday after-close automation runs the capture command. Other machines can schedule it at 4:30 p.m. New York time. It safely skips exchange holidays. The local SQLite ledger lives in the ignored GDELT data directory and records signal time, cutoff, target date, coverage, features, training size and forecasts. Separate rows preserve the first resolved prices. Retries cannot overwrite an existing signal. Offline integration tests simulate a sequence of sessions and check when forecasts and scores first become available.
