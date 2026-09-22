# News research: collection foundation and next experiment

## Why investigate news?

The [direction diagnostic audit](direction_diagnostic_audit.md) compares saved predictions on identical dates against both constant directions, training-majority and a causal 20-session majority rule. In the May 26–September 18, 2026 period, selected logistic regression scored 51.85%, always-up 47.74%, always-down 52.26%, and recent-majority 48.15%. Logistic regression recognized 11 of 127 down days while getting 115 of 116 up days right. Its balanced accuracy was 53.90%, but all improvement over always-up came from NVIDIA. Its probability error also remained slightly worse than a training-prevalence constant.

There is no evidence that a loss of accuracy was required to beat always-up. The periods have different up-day frequencies. On the identical newer dates, full-history logistic regression scored 47.33% and validation-selected history scored 51.85%. The longer historical evaluation still shows little separation beyond chance. Another cutoff search on the already-inspected confirmation dates would not provide independent evidence.

News is a hypothesis for adding information absent from price indicators, not a demonstrated improvement. Prefer concrete events—guidance changes, earnings announcements, regulatory action, deals and product announcements—over an undifferentiated positive/negative headline score. Whether an event was expected and how the stock already reacted may matter more than its tone. Those are proposed inputs, not implemented predictive features.

## What is implemented

`scripts/collect_public_news.py` retrieves public RSS/Atom feeds, stores raw snapshots and SHA-256 hashes, and maintains a local headline ledger. Publication, update and first-observed timestamps are distinct. Headlines first collected today cannot enter yesterday's predictions even if their publication dates are older. Missing or future publication timestamps are rejected; headline revisions are separate versions; repeats preserve their first observation. Use one collector process at a time.

The first collection on September 20, 2026 contained 50 accepted article versions: 20 Apple, 10 Microsoft and 20 NVIDIA. These are a startup backlog, not 50 new stories that day. No news model has been trained, no sentiment classifier is included, and no automatic collection schedule is enabled.

Sources and collection findings:

| Company | Public source | Initial coverage finding |
|---|---|---|
| Apple | [Newsroom feed](https://www.apple.com/newsroom/rss-feed.rss), listed on [Apple's RSS page](https://www.apple.com/ca/rss/) | 20 articles, August 13–September 18, 2026 |
| Microsoft | [Legacy news feed](https://news.microsoft.com/feed/) | 10 articles, February–May 2025: **stale, not usable as current coverage** |
| NVIDIA | [Press-release feed](https://nvidianews.nvidia.com/releases.xml), listed in its [RSS directory](https://nvidianews.nvidia.com/rss) | 20 articles, September 3–17, 2026 |

Microsoft's current Source and corporate-blog feed endpoints returned HTTP 403 when tested. Historical and recent GDELT API probes returned HTTP 429; no historical news dataset was obtained. These are access findings from this run, not proof of permanent unavailability. There are currently only two sources with recent dated content; we cannot claim complete three-company news coverage. A successful HTTP response alone is insufficient.

Every collection manifest reports failures, rejected entries, empty feeds and a stale warning when the newest accepted article is more than 30 days old. That threshold is an operational warning, not proof of feed failure or a predictor. Stale articles remain in the evidence ledger; downstream research must consult manifests, source coverage and article age. A failed or stale source must not become a zero-news feature. The command returns a nonzero status when any source is not healthy, even if other snapshots were saved successfully.

These are company-published feeds, not independent financial journalism. They are useful for event discovery but insufficient for broad sentiment coverage. Raw source material stays local and Git-ignored; the website is unchanged.

## Experiment order

1. **Completed: test a small, target-specific price feature group.** The [intraday study](intraday_study.md) added completed-session open-to-close return, its five-session mean, and trailing 20-session intraday up frequency. With all other settings fixed, historical accuracy fell from 52.71% to 52.63% for logistic regression and from 53.67% to 53.03% for XGBoost; always-up was 53.79%. Probability errors also increased. All six controls reproduced v8's full-history results. This group is not promoted; changing its windows repeatedly on the same results would be further exploration, not confirmation.
2. **Continue timestamped news collection and repair coverage.** Seek a working public Microsoft source and independently reported company events. Define company relevance, duplicate handling and a fixed after-close cutoff before creating features. The existing protocol uses completed-session data; overnight news requires a separately defined premarket experiment.
3. **Compare three fixed variants on the same dates:** price-only control; price plus news event counts/types and novelty; then the same features plus a frozen sentiment extractor. Use each article's availability and publication age, take only its latest version available at that cutoff, and distinguish coverage gaps from no news. A small initial headline snapshot is insufficient for training.
4. **Keep development separate from future evaluation.** Develop the price variant only within the already-used historical period, select on chronological validation, and freeze the next comparison before collecting its future outcomes. Preserve the existing frozen confirmation models and six prospective records. Report accuracy, both constant baselines, balanced accuracy, down/up recall, probability quality, per-company results and costs where trading is evaluated. Do not choose winners from repeated inspection of future outcomes.

Start with one bounded feature-group comparison rather than a large feature or hyperparameter search. A larger reported accuracy obtained by skipping difficult days must also show coverage and the same-day baseline; it cannot replace all-day accuracy. Weather has no defined mechanism for these three technology companies in this study, so it is a lower-priority hypothesis.

## Reproduce locally

Run from the project root with the existing saved v8 predictions and processed history:

```sh
PYTHONPATH=src .venv/bin/python scripts/audit_direction_baselines.py
PYTHONPATH=src .venv/bin/python scripts/collect_public_news.py
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantlab-matplotlib \
  .venv/bin/python -m unittest discover -s tests -v
```

The collector currently reports Microsoft's stale coverage and therefore exits with status 1; this is intentional. Do not treat that as loss of the successful Apple/NVIDIA snapshots. Source manifests and the article ledger are under `data/news_v9/`. No keys or subscriptions are required. Collection must be run again to accumulate future evidence; nothing runs automatically after this task.

The diagnostic report is ready to share as research. A news-performance website claim must wait until news features have actually been evaluated on a frozen, later period. Neither collection success nor software test success demonstrates forecasting accuracy.
