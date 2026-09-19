# Public-source earnings study

**Exploratory results, not historically vintage-verified evidence.** EPS estimates, actuals and surprise percentages were downloaded as current Yahoo snapshots. They may contain revisions or adjustments. No defaults or demo results were changed.

Surprises are available no earlier than the calendar day after the reported date. Upcoming-call features are derived only from dated NVIDIA newsroom announcements, available from the day after publication. A reported earnings date is never used to reconstruct advance notice.

## Surprise snapshots: all three companies

| Model | Inputs | Mean accuracy | Mean AUC | Mean Brier ↓ | Mean ticker net return |
|---|---|---:|---:|---:|---:|
| logistic_regression | price_only | 51.71% | 0.494 | 0.2520 | 15.84% |
| logistic_regression | surprise_snapshot | 51.79% | 0.494 | 0.2520 | 11.56% |
| xgboost | price_only | 52.27% | 0.506 | 0.2516 | -0.06% |
| xgboost | surprise_snapshot | 52.79% | 0.505 | 0.2518 | -16.75% |

Mean ticker returns are cumulative and are not a portfolio simulation. Same 5 bps fee and 2 bps slippage per side as earlier studies.

## NVIDIA: dated schedule and combined experiments

| Model | Inputs | Accuracy | Brier ↓ | Net return | Trades |
|---|---|---:|---:|---:|---:|
| logistic_regression | price_only | 51.44% | 0.2509 | 60.83% | 35 |
| logistic_regression | surprise_snapshot | 51.79% | 0.2510 | 56.95% | 44 |
| logistic_regression | dated_schedule | 51.44% | 0.2509 | 60.83% | 35 |
| logistic_regression | snapshot_and_schedule | 51.79% | 0.2510 | 56.95% | 44 |
| xgboost | price_only | 50.72% | 0.2547 | -1.69% | 443 |
| xgboost | surprise_snapshot | 51.56% | 0.2549 | -10.05% | 382 |
| xgboost | dated_schedule | 50.72% | 0.2547 | -1.69% | 443 |
| xgboost | snapshot_and_schedule | 50.72% | 0.2545 | -15.00% | 436 |

Schedule-only selection is separate from the snapshot experiments and has no surprise columns. Schedule dates are earnings-call dates, not exact earnings-release timestamps. Public archive coverage is incomplete. Apple/Microsoft schedule models were not evaluated.

## Individual surprise results

| Ticker | Model | Price accuracy | Surprise accuracy | Always up | Price Brier | Surprise Brier | Training-prior Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| AAPL | logistic_regression | 52.75% | 52.87% | 55.38% | 0.2499 | 0.2500 | 0.2475 |
| AAPL | xgboost | 53.83% | 54.19% | 55.38% | 0.2474 | 0.2476 | 0.2475 |
| MSFT | logistic_regression | 50.96% | 50.72% | 52.15% | 0.2551 | 0.2549 | 0.2497 |
| MSFT | xgboost | 52.27% | 52.63% | 52.15% | 0.2527 | 0.2528 | 0.2497 |
| NVDA | logistic_regression | 51.44% | 51.79% | 53.83% | 0.2509 | 0.2510 | 0.2494 |
| NVDA | xgboost | 50.72% | 51.56% | 53.83% | 0.2547 | 0.2549 | 0.2494 |

## Coverage

- AAPL: 38 complete estimate/actual/surprise rows from 2017 through the study cutoff; surprise history available on 100.0% of feature dates.
- MSFT: 38 complete estimate/actual/surprise rows from 2017 through the study cutoff; surprise history available on 100.0% of feature dates.
- NVDA: 38 complete estimate/actual/surprise rows from 2017 through the study cutoff; surprise history available on 100.0% of feature dates.
- NVIDIA: 34 advance notices; scheduled future call known on 21.2% of feature dates. Missing matches: 2017-08-10, 2018-02-08, 2019-02-14, 2022-08-24.

## Interpretation limits

Candidates and thresholds were selected using earlier validation only. Price-only controls reproduce the prior feature study on the same dates. This does not eliminate repeated-research bias, current-snapshot revisions, schedule-archive gaps, or the small number of independent quarterly events. Daily rows repeatedly reuse quarterly values.

Any improvement in this report needs confirmation on fresh observations recorded when received. The code rejects unverified surprise snapshots by default; this run explicitly opted into retrospective exploration. No paid data was used.

Sources: [Yahoo public earnings tables](https://finance.yahoo.com/calendar/earnings/), [NVIDIA newsroom archive](https://nvidianews.nvidia.com/news). Per-event URLs and source hashes are saved with the downloaded data. Full validation scores and timing assumptions are in the JSON report.
