# Market and earnings-context study

The added inputs are technology-sector performance (XLK), market volatility (VIX), long/short yield proxies (TNX/IRX), and recency of earnings-related SEC filings for AAPL, MSFT and NVDA. ETFs have no company earnings event features.

Market data uses only previous available sessions. Earnings features become usable the calendar day after filing. Filing dates can lag actual earnings announcements; this conservative first version contains no upcoming earnings schedule, actual EPS, or analyst surprise data.

Each comparison selects its model and trading threshold using earlier validation only, on exactly the same dates and costs as the preceding feature study. Every context selection can retain a price-only candidate. These remain retrospective experiments on previously examined dates; defaults and public demo results are unchanged.

## Mean results across five tickers

| Model | Available inputs | Accuracy | AUC | Brier ↓ | Log loss ↓ | Mean ticker net return |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | price_only | 52.11% | 0.488 | 0.2514 | 0.6961 | 10.04% |
| logistic_regression | with_market | 51.24% | 0.483 | 0.2516 | 0.6965 | -2.13% |
| logistic_regression | with_earnings | 52.03% | 0.488 | 0.2514 | 0.6961 | 9.69% |
| logistic_regression | with_both | 51.20% | 0.483 | 0.2516 | 0.6965 | -1.28% |
| xgboost | price_only | 52.08% | 0.497 | 0.2510 | 0.6953 | -0.13% |
| xgboost | with_market | 52.32% | 0.503 | 0.2508 | 0.6949 | -5.76% |
| xgboost | with_earnings | 51.99% | 0.497 | 0.2511 | 0.6955 | -2.04% |
| xgboost | with_both | 52.25% | 0.502 | 0.2507 | 0.6947 | -2.10% |

Mean ticker returns are not a simulated portfolio. Both-side fees and slippage total approximately 14 bps per round trip. The earnings comparisons contain three companies and two ETFs; company-only results follow.

## Earnings context: companies only

| Model | Available inputs | Accuracy | Brier ↓ |
|---|---|---:|---:|
| logistic_regression | price_only | 51.71% | 0.2520 |
| logistic_regression | with_market | 51.67% | 0.2515 |
| logistic_regression | with_earnings | 51.59% | 0.2520 |
| logistic_regression | with_both | 51.59% | 0.2515 |
| xgboost | price_only | 52.27% | 0.2516 |
| xgboost | with_market | 52.35% | 0.2515 |
| xgboost | with_earnings | 52.11% | 0.2517 |
| xgboost | with_both | 52.23% | 0.2514 |

## Combined context versus price-only selection

| Ticker | Model | Price accuracy | Context accuracy | Always up | Price Brier | Context Brier | Prior Brier | Context net return | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | logistic_regression | 52.75% | 52.87% | 55.38% | 0.2499 | 0.2510 | 0.2475 | -21.48% | 220 |
| AAPL | xgboost | 53.83% | 53.95% | 55.38% | 0.2474 | 0.2478 | 0.2475 | 4.05% | 89 |
| MSFT | logistic_regression | 50.96% | 50.72% | 52.15% | 0.2551 | 0.2547 | 0.2497 | -12.29% | 44 |
| MSFT | xgboost | 52.27% | 51.79% | 52.15% | 0.2527 | 0.2535 | 0.2497 | -20.45% | 86 |
| NVDA | logistic_regression | 51.44% | 51.20% | 53.83% | 0.2509 | 0.2488 | 0.2494 | 21.91% | 150 |
| NVDA | xgboost | 50.72% | 50.96% | 53.83% | 0.2547 | 0.2527 | 0.2494 | 11.11% | 417 |
| SPY | logistic_regression | 54.43% | 51.08% | 54.78% | 0.2490 | 0.2506 | 0.2478 | -0.32% | 1 |
| SPY | xgboost | 52.15% | 51.67% | 54.78% | 0.2505 | 0.2497 | 0.2478 | -3.87% | 14 |
| QQQ | logistic_regression | 50.96% | 50.12% | 55.38% | 0.2521 | 0.2530 | 0.2473 | 5.76% | 11 |
| QQQ | xgboost | 51.44% | 52.87% | 55.38% | 0.2498 | 0.2499 | 0.2473 | -1.36% | 6 |

## Paired probability-error uncertainty

Context minus price-only Brier; negative is better. Descriptive 95% intervals from 20-session block resampling after averaging same-date differences across tickers. These are not corrected for repeated research or multiple comparisons.

- logistic_regression/with_market: 0.0002, interval [-0.0006, 0.0010].
- logistic_regression/with_earnings: 0.0000, interval [-0.0001, 0.0001].
- logistic_regression/with_both: 0.0002, interval [-0.0005, 0.0011].
- xgboost/with_market: -0.0002, interval [-0.0010, 0.0005].
- xgboost/with_earnings: 0.0001, interval [-0.0001, 0.0002].
- xgboost/with_both: -0.0003, interval [-0.0011, 0.0005].

## Coverage and reproducibility

- AAPL: 2088 feature rows; earnings applicable: True; past filing known on 100.0% of rows; maximum market age 4 calendar days.
- MSFT: 2088 feature rows; earnings applicable: True; past filing known on 100.0% of rows; maximum market age 4 calendar days.
- NVDA: 2088 feature rows; earnings applicable: True; past filing known on 100.0% of rows; maximum market age 4 calendar days.
- SPY: 2089 feature rows; earnings applicable: False; past filing known on 0.0% of rows; maximum market age 4 calendar days.
- QQQ: 2088 feature rows; earnings applicable: False; past filing known on 0.0% of rows; maximum market age 4 calendar days.

Test executions: 2023-01-24 00:00:00 through 2026-05-22 00:00:00; 836 observations per experiment.

The JSON report records validation selections, fixed compact-feature diagnostics, data hashes and cost assumptions. Context files have an integrity-checked manifest. Selected model artifacts store the actual feature list and threshold. Observations with missing or stale market context raise an error rather than silently changing comparison dates.

Public sources: [SEC submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [Form 8-K item 2.02](https://www.sec.gov/files/form8-k.pdf), and Yahoo Finance through yfinance. Yield curve is a quoted-index difference, not an official constant-maturity spread. Current historical market downloads can contain corrections; these are not archived point-in-time market vintages.
