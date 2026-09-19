# Corrected baseline comparison

Signal after close; next-session open-to-close long/cash trades. Thresholds selected on earlier validation only.

Costs per side: 5 bps fee + 2 bps slippage. Cash earns zero. Fixed model settings; no search on test results. These costs are assumptions, not measured fills.

Historical walk-forward results are a research baseline, not an untouched prospective trial. The dates were used in earlier project research.

Buy-and-hold includes overnight exposure and uses adjusted prices; always-long trades only open-to-close, matching the model's holding window. Returns are cumulative, not annualized.

| Ticker | Model | Accuracy | Always up | AUC | Brier ↓ | Prior Brier ↓ | Net return | Always long | Momentum | Buy & hold | Max drawdown | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | logistic_regression | 49.2% | 55.4% | 0.457 | 0.255 | 0.247 | 11.6% | -15.3% | -2.0% | 123.5% | -10.0% | 16 |
| AAPL | xgboost | 48.0% | 55.4% | 0.524 | 0.307 | 0.247 | 7.6% | -15.3% | -2.0% | 123.5% | -14.6% | 101 |
| MSFT | logistic_regression | 50.2% | 52.2% | 0.475 | 0.259 | 0.250 | -12.6% | -63.1% | -36.6% | 77.3% | -18.3% | 38 |
| MSFT | xgboost | 46.4% | 52.2% | 0.453 | 0.330 | 0.250 | -18.6% | -63.1% | -36.6% | 77.3% | -27.6% | 109 |
| NVDA | logistic_regression | 47.8% | 53.8% | 0.486 | 0.290 | 0.249 | -8.5% | -50.3% | -59.0% | 1043.3% | -16.2% | 41 |
| NVDA | xgboost | 49.3% | 53.8% | 0.486 | 0.296 | 0.249 | -46.3% | -50.3% | -59.0% | 1043.3% | -47.0% | 186 |
| SPY | logistic_regression | 53.9% | 54.8% | 0.488 | 0.250 | 0.248 | 9.0% | -57.2% | -42.7% | 94.7% | -2.9% | 5 |
| SPY | xgboost | 50.0% | 54.8% | 0.509 | 0.292 | 0.248 | 8.0% | -57.2% | -42.7% | 94.7% | -10.8% | 150 |
| QQQ | logistic_regression | 48.9% | 55.4% | 0.501 | 0.254 | 0.247 | -9.6% | -54.1% | -44.7% | 154.2% | -10.0% | 42 |
| QQQ | xgboost | 48.3% | 55.4% | 0.498 | 0.320 | 0.247 | 1.1% | -54.1% | -44.7% | 154.2% | -10.9% | 122 |

Each experiment JSON records exact windows, costs, data hashes and package versions. Daily CSVs contain every signal and execution price. Brier and log loss measure probability quality; lower is better. Very few trades do not establish a reliable strategy.

Execution coverage: 2023-01-24 00:00:00 through 2026-05-22 00:00:00 (836 sessions for the first experiment).
