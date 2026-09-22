# Direction diagnostic audit

These are post-hoc diagnostics of existing frozen predictions, not new untouched evidence. No model, cutoff, training window, or prospective forecast has changed. Always-down is a fixed diagnostic, not a policy selected for deployment using these outcomes. Rolling-majority uses the last 20 completed intraday outcomes available at each signal close; ties predict up. Training-majority uses saved training prevalence.

## Historical: 2023-01-24 through 2026-05-22

| Comparison | Rows | Accuracy | Balanced accuracy | Down recall | Up recall |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 2508 | 52.43% | 50.44% | 24.07% | 76.80% |
| XGBoost | 2508 | 53.51% | 50.41% | 9.49% | 91.33% |
| Always Up | 2508 | 53.79% | 50.00% | 0.00% | 100.00% |
| Always Down | 2508 | 46.21% | 50.00% | 100.00% | 0.00% |
| Training Majority (LR) | 2508 | 53.15% | 49.99% | 8.28% | 91.70% |
| Recent Majority | 2508 | 50.48% | 48.76% | 26.14% | 71.39% |

### Logistic regression by company and time

| Comparison | Rows | Accuracy | Balanced accuracy | Down recall | Up recall |
|---|---:|---:|---:|---:|---:|
| AAPL | 836 | 55.02% | 50.09% | 4.29% | 95.90% |
| MSFT | 836 | 49.28% | 48.62% | 33.25% | 63.99% |
| NVDA | 836 | 52.99% | 51.62% | 33.68% | 69.56% |
| First half | 1254 | 52.47% | 50.82% | 31.59% | 70.04% |
| Second half | 1254 | 52.39% | 50.20% | 16.72% | 83.68% |

logistic_regression: pooled ROC AUC 0.4952; Brier 0.2520; training-prevalence Brier 0.2489 (lower is better).

xgboost: pooled ROC AUC 0.5118; Brier 0.2515; training-prevalence Brier 0.2489 (lower is better).

## Confirmation: 2026-05-26 through 2026-09-18

| Comparison | Rows | Accuracy | Balanced accuracy | Down recall | Up recall |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 243 | 51.85% | 53.90% | 8.66% | 99.14% |
| XGBoost | 243 | 48.15% | 50.21% | 4.72% | 95.69% |
| Always Up | 243 | 47.74% | 50.00% | 0.00% | 100.00% |
| Always Down | 243 | 52.26% | 50.00% | 100.00% | 0.00% |
| Training Majority (LR) | 243 | 47.74% | 50.00% | 0.00% | 100.00% |
| Recent Majority | 243 | 48.15% | 48.68% | 37.01% | 60.34% |

### Logistic regression by company and time

| Comparison | Rows | Accuracy | Balanced accuracy | Down recall | Up recall |
|---|---:|---:|---:|---:|---:|
| AAPL | 81 | 53.09% | 50.00% | 0.00% | 100.00% |
| MSFT | 81 | 46.91% | 50.00% | 0.00% | 100.00% |
| NVDA | 81 | 55.56% | 60.53% | 23.91% | 97.14% |
| First half | 120 | 49.17% | 52.34% | 4.69% | 100.00% |
| Second half | 123 | 54.47% | 55.52% | 12.70% | 98.33% |

logistic_regression: pooled ROC AUC 0.5413; Brier 0.2539; training-prevalence Brier 0.2531 (lower is better).

xgboost: pooled ROC AUC 0.4877; Brier 0.2559; training-prevalence Brier 0.2516 (lower is better).

## Interpretation

The model's absolute accuracy and its margin over always-up answer different questions. A period with fewer up days makes always-up less accurate. The newer result is concentrated in NVIDIA; matching the baseline on other companies is not replication. Balanced accuracy weights up-day and down-day recall equally. These tables diagnose weaknesses; they do not justify choosing a new rule from the scored dates. No trading-return or profitability claim follows from these classification metrics.
