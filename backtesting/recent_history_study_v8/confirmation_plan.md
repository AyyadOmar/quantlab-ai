# Frozen recent-history confirmation plan

Frozen at 2026-09-20T06:30:59.794002+00:00, before fetching any new prices in this experiment.
Only outcomes after 2026-05-22 may enter confirmation. Frozen models are not refitted during confirmation.

Each family is reported separately against always-up and its frozen full-history control. The recent-history choice uses only earlier validation accuracy, then fewer down calls, Brier, log loss, then full/four/two-year order. A fallback match is not evidence of skill.

This is a previously unscored retrospective confirmation period, not a prediction recorded before those historical events. Results require continued prospective validation.

| Company | Model | Selected window | Training through | Validation through |
|---|---|---|---|---|
| AAPL | logistic_regression | two_years | 2025-02-21 00:00:00 | 2026-05-22 00:00:00 |
| AAPL | xgboost | full | 2025-02-21 00:00:00 | 2026-05-22 00:00:00 |
| MSFT | logistic_regression | two_years | 2025-02-21 00:00:00 | 2026-05-22 00:00:00 |
| MSFT | xgboost | four_years | 2025-02-21 00:00:00 | 2026-05-22 00:00:00 |
| NVDA | logistic_regression | two_years | 2025-02-21 00:00:00 | 2026-05-22 00:00:00 |
| NVDA | xgboost | full | 2025-02-21 00:00:00 | 2026-05-22 00:00:00 |
