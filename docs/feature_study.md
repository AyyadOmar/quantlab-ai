# Feature and regularization study

The `feature-study` command compares improvements against the corrected open-to-close baseline without changing the historical baseline or public demo. The completed local results are in [the study report](../backtesting/feature_study_v3/feature_study.md).

## Changes

Three fixed feature profiles are available through `ClassicalModelTrainer`:

- `original`: the existing 24 features, unchanged.
- `relative`: 24 features; four raw moving-average levels become `close / average - 1`, and the three MACD values are divided by close. Multiplying all historical prices by a constant therefore does not change these features.
- `compact`: 14 predefined features, retaining one moving-average distance and one MACD component, and removing overlapping momentum/volatility inputs. No test-data feature ranking is used.

Two model profiles are available:

- `baseline`: existing settings.
- `regularized`: logistic regression uses `C=0.01`; XGBoost uses 100 depth-2 trees, learning rate 0.03, minimum child weight 20, L2 penalty 10, L1 penalty 1, and row/column sampling 0.8. The baseline uses 300 depth-5 trees. These settings were fixed before running the study.

Example for a standalone experiment through the Python interface:

```python
trainer = ClassicalModelTrainer(
    settings, "xgboost", feature_set="compact", model_profile="regularized"
)
artifacts = trainer.train(features)
```

Alternative profile artifacts have distinct filenames and retain their actual feature list. Defaults remain unchanged: the study evaluates improvements but does not select a universal production model from retrospective test scores.

## Run the complete comparison

From the project root:

```sh
PYTHONPATH=src \
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
.venv/bin/python -m quantlab_ai.cli feature-study \
  --cached --start 2018-01-01 --end 2026-05-25
```

The library-path setting addresses this Mac's existing OpenMP installation. See [the execution protocol](research_protocol.md) for cost and timing assumptions. The study uses precisely the same labels, dates, split boundaries, fees and slippage as the corrected baseline. It includes the last partial fold. Neither new external data nor revised historical prices were downloaded.

Within each ticker, model family and fold, all six candidates fit on the training window. The selection rule minimizes validation Brier score (probability squared error), then log loss, with deterministic feature-count/name tie-breaks. The selected candidate and its validation-tuned trading threshold are fixed before scoring test rows. The same validation window serves model and threshold selection; final test outcomes serve neither. Each static candidate is also scored for diagnostic comparison. Avoid choosing a different candidate based on those test results.

The output has 60 fixed-candidate experiments and 10 validation-selected comparisons across five tickers and two model families. Runtime includes 300 fits across five folds per ticker/model. The selected-model fit time alone excludes the cost of fitting the alternatives.

## Saved models and audit files

Outputs live in `backtesting/feature_study_v3/`, with selected models in `models/feature_study_v3/`. The `study_plan.json` records the fixed candidates and settings before fitting. `feature_study.json` includes every validation score, threshold sweep, selection, timing, data hash and package version. Individual CSVs contain predictions and equity curves.

A selected artifact is a joblib dictionary containing `model`, `feature_columns`, `candidate`, `threshold`, `trained_through`, `validation_through`, and `protocol`. It is the model selected for the last evaluation fold, without refitting on its test labels. For prediction, build features with `FeatureBuilder.build_for_inference`, select columns in the saved order, and call the saved model's `predict_proba`. The chosen model can differ across tickers and folds.

These files are research artifacts. A prospective deployment should establish a retraining schedule and test on newly arriving data with the selection rule frozen. Today's study reuses a period already examined in earlier research.

## Verification

```sh
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/quantlab-matplotlib \
.venv/bin/python -m unittest discover -s tests -v
```

The suite checks price-unit invariance, absence of future-price leakage, selection independence from test outcomes, feature order in saved models, execution costs and existing split protections. The local run passed 19 tests. All ten original-profile study predictions and signals matched the previous baseline files exactly.
