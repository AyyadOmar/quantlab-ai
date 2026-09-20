# Research protocol: open-to-close v2

## Prediction and execution

A row dated session T uses that session's completed OHLCV and market context. Its label is whether the **next session's close exceeds its open**. The strategy is long or cash, enters at the next open, and exits that same session at the close. Overnight returns are not credited to the model. Unresolved final rows are excluded from training but retained for inference. Downloads exclude today's session until 16:15 America/New_York; this conservative rule is specific to US equities.

Direction classification uses probability 0.5. A separate validation-selected probability threshold controls trading; a classification prediction and a trade signal can therefore differ.

## Chronological evaluation

The first test starts at 60% of the feature history. Immediately before it, 15% of the full history is reserved for validation. Earlier rows train the model. One signal-session gap separates training from validation and another separates validation from testing, so every earlier label outcome predates the following window's first signal date.

Test windows advance by 10% of the full history and include the final partial window. Within each fold:

1. Fit the model and preprocessing only on training rows.
2. Score the later validation window.
3. Select the trading threshold from the fixed candidate list using validation Sharpe, then return, then drawdown. Exact ties favor the higher threshold.
4. Freeze that model and threshold and score the later test window.

There is no parameter search on test data. Earlier test observations may enter training or validation in a later fold, once their outcomes are historical. Cross-validation means weight folds by row count so the short final window does not dominate. Aggregate probability and classification metrics also use all test predictions directly.

The persisted research model is the last evaluated fold's model, saved with its own threshold and training cutoff. It is not refitted on test outcomes. `predict-latest` fits a separate model with a trailing historical validation window, selects a threshold there, and scores the latest complete session. Its outcomes resolve against next-session open-to-close returns. V2 uses a separate database, so old close-to-close predictions are not mixed in.

These are historical research results. The period has already appeared in earlier project research; changing the protocol does not turn it into a genuinely untouched prospective sample. Subsequent feature and model selection must use development data and reserve new dates for final confirmation.

## Trading assumptions

Defaults are **5 basis points commission and 2 basis points adverse slippage per side** (roughly 14 basis points per round trip). These are configurable assumptions, not measured broker fills. A basis point is 0.01%.

For a selected trade:

```
net_growth = (exit_close / entry_open) * (1 - slippage) * (1 - fee)
             / ((1 + slippage) * (1 + fee))
```

Every active day is a complete round trip, even when consecutive signals are long. Cash earns zero; positions use the available strategy capital with no leverage. Drawdown starts from initial equity 1, so the first loss is included. Sharpe uses 252 sessions and a default zero risk-free rate.

Benchmarks share evaluation dates:

- **Always-up classifier:** predicts up on every session. Its 100% confidence is deliberately naive; the training-prevalence baseline is the meaningful probability comparison.
- **Training prevalence:** constant up probability estimated only from each fold's training labels.
- **Always-long intraday:** enters every open and exits every close, with both-side costs.
- **Momentum:** trades the next session when the signal session's close-to-close return was positive. It uses the original feature history, not a percentage change across truncated test windows.
- **Buy-and-hold:** one initial entry and one final exit; includes overnight exposure. Adjusted close and corresponding adjusted open approximate total returns with distributions reinvested. Costs apply only at the initial and final transactions. This is a different exposure profile from intraday trading.

Accuracy is supplemented with balanced accuracy, ROC-AUC, Brier score, log loss, net returns, drawdown, Sharpe, active-session fraction and round-trip/order counts. Lower Brier and log loss are better. AUC is reported as 0.5 when a window has only one class, where ranking performance cannot be estimated. Very few trades do not establish a reliable strategy.

## Reproduce

From the project root, using its existing environment:

```sh
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -m quantlab_ai.cli baseline \
  --cached --start 2018-01-01 --end 2026-05-25

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The cached end date is exclusive. This snapshot ends on May 22, 2026; offline mode never downloads newer prices. `run`, `run-batch`, and `baseline` accept `--cached`, `--fee-bps`, and `--slippage-bps`. Cost assumptions must be set before interpreting results; changing costs and choosing the best retrospective outcome is another form of selection.

On this Mac, XGBoost's OpenMP dependency is bundled with scikit-learn but not installed at the system location. The successful local run used this additional environment variable:

```sh
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs"
```

Set it on the same command (or export it in the shell) before running XGBoost. `MPLCONFIGDIR=/tmp/quantlab-matplotlib` can also keep plotting caches outside the user configuration directory.

## Outputs and provenance

New outputs live under `backtesting/open_to_close_v2/`, `models/open_to_close_v2/`, `visualizations/open_to_close_v2/`, and `data/processed/open_to_close_v2/`.

Start with `backtesting/open_to_close_v2/baseline_comparison.md`. Each experiment also records the data hashes, settings, package versions, exact split windows, validation threshold scores, and all test predictions. Daily CSVs include execution prices, signals and returns for auditing.

Existing `public/demo` and `docs/demo` snapshots are archived close-to-close results. The website now uses the separate `public/research` snapshot to display current open-to-close research. Do not compare the archived metrics directly with v2: labels, execution, costs, selection and benchmark accounting all changed.
