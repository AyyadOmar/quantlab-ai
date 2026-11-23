# Guidance feature-layout diagnostic

Post-hoc follow-up to the [guidance study](guidance_study.md). Microsoft had no extracted guidance content, yet adding six all-zero columns changed its XGBoost predictions. XGBoost samples 80% of columns per tree: column count and layout can change the sampled price inputs even when the added columns contain no information.

This diagnostic preserves the 21-column content-model layout and release timing, but zeros all six guidance-content columns. It uses the same historical dates, settings and validation-only selection. The plan was recorded before fitting this diagnostic, after inspecting the original study. It is not a new model candidate or fresh confirmation.

| XGBoost inputs | Accuracy | Brier |
|---|---:|---:|
| price_control | 53.67% | 0.2510 |
| release_timing | 52.87% | 0.2513 |
| guidance_content | 53.87% | 0.2512 |
| zero_content_placebo | 53.87% | 0.2512 |

Same-date always-up: 53.79%. Real content minus zero-content placebo: +0.00 percentage points; descriptive 95% block-bootstrap interval [+0.00, +0.00]. Intervals do not adjust for repeated research.

The two variants differ on **0 of 2,508 direction calls**. A zero-width accuracy interval here reflects identical historical decisions, not certainty about future performance. Some NVIDIA probabilities changed without changing a direction call.

Microsoft's zero-content placebo reproduces all actual-content predictions and native-precision probabilities. Its score change therefore cannot be attributed to guidance information. The original aggregate lead is not evidence that news helped.

| Company | Inputs | Accuracy |
|---|---|---:|
| AAPL | price_control | 55.38% |
| AAPL | release_timing | 55.02% |
| AAPL | guidance_content | 55.14% |
| AAPL | zero_content_placebo | 55.14% |
| MSFT | price_control | 51.91% |
| MSFT | release_timing | 50.72% |
| MSFT | guidance_content | 52.75% |
| MSFT | zero_content_placebo | 52.75% |
| NVDA | price_control | 53.71% |
| NVDA | release_timing | 52.87% |
| NVDA | guidance_content | 53.71% |
| NVDA | zero_content_placebo | 53.71% |

## Reproduce

```sh
PYTHONPATH=src DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs" \
  .venv/bin/python scripts/check_guidance_placebo.py
```

Requires the local ignored guidance dataset, v8 features and completed v11 study. No newer outcomes, website updates or default-model changes are included. Future feature studies should retain this placebo control or disable column sampling consistently across every compared variant before evaluating fresh dates.
