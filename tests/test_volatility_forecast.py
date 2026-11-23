import unittest

import numpy as np
import pandas as pd

from quantlab_ai.features.volatility_forecast import (
    BASELINE_COLUMN, VOLATILITY_COLUMNS, add_volatility_features, score_volatility, score_volatility_gain,
)


def daily_prices():
    # The first 20 sessions are flat, then an intraday shock changes the forecast.
    prices = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=26),
                           "open": 100.0, "close": 100.0, "ticker": "TEST"})
    prices.loc[20, "close"] = 110.0
    return prices


class VolatilityForecastTests(unittest.TestCase):
    def test_shock_updates_next_session_forecast_at_signal_close(self):
        prices = daily_prices()
        result = add_volatility_features(prices.iloc[19:], prices)
        np.testing.assert_allclose(result.loc[19, VOLATILITY_COLUMNS].to_numpy(dtype=float), 0)
        after_shock = .10 * np.sqrt(.06)
        self.assertAlmostEqual(result.loc[20, "intraday_vol_forecast"], after_shock)
        self.assertAlmostEqual(result.loc[20, "intraday_vol_delta_1d"], after_shock)
        self.assertAlmostEqual(result.loc[20, BASELINE_COLUMN], .10 / np.sqrt(20))
        self.assertAlmostEqual(result.loc[21, "intraday_vol_delta_1d"], (np.sqrt(.94) - 1) * after_shock)

    def test_future_prices_and_outcome_columns_cannot_affect_earlier_signals(self):
        prices = daily_prices()
        history = prices.iloc[19:].assign(next_day_return=9, target=0)
        original = add_volatility_features(history, prices)
        changed = prices.copy()
        changed.loc[22:, "close"] *= 5
        updated = add_volatility_features(changed.iloc[19:].assign(next_day_return=-.5, target=1), changed)
        truncated = add_volatility_features(history.loc[:21], prices.loc[:21])
        columns = VOLATILITY_COLUMNS + [BASELINE_COLUMN]
        pd.testing.assert_frame_equal(original.loc[:21, columns], updated.loc[:21, columns])
        pd.testing.assert_frame_equal(original.loc[:21, columns], truncated[columns])

    def test_mismatched_sessions_and_prices_are_rejected(self):
        prices = daily_prices()
        history = prices.iloc[19:]
        for invalid in [prices.iloc[2:], prices.assign(ticker="OTHER"), prices.drop(index=21),
                        prices.assign(close=-1), prices.assign(close=prices.close + .1)]:
            with self.subTest(), self.assertRaises(ValueError):
                add_volatility_features(history, invalid)
        with self.assertRaisesRegex(ValueError, "No completed"):
            add_volatility_features(prices.iloc[:1], prices.iloc[1:])

    def test_proxy_scores_size_and_counts_zero_returns(self):
        frame = pd.DataFrame({"next_day_return": [.02, -.02, 0],
                              "intraday_vol_forecast": [.02, .02, 0], BASELINE_COLUMN: [0, 0, 0]})
        scored = score_volatility(frame)
        self.assertEqual(scored["rows"], 3)
        self.assertEqual(scored["ewma_variance_mse"], 0)
        self.assertAlmostEqual(scored["baseline_20_variance_mse"], 32/3)
        with self.assertRaises(ValueError):
            score_volatility(frame.assign(next_day_return=np.nan))

    def test_gain_keeps_companies_together_and_handles_zero_baseline_error(self):
        # The model wins A, baseline wins B. Every session ties once both are included.
        dates = pd.bdate_range("2024-02-01", periods=45)
        paired = pd.DataFrame({"execution_date": np.repeat(dates, 2), "ticker": ["A", "B"] * 45,
                               "next_day_return": .01, "intraday_vol_forecast": [.01, 0] * 45,
                               BASELINE_COLUMN: [0, .01] * 45})
        result = score_volatility_gain(paired.sample(frac=1, random_state=1))
        self.assertEqual(result["error_reduction"], 0)
        self.assertEqual(result["interval_95"], [0, 0])
        self.assertEqual(result["sessions"], 45)
        with self.assertRaisesRegex(ValueError, "one score"):
            score_volatility_gain(pd.concat([paired, paired.iloc[:1]]))
        undefined = score_volatility_gain(paired.assign(**{BASELINE_COLUMN: .01}))
        self.assertIsNone(undefined["error_reduction"])
        self.assertIsNone(undefined["interval_95"])
