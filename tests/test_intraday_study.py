from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.features.builder import FeatureBuilder
from quantlab_ai.intraday_study import (EXTRA_COLUMNS, add_intraday_features, evaluate_fold,
                                       paired_gain, verify_control)
from quantlab_ai.models.classical import ClassicalModelTrainer
from test_research_protocol import prices


class IntradayStudyTests(unittest.TestCase):
    def test_features_use_completed_signal_day_and_never_future_prices_or_labels(self):
        raw = prices(100)
        history = raw.iloc[20:].copy()
        history["target"] = 1
        original = add_intraday_features(history, raw)
        day = raw.iloc[50]
        row = original.loc[original.date == day.date].iloc[0]
        self.assertAlmostEqual(row.intraday_return, day.close / day.open - 1)
        self.assertAlmostEqual(row.intraday_return_mean_5, (raw.close.iloc[46:51] / raw.open.iloc[46:51] - 1).mean())
        self.assertAlmostEqual(row.intraday_up_share_20, (raw.close.iloc[31:51] > raw.open.iloc[31:51]).mean())
        changed = raw.copy()
        changed.loc[51:, "close"] *= 1.3
        changed_history = changed.iloc[20:].copy().assign(target=0)
        after = add_intraday_features(changed_history, changed)
        pd.testing.assert_frame_equal(original.loc[original.date <= day.date, EXTRA_COLUMNS],
                                      after.loc[after.date <= day.date, EXTRA_COLUMNS])

    def test_inputs_are_price_scale_invariant_and_bad_coverage_fails(self):
        raw = prices(100)
        history = raw.iloc[20:].copy()
        before = add_intraday_features(history, raw)
        scaled = raw.copy()
        scaled[["open", "close"]] *= 10
        after = add_intraday_features(scaled.iloc[20:], scaled)
        np.testing.assert_allclose(before[EXTRA_COLUMNS], after[EXTRA_COLUMNS], atol=1e-14)
        for bad in [raw.iloc[20:], raw.drop(index=50), raw.assign(open=0), pd.concat([raw, raw.iloc[[-1]]])]:
            with self.assertRaises(ValueError):
                add_intraday_features(history, bad)

    def test_future_test_labels_cannot_change_fitted_model_or_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(project_root=Path(folder))
            settings.ensure_directories()
            raw = prices(450)
            frame = add_intraday_features(FeatureBuilder(settings).build(raw, "TEST"), raw)
            trainer = ClassicalModelTrainer(settings, "logistic_regression", "compact", "regularized")
            train, validation, test = trainer.walk_forward_splits(frame)[0]
            original, audit, _ = evaluate_fold(settings, train, validation, test, "logistic_regression", "compact_intraday")
            changed, new_audit, _ = evaluate_fold(settings, train, validation, test.assign(target=1-test.target), "logistic_regression", "compact_intraday")
            self.assertEqual(audit, new_audit)
            np.testing.assert_array_equal(original.prob_up, changed.prob_up)
            np.testing.assert_array_equal(original.prediction, changed.prediction)
            with self.assertRaises(ValueError):
                evaluate_fold(settings, train, validation, validation, "logistic_regression", "compact_intraday")

    def test_paired_comparison_rejects_unmatched_days_and_preserves_zero_gain(self):
        frame = pd.DataFrame({"ticker": ["AAPL", "MSFT"] * 20,
                              "execution_date": np.repeat(pd.date_range("2020-01-01", periods=20), 2),
                              "target": [0, 1] * 20, "prediction": [1, 1] * 20})
        same = paired_gain(frame, frame, repetitions=100)
        self.assertEqual(same["gain"], 0)
        self.assertEqual(same["interval_95"], [0, 0])
        with self.assertRaises(ValueError):
            paired_gain(frame, frame.iloc[1:])
        with self.assertRaises(ValueError):
            paired_gain(frame, frame.assign(target=0))

    def test_control_guard_detects_changed_predictions(self):
        frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "execution_date": [pd.Timestamp("2020-01-02")],
                              "ticker": ["AAPL"], "target": [1], "prediction": [1], "fixed_prediction": [1],
                              "fold": [1], "prob_up": [.6]})
        verify_control(frame.copy(), frame.copy())
        with self.assertRaises(AssertionError):
            verify_control(frame.copy(), frame.assign(prob_up=.61))

    def test_control_comparison_restores_float32_csv_precision(self):
        import io
        frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "execution_date": [pd.Timestamp("2020-01-02")],
                              "ticker": ["AAPL"], "target": [1], "prediction": [1], "fixed_prediction": [1],
                              "fold": [1], "prob_up": np.array([.425271421], dtype=np.float32)})
        restored = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        verify_control(frame, restored)
