import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.direction_study import (
    accuracy_interval, apply_direction_rule, down_call_audit,
    evaluate_direction_fold, select_direction_rule,
)
from quantlab_ai.feature_study import Candidate
from quantlab_ai.features.builder import FeatureBuilder
from quantlab_ai.models.classical import ClassicalModelTrainer
from test_research_protocol import prices


class DirectionStudyTests(unittest.TestCase):
    def test_accuracy_tie_and_losing_cutoffs_fall_back_to_always_up(self):
        labels = np.array([0, 1] * 20 + [1] * 20)
        probabilities = np.array([.2] * 40 + [.8] * 20)
        selected = select_direction_rule(labels, probabilities)["selected"]
        self.assertEqual(selected["rule"], "always_up")
        losing = select_direction_rule(np.ones(60), probabilities)["selected"]
        self.assertEqual(losing["rule"], "always_up")
        np.testing.assert_array_equal(apply_direction_rule([0, .3, 1], losing), [1, 1, 1])

    def test_useful_down_calls_win_and_cutoff_boundary_predicts_up(self):
        labels = np.array([0] * 30 + [1] * 30)
        probabilities = np.array([.4] * 30 + [.8] * 30)
        selected = select_direction_rule(labels, probabilities)["selected"]
        self.assertEqual(selected["cutoff"], .45)
        np.testing.assert_array_equal(apply_direction_rule([.449, .45, .451], selected), [0, 1, 1])

    def test_too_few_validation_down_calls_cannot_win(self):
        labels = np.array([0] * 19 + [1] * 41)
        probabilities = np.array([.1] * 19 + [.9] * 41)
        self.assertEqual(select_direction_rule(labels, probabilities)["selected"]["rule"], "always_up")

    def test_down_audit_reconciles_accuracy_on_all_days(self):
        labels = np.array([0] * 46 + [1] * 54)
        prediction = np.ones(100, dtype=int)
        prediction[:7], prediction[-3:] = 0, 0
        audit = down_call_audit(labels, prediction)
        self.assertEqual(audit["correct_down"], 7)
        self.assertEqual(audit["incorrect_down"], 3)
        self.assertEqual(audit["rows"], 100)
        self.assertAlmostEqual(audit["accuracy"], .58)
        self.assertAlmostEqual(audit["accuracy_gain"], .04)
        self.assertIsNone(down_call_audit(labels, np.ones(100))["down_precision"])

    def test_invalid_inputs_fail_instead_of_silently_dropping_rows(self):
        for labels, probabilities in [([], []), ([0, 1], [.4]), ([2], [.4]),
                                      ([0], [np.nan]), ([0], [1.1])]:
            with self.assertRaises(ValueError):
                select_direction_rule(labels, probabilities)
        with self.assertRaises(ValueError):
            down_call_audit([0, 1], [.2, .8])

    def test_date_block_uncertainty_preserves_cross_asset_dependence(self):
        frame = pd.DataFrame({"ticker": "A", "execution_date": pd.bdate_range("2020-01-01", periods=60),
                              "target": [0, 1, 0] * 20, "prediction": [0, 0, 1] * 20})
        single = accuracy_interval([frame], 42, repetitions=100)
        duplicate_asset = frame.assign(ticker="B")
        together = accuracy_interval([frame, duplicate_asset], 42, repetitions=100)
        self.assertEqual(single, together)
        with self.assertRaises(ValueError):
            accuracy_interval([frame, frame], 42)

    def test_test_outcomes_cannot_change_cutoff_model_or_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.ensure_directories()
            history = FeatureBuilder(settings).build(prices(240), "TEST")
            trainer = ClassicalModelTrainer(settings, "logistic_regression")
            train, validation, test = trainer.walk_forward_splits(history)[0]
            candidates = (Candidate("relative", "regularized"), Candidate("compact", "regularized"))
            before, audit_before, artifact_before = evaluate_direction_fold(
                settings, history, train, validation, test, "logistic_regression", candidates)
            changed = test.copy()
            changed["target"] = 1 - changed.target
            changed["exit_close"] *= 3
            changed["next_day_return"] = changed.exit_close / changed.entry_open - 1
            after, audit_after, artifact_after = evaluate_direction_fold(
                settings, history, train, validation, changed, "logistic_regression", candidates)
            self.assertEqual(audit_before, audit_after)
            self.assertEqual(artifact_before["direction_rule"], artifact_after["direction_rule"])
            np.testing.assert_array_equal(before.prob_up, after.prob_up)
            np.testing.assert_array_equal(before.prediction, after.prediction)
            np.testing.assert_array_equal(before.fixed_prediction, after.fixed_prediction)


if __name__ == "__main__":
    unittest.main()
