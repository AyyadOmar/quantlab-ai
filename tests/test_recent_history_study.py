import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.features.builder import FeatureBuilder
from quantlab_ai.models.classical import ClassicalModelTrainer
from quantlab_ai.recent_history_study import choose_window, evaluate_recent_fold, window_training_rows
from test_research_protocol import prices


class RecentHistoryTests(unittest.TestCase):
    def test_calendar_windows_are_anchored_to_training_end(self):
        dates = pd.bdate_range("2018-01-01", "2024-02-29")
        frame = pd.DataFrame({"date": dates, "target": np.arange(len(dates)) % 2})
        two = window_training_rows(frame, "two_years")
        four = window_training_rows(frame, "four_years")
        self.assertTrue((two.date > pd.Timestamp("2022-02-28")).all())
        self.assertTrue((four.date > pd.Timestamp("2020-02-29")).all())
        self.assertEqual(two.date.max(), frame.date.max())
        self.assertLess(len(two), len(four))
        pd.testing.assert_frame_equal(window_training_rows(frame, "full"), frame)

    def test_early_four_year_window_uses_available_history_and_rejects_bad_input(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        frame = pd.DataFrame({"date": dates, "target": np.arange(300) % 2})
        pd.testing.assert_frame_equal(window_training_rows(frame, "four_years"), frame)
        for bad in [frame.iloc[:20], frame.iloc[::-1], frame.assign(target=1)]:
            with self.assertRaises(ValueError):
                window_training_rows(bad, "two_years")

    def test_window_choice_uses_validation_accuracy_then_conservative_ties(self):
        scores = [{"window": w, "correct": 60, "down_calls": 0, "brier_score": .25, "log_loss": .7}
                  for w in ["two_years", "four_years", "full"]]
        self.assertEqual(choose_window(scores), "full")
        scores[0]["correct"] = 61
        self.assertEqual(choose_window(scores), "two_years")
        scores[1]["correct"] = 61
        scores[0]["down_calls"] = 20
        self.assertEqual(choose_window(scores), "four_years")
        scores[1]["brier_score"] = np.nan
        with self.assertRaises(ValueError):
            choose_window(scores)

    def test_test_labels_cannot_change_window_cutoff_or_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.ensure_directories()
            history = FeatureBuilder(settings).build(prices(1600), "TEST")
            trainer = ClassicalModelTrainer(settings, "logistic_regression", "compact", "regularized")
            train, validation, test = trainer.walk_forward_splits(history)[-2]
            before, audit_before, _ = evaluate_recent_fold(settings, train, validation, test, "logistic_regression")
            changed = test.copy()
            changed["target"] = 1 - changed.target
            after, audit_after, _ = evaluate_recent_fold(settings, train, validation, changed, "logistic_regression")
            self.assertEqual(audit_before, audit_after)
            for window in before:
                np.testing.assert_array_equal(before[window].prob_up, after[window].prob_up)
                np.testing.assert_array_equal(before[window].prediction, after[window].prediction)
            self.assertLess(audit_before["training_windows"]["two_years"]["rows"],
                            audit_before["training_windows"]["full"]["rows"])
            with self.assertRaises(ValueError):
                evaluate_recent_fold(settings, train, validation, validation, "logistic_regression")


if __name__ == "__main__":
    unittest.main()
