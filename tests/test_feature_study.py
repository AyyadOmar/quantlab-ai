import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from quantlab_ai.config import Settings
from quantlab_ai.feature_study import Candidate, choose_candidate, evaluate_fold
from quantlab_ai.features.builder import FeatureBuilder, FEATURE_COLUMNS
from quantlab_ai.features.profiles import feature_columns
from quantlab_ai.models.classical import ClassicalModelTrainer
from test_research_protocol import prices


class FeatureStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(project_root=Path(self.temp.name))
        self.settings.ensure_directories()
        self.builder = FeatureBuilder(self.settings)

    def test_relative_profiles_are_invariant_to_price_units(self):
        raw = prices(180)
        scaled = raw.copy()
        scaled[["open", "high", "low", "close", "adj_close"]] *= 10
        original = self.builder.build(raw, "TEST")
        alternative = self.builder.build(scaled, "TEST")
        for profile in ["relative", "compact"]:
            np.testing.assert_allclose(original[feature_columns(profile)], alternative[feature_columns(profile)], atol=1e-10)
        self.assertEqual(feature_columns("original"), FEATURE_COLUMNS)
        self.assertEqual(len(feature_columns("relative")), 24)
        self.assertEqual(len(feature_columns("compact")), 14)

    def test_future_prices_cannot_change_earlier_relative_features(self):
        raw = prices(180)
        modified = raw.copy()
        modified.loc[120:, ["open", "high", "low", "close", "adj_close"]] *= 2
        before = self.builder.build(raw, "TEST")
        after = self.builder.build(modified, "TEST")
        cutoff = raw.date.iloc[119]
        np.testing.assert_allclose(before.loc[before.date.le(cutoff), feature_columns("relative")],
                                   after.loc[after.date.le(cutoff), feature_columns("relative")])

    def test_test_outcomes_do_not_change_selected_candidate_or_signals(self):
        history = self.builder.build(prices(180), "TEST")
        trainer = ClassicalModelTrainer(self.settings, "logistic_regression")
        train, val, test = trainer.walk_forward_splits(history)[0]
        candidates = (Candidate("original", "baseline"), Candidate("compact", "regularized"))
        before, report_before, _ = evaluate_fold(self.settings, history, train, val, test, "logistic_regression", candidates)
        changed = test.copy()
        changed["target"] = 1 - changed.target
        changed["exit_close"] *= 3
        changed["next_day_return"] = changed.exit_close / changed.entry_open - 1
        after, report_after, _ = evaluate_fold(self.settings, history, train, val, changed, "logistic_regression", candidates)
        self.assertEqual(report_before["selected_candidate"], report_after["selected_candidate"])
        self.assertEqual(report_before["validation_scores"], report_after["validation_scores"])
        self.assertEqual(report_before["thresholds"], report_after["thresholds"])
        for name in before:
            np.testing.assert_array_equal(before[name].prob_up, after[name].prob_up)
            np.testing.assert_array_equal(before[name].signal, after[name].signal)

    def test_profile_artifact_keeps_feature_order_and_predictions(self):
        history = self.builder.build(prices(180), "TEST")
        trainer = ClassicalModelTrainer(self.settings, "logistic_regression", "compact", "regularized")
        artifact = trainer.train(history)
        saved = joblib.load(artifact.artifact_path)
        self.assertEqual(saved["feature_columns"], feature_columns("compact"))
        self.assertIn("compact_regularized", artifact.artifact_path)
        test = trainer.walk_forward_splits(history)[-1][2]
        np.testing.assert_allclose(saved["model"].predict_proba(test[saved["feature_columns"]])[:, 1],
                                   artifact.predictions.loc[artifact.predictions.fold.eq(artifact.predictions.fold.max()), "prob_up"])
        self.assertEqual(saved["model"][-1].C, .01)

    def test_selection_uses_probability_quality_and_rejects_invalid_scores(self):
        scores = [{"candidate": "a", "brier_score": .2, "log_loss": .6, "feature_count": 24},
                  {"candidate": "b", "brier_score": .19, "log_loss": .7, "feature_count": 14}]
        self.assertEqual(choose_candidate(scores), "b")
        scores[1]["brier_score"] = float("nan")
        with self.assertRaises(ValueError):
            choose_candidate(scores)
        with self.assertRaises(ValueError):
            feature_columns("unknown")


if __name__ == "__main__":
    unittest.main()
