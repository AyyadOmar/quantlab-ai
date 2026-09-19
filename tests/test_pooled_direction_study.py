import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.features.builder import FeatureBuilder
from quantlab_ai.models.classical import ClassicalModelTrainer
from quantlab_ai.pooled_direction_study import pooled_training_frame, score_fitted_direction
from test_research_protocol import prices


class PooledDirectionStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(project_root=Path(self.temp.name))
        self.settings.ensure_directories()
        self.a = FeatureBuilder(self.settings).build(prices(240).assign(ticker="A"), "A")
        self.b = self.a.copy().assign(ticker="B")
        self.b["target"] = 1 - self.b.target
        self.trainer = ClassicalModelTrainer(self.settings, "logistic_regression", "compact", "regularized")
        self.train, self.val, self.test = self.trainer.walk_forward_splits(self.a)[0]

    def test_pool_uses_same_dates_and_excludes_other_company_future_outcomes(self):
        pooled = pooled_training_frame({"A": self.a, "B": self.b}, self.train.date, self.val.date.min())
        changed = self.b.copy()
        changed.loc[changed.date >= self.val.date.min(), "target"] = 1
        changed.loc[changed.date >= self.val.date.min(), "return_1d"] = 100
        alternative = pooled_training_frame({"A": self.a, "B": changed}, self.train.date, self.val.date.min())
        pd.testing.assert_frame_equal(pooled, alternative)
        self.assertEqual(len(pooled), 2 * len(self.train))
        self.assertLess(pooled.execution_date.max(), self.val.date.min())

    def test_missing_company_sessions_and_overlapping_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            pooled_training_frame({"A": self.a, "B": self.b.iloc[1:]}, self.train.date, self.val.date.min())
        with self.assertRaises(ValueError):
            pooled_training_frame({"A": self.a, "B": self.b}, self.train.date, self.train.execution_date.max())
        with self.assertRaises(ValueError):
            pooled_training_frame({"A": self.a, "B": self.a}, self.train.date, self.val.date.min())

    def test_shared_predictions_and_selection_do_not_use_test_outcomes(self):
        pooled = pooled_training_frame({"A": self.a, "B": self.b}, self.train.date, self.val.date.min())
        fitted = self.trainer.fit_frame(pooled)
        before, selected = score_fitted_direction(self.trainer, fitted, self.a, self.val, self.test)
        changed = self.test.copy()
        changed["target"] = 1 - changed.target
        changed["exit_close"] *= 3
        after, selected_after = score_fitted_direction(self.trainer, fitted, self.a, self.val, changed)
        self.assertEqual(selected, selected_after)
        np.testing.assert_array_equal(before.prob_up, after.prob_up)
        np.testing.assert_array_equal(before.prediction, after.prediction)


if __name__ == "__main__":
    unittest.main()
