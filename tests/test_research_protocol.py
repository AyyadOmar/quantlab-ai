import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.features.builder import FeatureBuilder, FEATURE_COLUMNS
from quantlab_ai.backtesting.engine import BacktestEngine
from quantlab_ai.models.classical import ClassicalModelTrainer
from quantlab_ai.models.evaluator import evaluate_classifier


def prices(n=200):
    rng = np.random.default_rng(8)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = close * np.exp(rng.normal(0, 0.005, n))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "ticker": "TEST",
                         "open": opens, "high": np.maximum(opens, close) * 1.01,
                         "low": np.minimum(opens, close) * 0.99, "close": close,
                         "adj_close": close, "volume": rng.integers(1000, 5000, n)})


def predictions():
    return pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                         "execution_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                         "entry_open": [100., 120., 80.], "exit_close": [110., 100., 88.],
                         "next_day_return": [.1, -1/6, .1], "benchmark_entry": [100., 120., 80.],
                         "benchmark_exit": [110., 100., 88.], "return_1d": [-.1, .1, .1],
                         "prob_up": [.8, .2, .8], "signal": [1, 0, 1]})


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(project_root=Path(self.temp.name))
        self.settings.ensure_directories()

    def test_target_and_features_use_correct_sessions(self):
        raw = prices()
        builder = FeatureBuilder(self.settings)
        data = builder.build(raw, "TEST")
        row = data.iloc[60]
        source_index = raw.index[raw.date.eq(row.date)][0]
        self.assertAlmostEqual(row.next_day_return, raw.close.iloc[source_index+1] / raw.open.iloc[source_index+1] - 1)
        self.assertEqual(row.target, int(raw.close.iloc[source_index+1] > raw.open.iloc[source_index+1]))
        self.assertEqual(row.execution_date, raw.date.iloc[source_index+1])
        changed = raw.copy()
        changed.loc[source_index+1:, ["open", "high", "low", "close", "adj_close"]] *= 2
        future_changed = builder.build(changed, "TEST")
        np.testing.assert_allclose(data.loc[data.date.le(row.date), FEATURE_COLUMNS],
                                   future_changed.loc[future_changed.date.le(row.date), FEATURE_COLUMNS])
        self.assertEqual(builder.build_for_inference(raw).date.iloc[-1], raw.date.iloc[-1])
        self.assertEqual(data.date.iloc[-1], raw.date.iloc[-2])

    def test_splits_purge_label_boundaries_and_include_tail(self):
        data = FeatureBuilder(self.settings).build(prices(303), "TEST")
        trainer = ClassicalModelTrainer(self.settings, "logistic_regression")
        splits = trainer.walk_forward_splits(data)
        tested = []
        for train, validation, test in splits:
            self.assertLess(train.execution_date.max(), validation.date.min())
            self.assertLess(validation.execution_date.max(), test.date.min())
            tested.extend(test.index)
        self.assertEqual(tested, list(range(int(len(data)*.6), len(data))))
        with self.assertRaises(ValueError):
            trainer.walk_forward_splits(data.head(50))

    def test_final_test_outcomes_cannot_change_its_threshold_or_probabilities(self):
        data = FeatureBuilder(self.settings).build(prices(200), "TEST")
        trainer = ClassicalModelTrainer(self.settings, "logistic_regression")
        before, _, report_before = trainer.evaluate_walk_forward(data)
        changed = data.copy()
        test_indices = trainer.walk_forward_splits(data)[0][2].index
        changed.loc[test_indices, "target"] = 1 - changed.loc[test_indices, "target"]
        changed.loc[test_indices, "exit_close"] *= 2
        changed.loc[test_indices, "next_day_return"] = changed.loc[test_indices, "exit_close"] / changed.loc[test_indices, "entry_open"] - 1
        after, _, report_after = trainer.evaluate_walk_forward(changed)
        self.assertEqual(report_before["folds"][0], report_after["folds"][0])
        np.testing.assert_array_equal(before.loc[before.fold.eq(1), "prob_up"], after.loc[after.fold.eq(1), "prob_up"])
        np.testing.assert_array_equal(before.loc[before.fold.eq(1), "signal"], after.loc[after.fold.eq(1), "signal"])

    def test_round_trip_costs_cash_and_passive_benchmark(self):
        frame = predictions()
        result = BacktestEngine(self.settings).run_with_threshold(frame, "test", "TEST", None, False)
        entry = (1+.0002)*(1+.0005)
        exit_ = (1-.0002)*(1-.0005)
        expected = (1.1*exit_/entry)**2 - 1
        self.assertAlmostEqual(result.metrics["total_return"], expected)
        self.assertAlmostEqual(result.benchmark_metrics["buy_and_hold"]["total_return"], .88*exit_/entry-1)
        self.assertEqual(result.metrics["trade_count"], 2)
        self.assertEqual(result.metrics["order_count"], 4)
        self.assertEqual(result.benchmark_metrics["buy_and_hold"]["order_count"], 2)
        self.assertAlmostEqual(result.equity_curve.equity_curve.iloc[0], result.equity_curve.equity_curve.iloc[1])

    def test_first_loss_counts_toward_drawdown_and_one_day_passive_costs(self):
        frame = predictions().iloc[[1]].copy()
        frame["signal"] = 1
        result = BacktestEngine(self.settings).run_with_threshold(frame, "test", "TEST", None, False)
        self.assertLess(result.metrics["max_drawdown"], 0)
        self.assertAlmostEqual(result.metrics["max_drawdown"], result.metrics["total_return"])
        self.assertAlmostEqual(result.metrics["total_return"], result.benchmark_metrics["buy_and_hold"]["total_return"])

    def test_test_signals_are_preserved_and_missing_signals_rejected(self):
        frame = predictions()
        frame["signal"] = 0
        engine = BacktestEngine(self.settings)
        result = engine.run_with_threshold(frame, "test", "TEST", None, False)
        self.assertEqual(result.metrics["total_return"], 0)
        self.assertEqual(result.metrics["trade_count"], 0)
        with self.assertRaises(ValueError):
            engine.run_with_threshold(frame.drop(columns="signal"), "test", "TEST", None, False)

    def test_probability_metrics_and_fixed_confusion_shape(self):
        good = evaluate_classifier(np.array([0, 1]), np.array([0, 1]), np.array([.1, .9]))
        bad = evaluate_classifier(np.array([0, 1]), np.array([0, 1]), np.array([.4, .6]))
        self.assertLess(good.brier_score, bad.brier_score)
        self.assertLess(good.log_loss, bad.log_loss)
        single = evaluate_classifier(np.array([1, 1]), np.array([1, 1]), np.array([.7, .7]))
        self.assertEqual(single.confusion_matrix, [[0, 0], [0, 2]])

    def test_duplicate_price_sessions_are_rejected(self):
        raw = prices()
        with self.assertRaises(ValueError):
            FeatureBuilder(self.settings).build(pd.concat([raw, raw.iloc[[0]]]), "TEST")

    def test_lstm_scores_every_test_session_with_prior_history(self):
        from quantlab_ai.models.lstm import LSTMTrainer
        self.settings.lstm_epochs = 1
        data = FeatureBuilder(self.settings).build(prices(180), "TEST")
        trainer = LSTMTrainer(self.settings)
        train, validation, test = trainer.walk_forward_splits(data)[0]
        fitted = trainer.fit_frame(train)
        probabilities = trainer.predict_frame(fitted, test, data)
        self.assertEqual(len(probabilities), len(test))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_saved_model_matches_last_evaluated_fold(self):
        import joblib
        data = FeatureBuilder(self.settings).build(prices(180), "TEST")
        trainer = ClassicalModelTrainer(self.settings, "logistic_regression")
        result = trainer.train(data)
        saved = joblib.load(result.artifact_path)
        last_test = trainer.walk_forward_splits(data)[-1][2]
        np.testing.assert_allclose(saved["model"].predict_proba(last_test[FEATURE_COLUMNS])[:, 1],
                                   result.predictions.loc[result.predictions.fold.eq(result.predictions.fold.max()), "prob_up"])
        self.assertEqual(saved["threshold"], result.threshold_report["latest_validation_threshold"])
        self.assertEqual(saved["protocol"], "open_to_close_v2")

    def test_latest_signal_uses_its_validation_threshold(self):
        import joblib
        builder = FeatureBuilder(self.settings)
        raw = prices(180)
        trainer = ClassicalModelTrainer(self.settings, "logistic_regression")
        result = trainer.predict_latest(builder.build(raw, "TEST"), builder.build_for_inference(raw))
        saved = joblib.load(result.artifact_path)
        self.assertEqual(result.signal, int(result.prob_up >= saved["threshold"]))
        self.assertEqual(result.as_of_date, str(raw.date.iloc[-1]))

    def test_incomplete_market_session_is_excluded(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from quantlab_ai.data.loader import MarketDataLoader
        raw = prices(3)
        today = raw.date.iloc[-1]
        before = datetime(today.year, today.month, today.day, 15, 59, tzinfo=ZoneInfo("America/New_York"))
        after = before.replace(hour=16, minute=15)
        self.assertEqual(len(MarketDataLoader.completed_sessions(raw, before)), 2)
        self.assertEqual(len(MarketDataLoader.completed_sessions(raw, after)), 3)

    def test_resolver_uses_next_open_instead_of_previous_close(self):
        from unittest.mock import patch
        from quantlab_ai.pipeline import PipelineRunner
        runner = PipelineRunner(self.settings)
        runner.repository.log_live_prediction("TEST", "logistic_regression", "2020-01-01", 0, .2, 0, "test")
        raw = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                            "open": [90., 120.], "close": [100., 110.]})
        with patch.object(runner.loader, "download", return_value=raw):
            result = runner.resolve_live_predictions()
        self.assertEqual(result["resolved"], 1)
        self.assertEqual(result["summary"]["models"][0]["live_accuracy"], 1.0)

    def test_return_execution_mismatch_is_rejected(self):
        frame = predictions()
        frame["next_day_return"] = 1
        with self.assertRaises(ValueError):
            BacktestEngine(self.settings).run_with_threshold(frame, "test", "TEST", None, False)


if __name__ == "__main__":
    unittest.main()
