import importlib.util
from pathlib import Path
import unittest
import pandas as pd

spec = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "scripts/audit_direction_baselines.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BaselineTests(unittest.TestCase):
    def test_rolling_baseline_cannot_see_future_prices_or_targets(self):
        dates = pd.date_range("2026-01-01", periods=30)
        history = pd.DataFrame({"date": dates, "open": 100., "close": 101., "target": 0})
        forecasts = pd.DataFrame({"date": [dates[19]], "training_prevalence": [.6]})
        first = audit.attach_baselines(forecasts, history)
        history.loc[20:, "close"] = 1.
        history["target"] = 1
        second = audit.attach_baselines(forecasts, history)
        self.assertEqual(first.recent_majority.iloc[0], 1)
        pd.testing.assert_frame_equal(first, second)

    def test_incomplete_history_fails_instead_of_guessing(self):
        history = pd.DataFrame({"date": ["2026-01-01"], "open": [100], "close": [101]})
        frame = pd.DataFrame({"date": ["2026-01-01"], "training_prevalence": [.6]})
        with self.assertRaisesRegex(ValueError, "Missing rolling history"):
            audit.attach_baselines(frame, history)
