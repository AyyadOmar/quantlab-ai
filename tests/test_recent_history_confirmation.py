from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.recent_history_confirmation import (append_forecasts, confirmation_rows,
    prospective_date_is_safe, run_confirmation, stitch_frozen_history)
from test_research_protocol import prices


class ConfirmationTests(unittest.TestCase):
    def test_no_network_access_without_existing_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            with patch('quantlab_ai.recent_history_confirmation.MarketDataLoader.download') as download:
                with self.assertRaises(FileNotFoundError):
                    run_confirmation(settings, '2026-09-21')
                download.assert_not_called()

    def test_frozen_prefix_preserved_and_revised_price_units_rejected(self):
        fresh = prices(120)
        old = fresh.iloc[:100].copy()
        cutoff = old.date.max()
        revised_adjustment = fresh.copy()
        revised_adjustment['adj_close'] *= .99
        joined = stitch_frozen_history(old, revised_adjustment, cutoff)
        pd.testing.assert_frame_equal(joined.iloc[:100], old)
        self.assertEqual(len(joined), len(fresh))
        split = fresh.copy()
        split['close'] /= 2
        with self.assertRaises(ValueError):
            stitch_frozen_history(old, split, cutoff)
        with self.assertRaises(ValueError):
            stitch_frozen_history(old, fresh.iloc[100:], cutoff)
        with self.assertRaises(ValueError):
            stitch_frozen_history(old, old, cutoff)

    def test_first_new_outcome_keeps_the_cutoff_session_signal(self):
        frame = pd.DataFrame({'date':pd.to_datetime(['2026-05-21','2026-05-22','2026-05-26']),
                              'execution_date':pd.to_datetime(['2026-05-22','2026-05-26','2026-05-27'])})
        rows = confirmation_rows(frame, '2026-05-22')
        self.assertEqual(rows.date.tolist(), [pd.Timestamp('2026-05-22'),pd.Timestamp('2026-05-26')])

    def test_ledger_does_not_rewrite_or_duplicate_existing_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ledger.jsonl'
            original={'forecast_id':'same','prediction':0}
            self.assertEqual(append_forecasts(path,[original]),1)
            self.assertEqual(append_forecasts(path,[{'forecast_id':'same','prediction':1}]),0)
            self.assertEqual(json.loads(path.read_text()),original)
            self.assertEqual(append_forecasts(path,[{'forecast_id':'next','prediction':1}]),1)
            self.assertEqual(len(path.read_text().splitlines()),2)

    def test_prospective_guard_does_not_backdate_forecasts(self):
        sunday=datetime(2026,9,20,18,tzinfo=timezone.utc)
        monday=datetime(2026,9,21,18,tzinfo=timezone.utc)
        self.assertTrue(prospective_date_is_safe('2026-09-18',sunday))
        self.assertFalse(prospective_date_is_safe('2026-09-18',monday))
        self.assertFalse(prospective_date_is_safe('2026-09-21',sunday))


if __name__ == '__main__':
    unittest.main()
