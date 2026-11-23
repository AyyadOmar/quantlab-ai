from hashlib import sha256
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.data.guidance import CONTENT_COLUMNS, EVENT_COLUMNS, build_event_features, extract_guidance, load_documents
from quantlab_ai.features.builder import FeatureBuilder
from quantlab_ai.guidance_study import evaluate_fold
from quantlab_ai.models.classical import ClassicalModelTrainer
from test_research_protocol import prices


class GuidanceStudyTests(unittest.TestCase):
    def test_placebo_removes_content_without_changing_prices_timing_or_source(self):
        remove_content = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_guidance_placebo.py"))["remove_content"]
        frame = pd.DataFrame({"return_1d": [.1, -.2], "target": [0, 1],
                              **{column: [1, 0] for column in EVENT_COLUMNS}})
        before = frame.copy(deep=True)
        placebo = remove_content(frame)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(placebo[["return_1d", "target", "earnings_release_recent"]],
                                      frame[["return_1d", "target", "earnings_release_recent"]])
        self.assertEqual(list(placebo.columns), list(frame.columns))
        content = [column for column in EVENT_COLUMNS if column != "earnings_release_recent"]
        self.assertTrue((placebo[content].to_numpy() == 0).all())

    def test_literal_revisions_exclude_negation_hypotheses_and_boilerplate(self):
        positive = extract_guidance("We raised our full-year revenue guidance.\nWe reduced our profit outlook.\nWe withdrew our annual guidance.")
        self.assertEqual([positive[c] for c in CONTENT_COLUMNS[1:]], [1, 1, 1])
        for sentence in ["We did not reduce our guidance.", "We may raise our guidance.", "If we lower our guidance, demand might fall.",
                         "Revenue increased 10%.\nForward-Looking Statements\nWe raised our guidance."]:
            row = extract_guidance(sentence)
            self.assertEqual([row[c] for c in CONTENT_COLUMNS[1:]], [0, 0, 0])

    def test_range_midpoint_and_units_are_not_confused_with_analyst_surprise(self):
        text = "The company posted quarterly revenue of $80 billion.\nGuidance for its fiscal 2020 second quarter:\nRevenue between $60 billion and $64 billion."
        row = extract_guidance(text)
        self.assertEqual(row["guidance_forward_growth_known"], 1)
        self.assertAlmostEqual(row["guidance_forward_growth"], 62/80-1)
        self.assertEqual(row["guidance_lowered"], 0)
        text = "We reported revenue for the first quarter ended March 31, 2020, of $500 million.\nOur outlook for the second quarter is as follows:\nRevenue is expected to be $0.6 billion."
        row = extract_guidance(text)
        self.assertAlmostEqual(row["guidance_forward_growth"], .2)

    def test_missing_quarterly_outlook_is_explicit(self):
        for text in ["Revenue was $80 billion.\nWe will provide guidance on the earnings call.",
                     "The company posted quarterly revenue of $80 billion.\nOur outlook for the full year:\nRevenue is expected to be $300 billion."]:
            row = extract_guidance(text)
            self.assertEqual(row["guidance_forward_growth_known"], 0)
            self.assertEqual(row["guidance_forward_growth"], 0)

    def test_filing_day_excluded_five_signal_sessions_and_future_events_cannot_leak(self):
        history = pd.DataFrame({"date": pd.bdate_range("2024-01-04", periods=12), "ticker": "TEST"})
        event = {"ticker": "TEST", "status": "complete", "filing_date": "2024-01-05", "available_date": "2024-01-06",
                 **extract_guidance("We raised our revenue guidance.")}
        before = build_event_features(history, [event])
        self.assertEqual(before.earnings_release_recent.tolist(), [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        later = {**event, "filing_date": "2024-01-17", "available_date": "2024-01-18"}
        after = build_event_features(history, [event, later])
        pd.testing.assert_frame_equal(before.iloc[:10], after.iloc[:10])
        self.assertEqual(len(before), len(history))
        with self.assertRaises(ValueError):
            build_event_features(history, [{**event, "status": "failed"}])

    def test_loader_rejects_incomplete_index_and_tampered_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data/guidance_v11"
            folder.mkdir(parents=True)
            path = folder / "test.txt"
            path.write_text("Test announcement")
            ref = {"accession": "test", "ticker": "TEST", "primary_url": "https://www.sec.gov/Archives/test/8k.htm"}
            record = {**ref, "status": "complete", "document_url": "https://www.sec.gov/Archives/test/ex99.htm",
                      "filing_date": "2024-01-05", "file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}
            manifest = folder / "source_manifest.json"
            with patch("quantlab_ai.data.guidance.filing_index", return_value=[ref]):
                manifest.write_text(json.dumps({"records": []}))
                with self.assertRaisesRegex(ValueError, "coverage"):
                    load_documents(root)
                manifest.write_text(json.dumps({"records": [record]}))
                self.assertEqual(len(load_documents(root)[0]), 1)
                path.write_text("Changed announcement")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    load_documents(root)

    def test_test_outcomes_cannot_change_model_cutoff_or_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.ensure_directories()
            history = FeatureBuilder(settings).build(prices(450), "TEST")
            for column in EVENT_COLUMNS:
                history[column] = 0
            history["earnings_release_recent"] = (np.arange(len(history)) % 20 < 5).astype(int)
            trainer = ClassicalModelTrainer(settings, "logistic_regression", "compact", "regularized")
            train, validation, test = trainer.walk_forward_splits(history)[0]
            before, audit, _ = evaluate_fold(settings, train, validation, test, "logistic_regression", "guidance_content")
            after, changed, _ = evaluate_fold(settings, train, validation, test.assign(target=1-test.target), "logistic_regression", "guidance_content")
            self.assertEqual(audit, changed)
            np.testing.assert_array_equal(before.prob_up, after.prob_up)
            np.testing.assert_array_equal(before.prediction, after.prediction)
            with self.assertRaises(ValueError):
                evaluate_fold(settings, train, validation, validation, "logistic_regression", "guidance_content")
