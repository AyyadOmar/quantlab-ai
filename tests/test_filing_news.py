from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from quantlab_ai.data.filing_news import FILING_NEWS_COLUMNS, add_filing_news_features, load_filing_events


class FilingNewsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "data/context_v4"
        self.folder.mkdir(parents=True)
        self.top = "CIK0000320193.json"
        self.archive = "CIK0000320193-submissions-001.json"
        columns = ["form", "items", "filingDate", "acceptanceDateTime", "accessionNumber", "primaryDocument"]

        def records(rows):
            return dict(zip(columns, map(list, zip(*rows))))

        older = records([
            ("8-K", "2.02,9.01", "2024-01-19", "2024-01-19T21:30:00Z", "a", "a.htm"),
            ("8-K/A", "2.02", "2024-01-19", "2024-01-19T21:31:00Z", "b", "b.htm"),
        ])
        recent = records([
            ("8-K", "2.02,9.01", "2024-01-19", "2024-01-19T21:30:00Z", "a", "a.htm"),
            ("8-K", "7.01,8.01,5.02", "2024-01-22", "2024-01-23T02:00:00Z", "c", "c.htm"),
            ("8-K", "2.02", "2024-02-20", "2024-02-21T17:00:00Z", "d", "d.htm"),
        ])
        self.payloads = {
            self.top: {"cik": "0000320193", "tickers": ["AAPL"], "filings": {
                "recent": recent, "files": [{"name": self.archive, "filingFrom": "2023-01-01", "filingTo": "2024-01-19"}]}},
            self.archive: older,
        }
        files = []
        for name, data in self.payloads.items():
            raw = json.dumps(data).encode()
            (self.folder / name).write_bytes(raw)
            files.append({"file": name, "url": f"https://data.sec.gov/submissions/{name}", "sha256": sha256(raw).hexdigest()})
        self.manifest = {"start": "2024-01-01", "end_exclusive": "2024-03-01", "retrieved_at": "2024-04-01T00:00:00Z",
                         "sources": [{"kind": "earnings_filings", "ticker": "AAPL", "sources": files}]}
        self.manifest_path = self.folder / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_event_windows_use_publication_and_count_distinct_filings(self):
        events, audit = load_filing_events(self.root, "AAPL")
        self.assertEqual(audit["filings"], 3)
        self.assertEqual(events.available_date.dt.strftime("%Y-%m-%d").tolist(),
                         ["2024-01-20", "2024-01-23", "2024-02-22"])
        history = pd.DataFrame({"ticker": "AAPL", "date": pd.to_datetime([
            "2024-01-19", "2024-01-22", "2024-01-23", "2024-01-26", "2024-01-29", "2024-02-01"])},
            index=[31, 33, 38, 44, 48, 66])
        result = add_filing_news_features(history, events)
        self.assertEqual(result.index.tolist(), history.index.tolist())
        self.assertEqual(result[FILING_NEWS_COLUMNS].values.tolist(), [
            [0, 0, 0, 0, 0], [1, 1, 0, 0, 1], [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2], [1, 0, 1, 1, 0], [0, 0, 0, 0, -2],
        ])
        pd.testing.assert_frame_equal(result, add_filing_news_features(history, events.iloc[:-1]))

    def test_missing_archives_hash_mismatch_and_uncovered_dates_stop_model(self):
        events, _ = load_filing_events(self.root, "AAPL")
        for date in ["2024-01-02", "2024-03-01"]:
            with self.assertRaisesRegex(ValueError, "coverage"):
                add_filing_news_features(pd.DataFrame({"ticker": ["AAPL"], "date": [date]}), events)
        (self.folder / self.archive).write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_filing_events(self.root, "AAPL")
        self.manifest["sources"][0]["sources"].pop()
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "archive coverage"):
            load_filing_events(self.root, "AAPL")
