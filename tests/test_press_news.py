from datetime import datetime, timedelta, timezone
from hashlib import md5
import json
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import requests

from quantlab_ai.data.news import append_versions, available_before
from quantlab_ai.data.press_news import collect, headline_activity_at, healthy_batches, latest_batch, parse_batch


NAME = "20261121181500.gkg.csv.zip"


def batch(*entries):
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        rows = []
        for title, date in entries:
            row = [""] * 27
            row[:5] = ["id", NAME[:14], "1", "paper.example", "https://paper.example/article"]
            row[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
            if date:
                row[26] += f"<PAGE_PRECISEPUBTIMESTAMP>{date}</PAGE_PRECISEPUBTIMESTAMP>"
            rows.append("\t".join(row))
        archive.writestr(NAME[:-4], "\n".join(rows))
    return payload.getvalue()


class PressNewsTests(unittest.TestCase):
    def test_hourly_delta_requires_observed_batches_and_counts_a_revision_once(self):
        end = datetime(2026, 11, 20, 21, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(1, 9):
                stamp = end - timedelta(minutes=15 * i)
                observed = (stamp + timedelta(minutes=3)).isoformat()
                path = root / "snapshots" / str(i) / "manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"status": "ok", "batch_at": stamp.isoformat(), "observed_at": observed}))
                rows.append({"ticker": "MSFT", "url": "revised" if i <= 2 else str(i),
                             "available_at": observed, "gdelt_batch_at": stamp.isoformat()})
            (root / "articles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            stats = headline_activity_at(end.isoformat(), root)["tickers"]
            self.assertEqual(stats["MSFT"], {"headline_urls_1h": 3, "headline_urls_delta_1h": -1})
            self.assertEqual(stats["AAPL"]["headline_urls_1h"], 0)
            path.write_text(json.dumps({"status": "ok", "batch_at": stamp.isoformat(),
                                        "observed_at": (end + timedelta(minutes=1)).isoformat()}))
            sparse = headline_activity_at(end.isoformat(), root)
            self.assertEqual(sparse["previous_batches"], 3)
            self.assertIsNone(sparse["tickers"]["MSFT"]["headline_urls_delta_1h"])
            (root / "snapshots/1/manifest.json").unlink()
            self.assertIsNone(headline_activity_at(end.isoformat(), root)["tickers"]["MSFT"]["headline_urls_1h"])

    def test_missing_dates_first_observation_and_revision(self):
        observed = datetime(2026, 11, 22, tzinfo=timezone.utc)
        payload = batch(("Apple harvest", None), ("Microsoft and NVIDIA", None),
                        ("iPhone sales", "20261121102000"), ("MSFT future", "20261123000000"))
        rows, stats = parse_batch(payload, NAME, observed)
        self.assertEqual([r["ticker"] for r in rows], ["MSFT", "NVDA", "AAPL"])
        self.assertEqual(stats["rejected"], 0)
        self.assertEqual(stats["future_publications"], [
            {"row": 4, "tickers": ["MSFT"], "published_at": "2026-11-23T00:00:00+00:00"},
        ])
        self.assertIsNone(rows[0]["published_at"])
        self.assertEqual(rows[-1]["published_at"], "2026-11-21T10:20:00+00:00")
        self.assertEqual(rows[-1]["publisher_domain"], "paper.example")
        self.assertEqual(available_before(rows, "2026-11-21T23:59:59Z"), [])
        later, _ = parse_batch(payload, NAME, observed.replace(hour=1))
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "articles.jsonl"
            self.assertEqual(append_versions(ledger, rows), 3)
            self.assertEqual(append_versions(ledger, later), 0)
            revised, _ = parse_batch(batch(("iPhone sales revised", "20261121102000")), NAME, observed)
            self.assertEqual(append_versions(ledger, revised), 1)

    def test_future_timestamp_does_not_invalidate_an_observed_batch(self):
        slot = datetime(2026, 11, 21, 18, 15, tzinfo=timezone.utc)
        observed = slot + timedelta(minutes=5)
        for extra, expected in [([], "ok"), ([("MSFT malformed date", "bad-date")], "partial")]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                payload = batch(("Nvidia on time", None), ("Microsoft future", "20261121200000"), *extra)
                index = f"{len(payload)} {md5(payload).hexdigest()} https://data.gdeltproject.org/gdeltv2/{NAME}"
                with patch("quantlab_ai.data.press_news.datetime", wraps=datetime) as clock, patch(
                    "quantlab_ai.data.press_news.requests.get",
                    side_effect=[Mock(text=index), Mock(content=payload)],
                ):
                    clock.now.return_value = observed
                    result = collect(folder)
                self.assertEqual(result["status"], expected)
                self.assertEqual(len(result["future_publications"]), 1)
                self.assertEqual(result["rejected"], len(extra))
                article = json.loads((folder / "articles.jsonl").read_text())
                self.assertEqual(article["title"], "Nvidia on time")
                self.assertEqual(healthy_batches(folder, observed), {slot} if expected == "ok" else set())
                self.assertEqual(healthy_batches(folder, slot), set())

    def test_require_official_index_and_observable_batch(self):
        digest = "ab" * 16
        with self.assertRaises(ValueError):
            latest_batch(f"42 {digest} http://example.com/gdeltv2/{NAME}")
        self.assertEqual(latest_batch(f"42 {digest} http://data.gdeltproject.org/gdeltv2/{NAME}"),
                         (NAME, 42, digest))
        with self.assertRaisesRegex(ValueError, "Future"):
            parse_batch(batch(("MSFT", None)), NAME, datetime(2026, 10, 1, tzinfo=timezone.utc))

    def test_failed_fetch_is_visible_and_does_not_create_a_ledger(self):
        with tempfile.TemporaryDirectory() as tmp, patch("quantlab_ai.data.press_news.requests.get",
                                                          side_effect=requests.Timeout("offline")):
            self.assertEqual(collect(Path(tmp))["status"], "failed")
            self.assertFalse((Path(tmp) / "articles.jsonl").exists())
            self.assertEqual(len(list(Path(tmp).rglob("manifest.json"))), 1)
