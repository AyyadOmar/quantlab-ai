from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
from quantlab_ai.data.news import append_versions, available_before, collect, feed_health, parse_feed


def rss(title="Earnings", published="Fri, 18 Sep 2026 12:00:00 GMT"):
    return f'<rss><channel><item><title>{title}</title><link>https://example.com/a</link><pubDate>{published}</pubDate></item></channel></rss>'


class NewsTests(unittest.TestCase):
    def test_old_publication_is_not_available_before_first_observation(self):
        rows, rejected = parse_feed(rss(), "AAPL", "2026-09-20T12:00:00Z")
        self.assertEqual(rejected, 0)
        self.assertEqual(available_before(rows, "2026-09-19T12:00:00Z"), [])
        self.assertEqual(len(available_before(rows, "2026-09-20T12:00:00Z")), 1)

    def test_future_missing_and_naive_dates_rejected(self):
        for stamp in ["", "2026-09-21T12:00:00Z", "2026-09-18T12:00:00"]:
            self.assertEqual(parse_feed(rss(published=stamp), "AAPL", "2026-09-20T12:00:00Z"), ([], 1))

    def test_atom_updated_date_and_alternate_link(self):
        payload = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Launch</title><link rel="self" href="https://example.com/api"/><link href="https://example.com/a"/><published>2026-09-18T12:00:00Z</published><updated>2026-09-19T12:00:00Z</updated></entry></feed>'
        rows, rejected = parse_feed(payload, "AAPL", "2026-09-20T12:00:00Z")
        self.assertEqual(rejected, 0)
        self.assertEqual(rows[0]["url"], "https://example.com/a")
        self.assertEqual(rows[0]["updated_at"], "2026-09-19T12:00:00+00:00")

    def test_dedup_preserves_first_seen_and_keeps_revision(self):
        first, _ = parse_feed(rss(), "AAPL", "2026-09-20T12:00:00Z")
        repeat, _ = parse_feed(rss(), "AAPL", "2026-09-21T12:00:00Z")
        revised, _ = parse_feed(rss(title="Updated earnings"), "AAPL", "2026-09-21T12:00:00Z")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.jsonl"
            self.assertEqual(append_versions(path, first + first), 1)
            self.assertEqual(append_versions(path, repeat), 0)
            self.assertEqual(append_versions(path, revised), 1)
            self.assertIn(first[0]["observed_at"], path.read_text().splitlines()[0])

    def test_feed_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder, patch("quantlab_ai.data.news.requests.get", side_effect=requests.Timeout("offline")):
            results = collect(Path(folder))
            self.assertEqual(len(results), 3)
            self.assertTrue(all(row["status"] == "failed" for row in results))
            self.assertFalse((Path(folder) / "articles.jsonl").exists())

    def test_stale_and_empty_feed_cannot_pass_as_healthy(self):
        rows, _ = parse_feed(rss(published="2025-05-07T04:00:00Z"), "MSFT", "2026-09-20T12:00:00Z")
        self.assertEqual(feed_health(rows, "2026-09-20T12:00:00Z")["status"], "stale")
        self.assertEqual(feed_health([], "2026-09-20T12:00:00Z")["status"], "empty")
