from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO
from zipfile import ZipFile

from quantlab_ai.data.news_sentiment import SCORER, score_pending, sentiment_at
from quantlab_ai.data.press_news import parse_batch
from test_press_news import NAME
import importlib.util

spec = importlib.util.spec_from_file_location("collect_script", Path(__file__).resolve().parents[1] / "scripts/collect_press_news.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class SentimentTests(unittest.TestCase):
    def test_cutoff_and_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = []
            for version, minute, title in [("a", 3, "Microsoft raises outlook"), ("b", 15, "Microsoft cuts outlook"),
                                          ("c", 55, "Nvidia and Microsoft sign contract")]:
                rows.append(dict(ticker="MSFT", title=title, version_id=version, url="mixed" if version=="c" else "one",
                                 available_at=f"2026-11-20T20:{minute:02d}:00Z", gdelt_batch_at="2026-11-20T20:00:00Z",
                                 article_tone=-2.0))
            (folder/"articles.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            scores = [dict(article_version=k, scorer=SCORER, scored_at=t, quote_probabilities=[],
                           probabilities={"positive": p, "negative": 1-p, "neutral": 0})
                      for k, t, p in [("a", "2026-11-20T20:04:00Z", 1), ("b", "2026-11-20T21:01:00Z", 0)]]
            (folder/"sentiment.jsonl").write_text("".join(json.dumps(s)+"\n" for s in scores))
            def get(cutoff, batches):
                return sentiment_at(cutoff, folder, dict(recent_batches=batches, expected_batches_per_hour=4))["MSFT"]
            self.assertEqual(get("2026-11-20T21:00:00Z",4)["eligible_urls"], 1)
            self.assertIsNone(get("2026-11-20T21:00:00Z",4)["headline_balance"])
            self.assertEqual(get("2026-11-20T21:00:00Z",4)["article_tone_mean"], -2)
            self.assertIsNone(get("2026-11-20T21:00:00Z",3)["article_tone_mean"])

    def test_article_fields_and_scores_survive_retry(self):
        archive = BytesIO()
        parts = [""]*27
        parts[1:5] = [NAME[:14], "1", "publisher", "https://publisher.example/news"]
        parts[15] = "-3.2,1,4,5,0,0,280"
        parts[22] = "15|25|said|Microsoft raised guidance.#50|19|said|Nvidia faces losses."
        parts[26] = "<PAGE_TITLE>Microsoft reviews its outlook</PAGE_TITLE>"
        with ZipFile(archive, "w") as z:
            z.writestr(NAME[:-4], "\t".join(parts))
        rows, _ = parse_batch(archive.getvalue(), NAME, datetime(2026,11,22,tzinfo=timezone.utc))
        self.assertEqual(rows[0]["company_quotes"], ["Microsoft raised guidance."])
        self.assertEqual(rows[0]["article_tone"], -3.2)
        with tempfile.TemporaryDirectory() as tmp, patch("quantlab_ai.data.news_sentiment.SentimentModel") as model:
            model.return_value.score.side_effect = lambda texts: [dict(positive=.8, negative=.1, neutral=.1) for _ in texts]
            folder = Path(tmp)
            (folder/"articles.jsonl").write_text(json.dumps(rows[0])+"\n")
            self.assertEqual(score_pending(folder),1)
            self.assertEqual(score_pending(folder),0)
            record = json.loads((folder/"sentiment.jsonl").read_text())
            self.assertEqual(record["quote_probabilities"][0]["positive"],.8)

    def test_market_window_holidays_and_early_closes(self):
        for date, expected in [("2026-11-23T18:59", False), ("2026-11-23T19:00", True),
                               ("2026-11-23T21:00", False), ("2026-11-26T20:00", False),
                               ("2026-11-27T16:00", True), ("2026-11-27T18:00", False)]:
            self.assertEqual(script.in_collection_window(datetime.fromisoformat(date).replace(tzinfo=timezone.utc)), expected)
