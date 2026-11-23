from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantlab_ai import prospective_news as study


def synthetic_prices(ticker, start, end):
    dates = study.xc.get_calendar("XNYS").sessions_in_range(start, pd.Timestamp(end)-pd.Timedelta(days=1))
    base = 100. + np.arange(len(dates))/10
    return pd.DataFrame({"ticker": ticker, "date": dates, "open": base, "close": base+np.where(dates.day%2,1.,-1.)})


class ProspectiveNewsTests(unittest.TestCase):
    def test_complete_window_with_no_articles_is_observable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            end = datetime(2026,11,23,21,tzinfo=timezone.utc)
            (root/"articles.jsonl").write_text("")
            for i in range(1, 9):
                stamp = end-timedelta(minutes=15*i)
                path = root/"snapshots"/str(i)/"manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"status":"ok", "batch_at":stamp.isoformat(),
                                            "observed_at":(stamp+timedelta(minutes=5)).isoformat()}))
            missing, coverage = study.news_features(end.isoformat(), root)
            self.assertEqual(missing, {})
            self.assertEqual(coverage["expected_morning_batches"], 10)
            morning_end = end - timedelta(hours=4)
            for i in range(1, 11):
                stamp = morning_end - timedelta(minutes=i*15)
                mp = root/"snapshots"/("morning"+str(i))/"manifest.json"
                mp.parent.mkdir()
                mp.write_text(json.dumps({"status":"ok", "batch_at":stamp.isoformat(),
                                          "observed_at":(stamp+timedelta(minutes=5)).isoformat()}))
            data, _ = study.news_features(end.isoformat(), root)
            self.assertEqual(data["NVDA"]["morning_headline_balance"], 0)
            # A morning article and an afternoon revision must retain the noon reading.
            first = {"ticker":"NVDA", "url":"story", "title":"Nvidia raises guidance", "version_id":"v1",
                     "gdelt_batch_at":(morning_end-timedelta(minutes=30)).isoformat(),
                     "available_at":(morning_end-timedelta(minutes=25)).isoformat(), "article_tone":2}
            revision = {**first,"version_id":"v2","article_tone":-3,"available_at":(end-timedelta(minutes=10)).isoformat()}
            (root/"articles.jsonl").write_text("".join(json.dumps(r)+"\n" for r in [first,revision]))
            scores = [{"scorer":study.SCORER,"article_version":r["version_id"],"scored_at":r["available_at"],
                       "probabilities":{"positive":p,"negative":1-p,"neutral":0},"quote_probabilities":[]}
                      for r,p in [(first,1),(revision,0)]]
            (root/"sentiment.jsonl").write_text("".join(json.dumps(r)+"\n" for r in scores))
            rows, _ = study.news_features(end.isoformat(), root)
            self.assertEqual(rows["NVDA"]["morning_headline_balance"], 1)
            self.assertEqual(rows["NVDA"]["morning_article_tone_mean"], 2)
            self.assertEqual(rows["NVDA"]["headline_urls_1h"], 0)
            self.assertEqual(data["NVDA"]["eligible_urls"], 0)
            self.assertEqual(data["NVDA"]["headline_balance"], 0)
            path.unlink()
            missing, coverage = study.news_features(end.isoformat(), root)
            self.assertEqual(missing,{})
            self.assertEqual(coverage["previous_batches"],3)

    def test_chronological_learning_settlement_and_idempotence(self):
        calendar = study.xc.get_calendar("XNYS")
        days = calendar.sessions_in_range(study.START,"2027-01-08")[:22]
        inputs = {t:{c:0 for c in study.COUNTS + study.TEXT + study.MORNING} for t in study.HEADLINES}
        with TemporaryDirectory() as tmp, closing(study.connect(Path(tmp)/"demo.db")) as db, \
             patch.object(study, "news_features", return_value=(inputs,{})):
            for i, day in enumerate(days):
                now = study.data_ready_after(calendar, day)+timedelta(minutes=15)
                saved = study.capture(db, clock=lambda:now, download=synthetic_prices)
                self.assertEqual(saved["status"],"forecast" if i>=20 else "training")
                if i==20:
                    self.assertEqual(saved["training_rows"],60)
                    score = study.report(db,now)
                    self.assertEqual(score["scored_forecasts_per_arm"],0)
                again = study.capture(db,clock=lambda:now,download=lambda *_:self.fail("duplicate fetch"))
                self.assertEqual(again["status"],"already_recorded")
            score = study.report(db,now)
            self.assertEqual(score["scored_forecasts_per_arm"],3)
            self.assertEqual(score["missing_runs"],[])
            self.assertEqual(score["wording_vs_counts"], {"calls_fixed_by_wording":0,"additional_errors_with_wording":0})
            self.assertEqual(set(score["results"]),{*study.ARMS,"always_up","always_down"})
            for arm in study.ARMS:
                self.assertEqual(score["results"][arm],score["results"]["price"])

    def test_failed_fetch_and_late_write_are_retryable(self):
        now = datetime(2026,11,23,21,30,tzinfo=timezone.utc)
        news = {t:{c:0 for c in study.COUNTS + study.TEXT + study.MORNING} for t in study.HEADLINES}
        with TemporaryDirectory() as tmp, closing(study.connect(Path(tmp)/"demo.db")) as db, \
             patch.object(study, "news_features",return_value=(news,{})):
            with self.assertRaises(ValueError):
                study.capture(db,clock=lambda:now,download=lambda *a:synthetic_prices(*a).iloc[:-1])
            clock = iter([now,now+timedelta(days=1)])
            with self.assertRaisesRegex(ValueError,"window expired"):
                study.capture(db,clock=lambda:next(clock),download=synthetic_prices)
            self.assertEqual(study.report(db,now)["sessions_recorded"],0)
            self.assertEqual(study.capture(db,clock=lambda:now,download=synthetic_prices)["status"],"training")
