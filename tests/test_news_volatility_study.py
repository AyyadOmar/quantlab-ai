import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quantlab_ai.config import Settings
from quantlab_ai.data.filing_news import add_filing_news_features
from quantlab_ai.features.builder import FeatureBuilder
from quantlab_ai.features.volatility_forecast import add_volatility_features
from quantlab_ai.news_volatility_study import ACTIVE, EXTRAS, PRICE, evaluate, inputs_for_variant
from test_research_protocol import prices


class NewsVolatilityStudyTests(unittest.TestCase):
    def test_combined_model_on_synthetic_prices_and_filings(self):
        # Exercise the study on a clean checkout without requiring market-data downloads.
        with TemporaryDirectory() as folder:
            settings = Settings(project_root=Path(folder))
            settings.ensure_directories()
            raw = prices(350)
            history = FeatureBuilder(settings).build(raw, "TEST")
            event_dates = pd.to_datetime(raw.date.iloc[::20]).reset_index(drop=True)
            filings = pd.DataFrame({"ticker": "TEST", "items": "2.02", "available_date": event_dates,
                                    "accession": [str(i) for i in range(len(event_dates))]})
            filings.attrs.update(ticker="TEST", start="2019-01-01", end_exclusive="2022-01-01")
            data = add_filing_news_features(add_volatility_features(history, raw), filings)
            before, audit = evaluate(settings, "logistic_regression", data, "combined")
            last = data.date >= before.loc[before.fold == before.fold.max(), "date"].min()
            data.loc[last, "target"] = 1 - data.loc[last, "target"]
            after, repeated = evaluate(settings, "logistic_regression", data, "combined")
            self.assertEqual(audit, repeated)
            self.assertGreater(int((before.target != after.target).sum()), 0)
            pd.testing.assert_frame_equal(before[["prob_up", "prediction"]], after[["prob_up", "prediction"]])

    def test_price_and_outcome_are_preserved_while_inactive_inputs_are_zeroed(self):
        data = pd.DataFrame({k: [0.1, 0.2] for k in PRICE + EXTRAS})
        data["target"] = [1, 0]
        data["next_day_return"] = [0.05, -0.06]
        original = data.copy()
        for name, active in ACTIVE.items():
            with self.subTest(variant=name):
                masked, cols = inputs_for_variant(data, name)
                self.assertEqual(cols, PRICE if name == "native_price" else PRICE + EXTRAS)
                pd.testing.assert_frame_equal(masked[PRICE + ["target", "next_day_return"]],
                                              original[PRICE + ["target", "next_day_return"]])
                if name != "native_price":
                    for column in EXTRAS:
                        if column in active:
                            pd.testing.assert_series_equal(original[column], masked[column])
                        else:
                            self.assertEqual(masked[column].tolist(), [0.0, 0.0])
        pd.testing.assert_frame_equal(original, data)
