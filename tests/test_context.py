import unittest

import numpy as np
import pandas as pd

from quantlab_ai.data.context import extract_earnings_filings
from quantlab_ai.features.context import add_market_context, add_earnings_context, MARKET_COLUMNS, EARNINGS_COLUMNS
from quantlab_ai.features.profiles import feature_columns


class ContextTests(unittest.TestCase):
    def histories(self):
        dates = pd.bdate_range('2024-01-01', periods=40)
        return {name: pd.DataFrame({'date': dates, 'close': np.arange(40)+100.})
                for name in ['technology', 'volatility', 'long_yield', 'short_yield']}

    def test_same_day_or_future_market_prices_cannot_enter_signal(self):
        sources = self.histories()
        date = sources['technology'].date.iloc[20]
        features = pd.DataFrame({'date': [date], 'return_5d': [.02]})
        before = add_market_context(features, sources)
        modified = {name: data.copy() for name, data in sources.items()}
        for data in modified.values():
            data.loc[data.date.ge(date), 'close'] *= 10
        after = add_market_context(features, modified)
        np.testing.assert_allclose(before[MARKET_COLUMNS], after[MARKET_COLUMNS])
        self.assertEqual(before.technology_source_date.iloc[0], sources['technology'].date.iloc[19])

    def test_market_context_missing_or_stale_fails_instead_of_dropping_rows(self):
        sources = self.histories()
        dates = [sources['technology'].date.iloc[-1] + pd.Timedelta(days=10)]
        with self.assertRaises(ValueError):
            add_market_context(pd.DataFrame({'date': dates, 'return_5d': [.01]}), sources)

    def test_earnings_only_appears_after_filing_and_preserves_prior_values(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-09', '2024-01-10', '2024-01-11', '2024-01-20'])})
        event = pd.DataFrame({'filing_date': ['2024-01-10'], 'available_date': ['2024-01-11']})
        result = add_earnings_context(frame, event)
        self.assertEqual(result.earnings_history_known.tolist(), [0, 0, 1, 1])
        self.assertEqual(result.earnings_recent_5d.tolist(), [0, 0, 1, 0])
        self.assertEqual(result.earnings_days_since_filing.tolist(), [180, 180, 1, 10])
        future = pd.concat([event, pd.DataFrame({'filing_date': ['2024-03-10'], 'available_date': ['2024-03-11']})])
        np.testing.assert_allclose(result[EARNINGS_COLUMNS], add_earnings_context(frame, future)[EARNINGS_COLUMNS])
        invalid = event.copy()
        invalid['available_date'] = invalid.filing_date
        with self.assertRaises(ValueError):
            add_earnings_context(frame, invalid)

    def test_etf_not_applicable_differs_from_missing_company_history(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-11'])})
        self.assertTrue((add_earnings_context(frame, None)[EARNINGS_COLUMNS] == 0).all().all())
        with self.assertRaises(ValueError):
            add_earnings_context(frame, pd.DataFrame())

    def test_sec_filters_earnings_items_amendments_duplicates_and_dates(self):
        data = {'form': ['8-K','8-K','8-K/A','10-Q','8-K'],
                'items': ['2.02,9.01','7.01','2.02','2.02','2.02'],
                'filingDate': ['2024-01-10']*4 + ['2025-01-10'],
                'accessionNumber': ['a','b','c','d','e']}
        events = extract_earnings_filings([{'filings': {'recent': data}}, data], 'TEST', '2024-01-01', '2025-01-01')
        self.assertEqual(len(events), 1)
        self.assertEqual(events.accession.iloc[0], 'a')
        self.assertEqual(events.available_date.iloc[0], '2024-01-11')

    def test_etfs_exclude_earnings_candidates_entirely(self):
        from quantlab_ai.context_study import candidates_for_ticker
        etf_candidates = candidates_for_ticker("SPY")
        self.assertTrue(all(not c.feature_set.endswith(("_earnings", "_context")) for c in etf_candidates))
        self.assertTrue(any(c.feature_set.endswith("_earnings") for c in candidates_for_ticker("AAPL")))

    def test_context_columns_preserve_base_profile_order(self):
        self.assertEqual(feature_columns('compact_context'), feature_columns('compact')+MARKET_COLUMNS+EARNINGS_COLUMNS)
        self.assertEqual(feature_columns('relative_market'), feature_columns('relative')+MARKET_COLUMNS)


if __name__ == '__main__':
    unittest.main()
