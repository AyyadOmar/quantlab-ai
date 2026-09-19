import unittest
import numpy as np
import pandas as pd

from quantlab_ai.data.earnings import normalize_snapshot
from quantlab_ai.features.earnings import add_surprises, add_schedules, SURPRISE_COLUMNS, SCHEDULE_COLUMNS


class EarningsEventTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame({'report_date':pd.to_datetime(['2024-01-10','2024-04-10']),
                             'available_date':pd.to_datetime(['2024-01-11','2024-04-11']),
                             'surprise_fraction':[.1,-.2], 'vintage_verified':[False,False]})

    def test_snapshot_opt_in_required(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-12'])})
        with self.assertRaises(ValueError):
            add_surprises(frame,self.events())

    def test_report_day_and_future_surprise_excluded(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-10','2024-01-11','2024-01-20'])})
        result=add_surprises(frame,self.events(),allow_retrospective_snapshot=True)
        self.assertEqual(result.earnings_surprise_known.tolist(),[0,1,1])
        np.testing.assert_allclose(result.last_eps_surprise,[0,.1,.1])
        changed=self.events()
        changed.loc[1,'surprise_fraction']=100
        np.testing.assert_allclose(result[SURPRISE_COLUMNS],add_surprises(frame,changed,allow_retrospective_snapshot=True)[SURPRISE_COLUMNS])

    def test_stale_surprise_is_unknown(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2025-01-01'])})
        result=add_surprises(frame,self.events(),allow_retrospective_snapshot=True)
        self.assertEqual(result.earnings_surprise_known.iloc[0],0)
        self.assertEqual(result.last_eps_surprise.iloc[0],0)
        self.assertEqual(result.decayed_eps_surprise.iloc[0],0)

    def test_snapshot_keeps_zero_eps_and_drops_unreported_future_events(self):
        raw=pd.DataFrame({'Earnings Date':['2024-01-10 16:00:00-05:00','2024-04-10 16:00:00-04:00'],
                          'EPS Estimate':[0.,1.], 'Reported EPS':[.1,np.nan], 'Surprise(%)':[100.,np.nan]})
        result=normalize_snapshot(raw,'TEST','2026-09-19T00:00:00Z')
        self.assertEqual(len(result),1)
        self.assertEqual(result.available_date.iloc[0],pd.Timestamp('2024-01-11'))
        self.assertFalse(result.vintage_verified.iloc[0])

    def test_schedules_are_unknown_before_publication_availability(self):
        events=pd.DataFrame({'event_id':['q1'], 'published_date':['2024-01-01'], 'known_at':['2024-01-02'], 'scheduled_date':['2024-01-10']})
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-01-02','2024-01-08','2024-01-11'])})
        result=add_schedules(frame,events)
        self.assertEqual(result.earnings_schedule_known.tolist(),[0,1,1,0])
        self.assertEqual(result.days_until_earnings_call.tolist(),[90,8,2,90])
        self.assertEqual(result.earnings_call_within_5d.tolist(),[0,0,1,0])

    def test_schedule_revision_only_affects_later_predictions(self):
        events=pd.DataFrame({'event_id':['q1','q1'], 'published_date':['2024-01-01','2024-01-05'],
                             'known_at':['2024-01-02','2024-01-06'], 'scheduled_date':['2024-01-10','2024-01-15']})
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-04','2024-01-06','2024-01-11'])})
        result=add_schedules(frame,events)
        self.assertEqual(result.days_until_earnings_call.tolist(),[6,9,4])
        before=add_schedules(frame.iloc[[0]],events.iloc[[0]])
        np.testing.assert_allclose(before[SCHEDULE_COLUMNS],result.iloc[[0]][SCHEDULE_COLUMNS])

    def test_ambiguous_or_same_day_availability_rejected(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-11'])})
        events=self.events()
        events.loc[0,'available_date']=events.loc[0,'report_date']
        with self.assertRaises(ValueError):
            add_surprises(frame,events,allow_retrospective_snapshot=True)
        schedule=pd.DataFrame({'event_id':['q1'], 'published_date':['2024-01-01'], 'known_at':['2024-01-01'], 'scheduled_date':['2024-01-10']})
        with self.assertRaises(ValueError):
            add_schedules(frame,schedule)


if __name__=='__main__':
    unittest.main()
