"""Surprise research and dated upcoming-call features with distinct data-quality rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

SURPRISE_COLUMNS = ['last_eps_surprise', 'decayed_eps_surprise', 'days_since_earnings_report', 'earnings_surprise_known']
SCHEDULE_COLUMNS = ['days_until_earnings_call', 'earnings_call_within_5d', 'earnings_schedule_known']


def add_surprises(features: pd.DataFrame, events: pd.DataFrame, *, allow_retrospective_snapshot: bool = False) -> pd.DataFrame:
    if events.empty:
        raise ValueError('No reported earnings events available.')
    if not events['vintage_verified'].eq(True).all() and not allow_retrospective_snapshot:
        raise ValueError('Unverified retrospective snapshot: explicitly opt into exploratory research only.')
    source = events[['report_date','available_date','surprise_fraction']].copy()
    for column in ['report_date','available_date']:
        source[column] = pd.to_datetime(source[column])
    if (source.available_date <= source.report_date).any():
        raise ValueError('Day-level earnings data cannot be used on the report date.')
    if source.available_date.duplicated().any() or not np.isfinite(source.surprise_fraction).all():
        raise ValueError('Invalid or duplicate surprise observations.')
    result = features.sort_values('date').copy()
    result['date'] = pd.to_datetime(result.date)
    source = source.sort_values('available_date').rename(columns={'available_date':'surprise_available_date'})
    result = pd.merge_asof(result, source, left_on='date', right_on='surprise_available_date', direction='backward')
    age = (result.date-result.report_date).dt.days
    known = age.notna() & age.le(180)
    result['last_eps_surprise'] = result.surprise_fraction.clip(-1,1).where(known,0.0)
    result['decayed_eps_surprise'] = result.last_eps_surprise * np.exp(-age.fillna(180)/20)
    result['days_since_earnings_report'] = age.clip(upper=180).fillna(180)
    result['earnings_surprise_known'] = known.astype(float)
    return result


def add_schedules(features: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Use the latest *known* revision for each event, not its ultimate date."""
    result = features.copy()
    result['date'] = pd.to_datetime(result.date)
    events = schedules.copy()
    for column in ['known_at','published_date','scheduled_date']:
        events[column] = pd.to_datetime(events[column])
    if (events.known_at <= events.published_date).any() or (events.scheduled_date < events.known_at).any():
        raise ValueError('Schedules require advance publication and conservative next-day availability.')
    if events.duplicated(['event_id','known_at']).any():
        raise ValueError('Ambiguous schedule revisions.')
    events = events.sort_values('known_at')
    days, known = [], []
    for date in result.date:
        available = events.loc[events.known_at.le(date)].drop_duplicates('event_id', keep='last')
        future = available.loc[available.scheduled_date.ge(date), 'scheduled_date']
        delta = int((future.min()-date).days) if len(future) else 90
        valid = len(future)>0 and delta<=90
        days.append(min(delta,90))
        known.append(float(valid))
    result['days_until_earnings_call'] = days
    result['earnings_schedule_known'] = known
    result['earnings_call_within_5d'] = ((result.days_until_earnings_call<=5)&result.earnings_schedule_known.eq(1)).astype(float)
    return result
