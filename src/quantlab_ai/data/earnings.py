"""Normalize public earnings snapshots without claiming historical-vintage certainty."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_snapshot(frame: pd.DataFrame, ticker: str, retrieved_at: str) -> pd.DataFrame:
    data = frame.copy()
    timestamps = pd.to_datetime(data['Earnings Date'], utc=True, errors='raise')
    data['report_date'] = timestamps.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    for column in ['EPS Estimate', 'Reported EPS', 'Surprise(%)']:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    # Upcoming estimate-only events are never treated as released earnings.
    data = data.loc[np.isfinite(data[['EPS Estimate','Reported EPS','Surprise(%)']]).all(axis=1)].copy()
    if data.report_date.duplicated().any():
        raise ValueError('Ambiguous duplicate earnings reports require source review.')
    data['ticker'] = ticker
    data['available_date'] = data.report_date + pd.Timedelta(days=1)
    data['surprise_fraction'] = data['Surprise(%)'] / 100
    data['retrieved_at'] = retrieved_at
    data['vintage_verified'] = False
    return data.sort_values('report_date').reset_index(drop=True)


def load_earnings_snapshots(root: Path) -> tuple[dict, dict]:
    folder = root / 'data' / 'earnings_v5'
    manifest = json.loads((folder/'source_manifest.json').read_text())
    snapshots = {}
    for source in manifest['sources']:
        if source['kind'] != 'earnings_snapshot':
            continue
        path = folder/source['file']
        if sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Snapshot hash mismatch: {path.name}')
        snapshots[source['ticker']] = normalize_snapshot(pd.read_csv(path), source['ticker'], manifest['retrieved_at'])
    return snapshots, manifest
