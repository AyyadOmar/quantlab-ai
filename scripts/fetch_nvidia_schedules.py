"""Collect public, dated earnings-call announcements; keep source pages for audit."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from urllib.parse import urljoin
import json
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser

folder = Path('data/earnings_v5/nvidia_schedules')
folder.mkdir(parents=True, exist_ok=True)
session = requests.Session()
records, sources, failures = [], [], []
for page in range(1, 101):
    url = f'https://nvidianews.nvidia.com/news?c=21926&page={page}'
    path = folder / f'archive_{page}.html'
    if not path.exists():
        response = session.get(url, timeout=30)
        response.raise_for_status()
        path.write_text(response.text)
        time.sleep(.25)
    soup = BeautifulSoup(path.read_text(), 'html.parser')
    items = soup.select('.index-item-wrapper')
    if not items:
        raise ValueError(f'Archive page {page} has no entries')
    dates = []
    for item in items:
        date_node = item.select_one('.index-item-text-info-date')
        link = item.select_one('h3 a')
        if not date_node or not link:
            continue
        published = pd.Timestamp(parser.parse(date_node.get_text(strip=True))).normalize()
        dates.append(published)
        if 'sets conference call' not in link.get_text(' ', strip=True).lower():
            continue
        if not pd.Timestamp('2017-01-01') <= published < pd.Timestamp('2026-05-25'):
            continue
        description = item.select_one('.index-item-text-description').get_text(' ', strip=True)
        match = re.search(r'(?:conference call|webcast) on (?:\w+,?\s+)?([A-Z][a-z]+\.?\s+\d{1,2})(?:,?\s+(20\d{2}))?', description)
        article_url = urljoin(url, link['href'])
        evidence_path = path
        if not match:
            evidence_path = folder / (sha256(article_url.encode()).hexdigest()[:16] + '.html')
            if not evidence_path.exists():
                response = session.get(article_url, timeout=30)
                response.raise_for_status()
                evidence_path.write_text(response.text)
                time.sleep(.25)
            article = BeautifulSoup(evidence_path.read_text(), 'html.parser').get_text(' ', strip=True)
            match = re.search(r'(?:conference call|webcast) on (?:\w+,?\s+)?([A-Z][a-z]+\.?\s+\d{1,2})(?:,?\s+(20\d{2}))?', article)
        if not match:
            failures.append({'url': article_url, 'published': str(published.date()), 'description': description})
            continue
        year = int(match.group(2)) if match.group(2) else published.year
        scheduled = pd.Timestamp(parser.parse(f'{match.group(1)} {year}')).normalize()
        if scheduled < published:
            scheduled = scheduled.replace(year=scheduled.year+1)
        if not 0 < (scheduled-published).days <= 90:
            failures.append({'url': article_url, 'reason': 'implausible advance notice', 'description': description})
            continue
        records.append({'ticker': 'NVDA', 'event_id': f'NVDA-call-{scheduled.date()}',
                        'published_date': str(published.date()),
                        'known_at': str((published+pd.Timedelta(days=1)).date()),
                        'scheduled_date': str(scheduled.date()), 'event_kind': 'earnings_call',
                        'source_url': article_url, 'archive_url': url,
                        'source_file': str(evidence_path), 'source_sha256': sha256(evidence_path.read_bytes()).hexdigest()})
    sources.append({'url': url, 'file': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()})
    print(f'Page {page}: {min(dates).date()} to {max(dates).date()}, schedules={len(records)}', flush=True)
    if min(dates) < pd.Timestamp('2017-01-01'):
        break
pd.DataFrame(records).drop_duplicates(['event_id','known_at']).sort_values('known_at').to_csv(folder.parent/'nvda_schedules.csv',index=False)
(folder/'manifest.json').write_text(json.dumps({'retrieved_at':datetime.now(timezone.utc).isoformat(),'sources':sources,'parse_failures':failures,'events':len(records),'coverage':'2017-01-01 through 2026-05-24 publication dates'},indent=2))
print('Complete:',len(records),'schedules;',len(failures),'parse failures',flush=True)
