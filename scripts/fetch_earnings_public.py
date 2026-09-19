from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import requests
import yfinance as yf
from bs4 import BeautifulSoup

folder=Path('data/earnings_v5')
folder.mkdir(parents=True,exist_ok=True)
manifest={'retrieved_at':datetime.now(timezone.utc).isoformat(),'sources':[]}
for ticker in ['AAPL','MSFT','NVDA']:
 data=yf.Ticker(ticker).get_earnings_dates(limit=100)
 if data is None:
  raise ValueError(f'Missing earnings for {ticker}')
 path=folder/f'{ticker.lower()}_yahoo_snapshot.csv'
 data.to_csv(path)
 manifest['sources'].append({'kind':'earnings_snapshot','ticker':ticker,'file':path.name,'sha256':sha256(path.read_bytes()).hexdigest(),'rows':len(data),'url':f'https://finance.yahoo.com/calendar/earnings?symbol={ticker}&offset=0&size=100','vintage_verified':False})
 print(ticker,len(data),list(data.columns),str(data.index.min()),str(data.index.max()),flush=True)
 print(data.tail(2).to_string(),flush=True)
for name,url in [('microsoft_archive','https://news.microsoft.com/tag/microsoft-earnings-results/'),('nvidia_archive','https://nvidianews.nvidia.com/news?c=21926&page=1')]:
 r=requests.get(url,timeout=30)
 path=folder/f'{name}.html'
 path.write_text(r.text)
 print(name,r.status_code,r.url,flush=True)
 soup=BeautifulSoup(r.text,'html.parser')
 matches=[(a.get_text(' ',strip=True),a.get('href')) for a in soup.find_all('a',href=True) if 'earnings release date' in a.get_text().lower() or 'sets conference call' in a.get_text().lower()]
 print(matches[:12],flush=True)
 manifest['sources'].append({'kind':'archive_probe','url':url,'status':r.status_code,'file':path.name,'sha256':sha256(path.read_bytes()).hexdigest()})
(folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
