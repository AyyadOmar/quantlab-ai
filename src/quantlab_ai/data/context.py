"""Download auditable market context and SEC earnings-related filing dates."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import pandas as pd
import requests
import yfinance as yf

from .loader import MarketDataLoader

MARKET_SYMBOLS = {"technology": "XLK", "volatility": "^VIX", "long_yield": "^TNX", "short_yield": "^IRX"}
COMPANY_CIKS = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810"}


def extract_earnings_filings(payloads: list[dict], ticker: str, start: str, end: str) -> pd.DataFrame:
    records = []
    for payload in payloads:
        data = payload.get("filings", {}).get("recent", payload)
        if not all(key in data for key in ["form", "items", "filingDate", "accessionNumber"]):
            raise ValueError("SEC history lacks fields needed to identify earnings-related filings.")
        for index, form in enumerate(data["form"]):
            items = {item.strip() for item in data["items"][index].split(",")}
            filed = data["filingDate"][index]
            if form == "8-K" and "2.02" in items and start <= filed < end:
                records.append({"ticker": ticker, "filing_date": filed,
                                "available_date": (pd.Timestamp(filed) + pd.Timedelta(days=1)).date().isoformat(),
                                "accession": data["accessionNumber"][index], "form": form, "item": "2.02"})
    columns = ["ticker", "filing_date", "available_date", "accession", "form", "item"]
    return pd.DataFrame(records, columns=columns).drop_duplicates("accession").sort_values("filing_date").reset_index(drop=True)


def fetch_context(root: Path, start: str = "2017-01-01", end: str = "2026-05-25") -> dict:
    folder = root / "data" / "context_v4"
    folder.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, symbol in MARKET_SYMBOLS.items():
        data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"No context history returned for {symbol}.")
        data = data.reset_index()
        data.columns = [MarketDataLoader._normalize_column_name(column) for column in data.columns]
        data = MarketDataLoader.completed_sessions(data)
        path = folder / f"{name}.csv"
        data.to_csv(path, index=False)
        sources.append({"kind": "market", "symbol": symbol, "file": path.name, "rows": len(data),
                        "start": str(data.date.min()), "end": str(data.date.max()),
                        "sha256": sha256(path.read_bytes()).hexdigest(), "provider": "Yahoo Finance via yfinance"})
        print(f"Fetched {symbol}: {len(data)} sessions", flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = "QuantLabAI research project"
    for ticker, cik in COMPANY_CIKS.items():
        urls = []
        def get_json(name: str) -> dict:
            url = f"https://data.sec.gov/submissions/{name}"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            path = folder / name
            path.write_text(json.dumps(payload))
            urls.append({"url": url, "file": name, "sha256": sha256(path.read_bytes()).hexdigest()})
            time.sleep(0.2)
            return payload
        top = get_json(f"CIK{cik}.json")
        if ticker not in top.get("tickers", []):
            raise ValueError(f"CIK/ticker mismatch for {ticker}.")
        payloads = [top]
        for archive in top["filings"]["files"]:
            if archive["filingFrom"] < end and archive["filingTo"] >= start:
                payloads.append(get_json(archive["name"]))
        events = extract_earnings_filings(payloads, ticker, start, end)
        if events.empty:
            raise ValueError(f"No earnings-related filings found for {ticker}.")
        path = folder / f"{ticker.lower()}_earnings_filings.csv"
        events.to_csv(path, index=False)
        sources.append({"kind": "earnings_filings", "ticker": ticker, "file": path.name, "events": len(events),
                        "first_filing": events.filing_date.min(), "last_filing": events.filing_date.max(),
                        "sha256": sha256(path.read_bytes()).hexdigest(), "sources": urls})
        print(f"Fetched {ticker}: {len(events)} earnings-related filings", flush=True)
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "start": start, "end_exclusive": end,
                "market_availability": "Previous available session only; maximum seven calendar days stale.",
                "earnings_availability": "8-K item 2.02, available from calendar day after filing. Not an announcement calendar or EPS surprise dataset.",
                "sources": sources}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
