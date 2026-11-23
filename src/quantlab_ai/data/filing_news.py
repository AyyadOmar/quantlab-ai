"""Dated corporate news from SEC filing metadata in the public submissions cache."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .context import COMPANY_CIKS


FILING_NEWS_COLUMNS = [
    "sec_filings_7d",
    "sec_earnings_filings_7d",
    "sec_disclosure_filings_7d",
    "sec_governance_filings_7d",
    "sec_filings_delta_7d",
]
_FIELDS = ("form", "items", "filingDate", "acceptanceDateTime", "accessionNumber", "primaryDocument")


def load_filing_events(root: Path, ticker: str) -> tuple[pd.DataFrame, dict]:
    """Load verified, distinct original 8-Ks for the manifest's historical period."""
    folder = Path(root) / "data/context_v4"
    manifest_bytes = (folder / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    start, end = manifest["start"], manifest["end_exclusive"]
    if pd.Timestamp(manifest["retrieved_at"]).date() < pd.Timestamp(end).date():
        raise ValueError("SEC snapshot predates the requested coverage end")
    sources = [s for s in manifest["sources"] if s["kind"] == "earnings_filings" and s["ticker"] == ticker]
    if len(sources) != 1:
        raise ValueError(f"Missing or ambiguous SEC source for {ticker}")
    cik = COMPANY_CIKS[ticker]
    files = sources[0]["sources"]
    payloads = {}
    for reference in files:
        name = reference["file"]
        if Path(name).name != name or reference["url"] != f"https://data.sec.gov/submissions/{name}":
            raise ValueError("Unexpected SEC submissions source")
        raw = (folder / name).read_bytes()
        if sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError(f"SEC hash mismatch for {name}")
        payloads[name] = json.loads(raw)
    top_name = f"CIK{cik}.json"
    top = payloads.get(top_name, {})
    if str(top.get("cik")) != cik or ticker not in top.get("tickers", []):
        raise ValueError(f"SEC submissions do not identify {ticker}")
    required = {top_name}
    for archive in top["filings"]["files"]:
        if archive["filingFrom"] < end and archive["filingTo"] >= start:
            required.add(archive["name"])
    if set(payloads) != required:
        raise ValueError(f"Incomplete or unexpected SEC archive coverage for {ticker}")

    records = []
    for name, payload in payloads.items():
        data = payload["filings"]["recent"] if name == top_name else payload
        frame = pd.DataFrame({field: data[field] for field in _FIELDS})
        records.append(frame.loc[(frame.form == "8-K") & (frame.filingDate >= start) & (frame.filingDate < end)])
    filings = pd.concat(records, ignore_index=True).drop_duplicates()
    if filings.accessionNumber.duplicated().any() or filings.accessionNumber.isna().any():
        raise ValueError("Conflicting SEC filings for the same accession")
    if filings["items"].isna().any():
        raise ValueError("Missing SEC filing item field")
    accepted = pd.to_datetime(filings.acceptanceDateTime, utc=True, errors="raise")
    if accepted.isna().any() or not filings.acceptanceDateTime.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").all():
        raise ValueError("SEC acceptance timestamps need an explicit time zone")
    accepted_day = accepted.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    filed_day = pd.to_datetime(filings.filingDate)
    # Delay to the following calendar day even for premarket filings and half-days.
    available = pd.concat([filed_day, accepted_day], axis=1).max(axis=1) + pd.Timedelta(days=1)
    events = pd.DataFrame({
        "ticker": ticker, "accession": filings.accessionNumber, "items": filings["items"],
        "filing_date": filed_day, "accepted_at": accepted, "available_date": available,
        "url": (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                + filings.accessionNumber.str.replace("-", "", regex=False) + "/" + filings.primaryDocument),
    }).sort_values(["available_date", "accession"]).reset_index(drop=True)
    provenance = {
        "ticker": ticker, "start": start, "end_exclusive": end, "retrieved_at": manifest["retrieved_at"],
        "manifest_sha256": sha256(manifest_bytes).hexdigest(), "files": files, "filings": len(events),
    }
    events.attrs.update(ticker=ticker, start=start, end_exclusive=end)
    return events, provenance


def add_filing_news_features(history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Use current and previous seven-calendar-day counts known by signal close."""
    if history.empty or not history.ticker.eq(events.attrs["ticker"]).all():
        raise ValueError("Expected nonempty history for the SEC source ticker")
    if not events.ticker.eq(events.attrs["ticker"]).all() or events.accession.duplicated().any():
        raise ValueError("Unexpected SEC ticker or duplicate accession")
    dates = pd.DatetimeIndex(pd.to_datetime(history.date))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or not dates.equals(dates.normalize()):
        raise ValueError("Expected unique, ordered signal session dates")
    start, end = pd.Timestamp(events.attrs["start"]), pd.Timestamp(events.attrs["end_exclusive"])
    if dates.min() < start + pd.Timedelta(days=14) or dates.max() >= end:
        raise ValueError("Signal dates exceed SEC archive coverage including lookback")
    categories = events["items"].str.split(",").map(lambda codes: {code.strip() for code in codes})
    counts = pd.DataFrame({
        "sec_filings_7d": 1,
        "sec_earnings_filings_7d": categories.map(lambda codes: "2.02" in codes),
        "sec_disclosure_filings_7d": categories.map(lambda codes: bool({"7.01", "8.01"} & codes)),
        "sec_governance_filings_7d": categories.map(lambda codes: any(c.startswith("5.") for c in codes)),
    }, index=events.index)
    daily = counts.groupby(events.available_date).sum().reindex(
        pd.date_range(start, dates.max(), freq="D"), fill_value=0)
    recent = daily.rolling(7, min_periods=7).sum()
    recent["sec_filings_delta_7d"] = recent["sec_filings_7d"].diff(7)
    result = history.copy()
    result[FILING_NEWS_COLUMNS] = recent.loc[dates, FILING_NEWS_COLUMNS].to_numpy(dtype=int)
    return result
