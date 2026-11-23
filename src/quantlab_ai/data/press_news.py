"""Collect one public GDELT 2.1 GKG batch for prospective company-news research."""
from datetime import datetime, timedelta, timezone
import fcntl
from hashlib import md5, sha256
from html import unescape
from io import BytesIO
import json
import math
from pathlib import Path
import re
from time import sleep
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

import requests

from .news import append_versions, utc

ARCHIVE = "https://data.gdeltproject.org/gdeltv2/"
INDEX = ARCHIVE + "lastupdate.txt"
DEFAULT_FOLDER = Path("data/news_v9/gdelt")
BATCH_PATTERN = r"(\d{14})\.gkg\.csv"
# Restrict Apple matches to company/product references to avoid fruit headlines.
HEADLINES = {
    "AAPL": re.compile(r"\b(?:AAPL|iPhone|iPad|MacBook|Apple(?:'s|’s)? (?:Inc|Watch|TV|Music|Pay|Vision|"
                       r"Intelligence|stock|shares|CEO|earnings|revenue|profit|guidance|outlook|"
                       r"(?:raises|cuts|lowers|withdraws) (?:its )?(?:guidance|forecast|outlook)))\b", re.I),
    "MSFT": re.compile(r"\b(?:Microsoft|MSFT)\b", re.I),
    "NVDA": re.compile(r"\b(?:Nvidia|NVDA)\b", re.I),
}


def gdelt_time(stamp):
    return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def latest_batch(index):
    """Accept only an official GKG archive name. Always fetch it over HTTPS."""
    for line in index.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        match = re.fullmatch(r"https?://data\.gdeltproject\.org/gdeltv2/(" + BATCH_PATTERN + r"\.zip)", parts[2])
        if match and re.fullmatch(r"[a-fA-F0-9]{32}", parts[1]):
            return match[1], int(parts[0]), parts[1].lower()
    raise ValueError("No GKG archive found in the GDELT index")


def parse_batch(payload, name, observed):
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("Observation time must include a timezone")
    batch = gdelt_time(name[:14])
    if batch > observed:
        raise ValueError("Future GDELT batch")
    stats = {"rows": 0, "web_rows": 0, "missing_title": 0, "rejected_rows": [],
             "future_publications": []}
    records = []
    with ZipFile(BytesIO(payload)) as archive:
        if archive.namelist() != [name[:-4]]:
            raise ValueError("Unexpected GKG archive member")
        with archive.open(name[:-4]) as member:
            for raw in member:
                stats["rows"] += 1
                row = raw.rstrip(b"\r\n").decode("utf-8", errors="replace").split("\t")
                if len(row) != 27 or row[1] != name[:14]:
                    stats["rejected_rows"].append({"row": stats["rows"], "reason":
                        "column count" if len(row) != 27 else "wrong batch timestamp"})
                    continue
                if row[2] != "1":  # GKG source collection 1 = web
                    continue
                stats["web_rows"] += 1
                extras = row[26]
                match = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", extras)
                if not match or not match[1].strip():
                    stats["missing_title"] += 1
                    continue
                title = " ".join(unescape(match[1]).split())
                tickers = [key for key, pattern in HEADLINES.items() if pattern.search(title)]
                if not tickers:
                    continue
                precise = re.search(r"<PAGE_PRECISEPUBTIMESTAMP>(.*?)</PAGE_PRECISEPUBTIMESTAMP>", extras)
                try:
                    publication = gdelt_time(precise[1]) if precise else None
                    parsed = urlsplit(row[4])
                    if not row[3].strip():
                        raise ValueError("Missing publisher")
                    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                        raise ValueError("Invalid article URL")
                except ValueError as error:
                    stats["rejected_rows"].append({"row": stats["rows"], "tickers": tickers, "reason": str(error)})
                    continue
                # A later publication is ineligible at this observation. Preserve
                # the archive row for audit without admitting it to the article ledger.
                if publication is not None and publication > observed:
                    stats["future_publications"].append({
                        "row": stats["rows"], "tickers": tickers, "published_at": publication.isoformat(),
                    })
                    continue
                published = publication.isoformat() if publication else None
                # GKG column 16 (V2Tone) begins with signed full-document tone on [-100, 100].
                try:
                    tone = float(row[15].split(",", 1)[0])
                except ValueError:
                    tone = None
                if tone is not None and (not math.isfinite(tone) or abs(tone) > 100):
                    tone = None
                quotes = []
                for block in row[22].split("#"):
                    parts = block.split("|", 3)
                    if len(parts) == 4 and parts[0].isdigit():
                        quote = " ".join(unescape(parts[3]).split())[:1000]
                        if quote:
                            quotes.append(quote)
                for ticker in tickers:
                    excerpts = list(dict.fromkeys(q for q in quotes if HEADLINES[ticker].search(q)
                                    and not any(p.search(q) for t, p in HEADLINES.items() if t != ticker)))[:3]
                    identity = [ticker, row[3], row[4], title, published]
                    if tone is not None:
                        identity.append(tone)
                    if excerpts:
                        identity.append(excerpts)
                    records.append({
                        "version_id": sha256(json.dumps(identity).encode()).hexdigest(),
                        "ticker": ticker, "publisher_domain": row[3], "url": row[4], "title": title,
                        "source_kind": "gdelt_web_headline", "match_rule": "company_or_product_in_title",
                        "published_at": published, "gdelt_batch_at": batch.isoformat(), "article_tone": tone,
                        "company_quotes": excerpts,
                        "observed_at": observed.isoformat(), "available_at": observed.isoformat(),
                    })
    stats["rejected"] = len(stats["rejected_rows"])
    metadata_errors = sum("tickers" in row for row in stats["rejected_rows"])
    if stats["web_rows"] <= stats["missing_title"] + metadata_errors:
        raise ValueError("Batch has no usable titled web records")
    return records, stats


def fetch_archive(url, retry_until):
    """Give a delayed scheduled archive time to appear at its own URL."""
    while True:
        try:
            response = requests.get(url, timeout=(5, 30))
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            if isinstance(error, requests.HTTPError) and error.response.status_code not in {404, 408, 429, 500, 502, 503, 504}:
                raise
            if retry_until is None or datetime.now(timezone.utc) + timedelta(seconds=20) >= retry_until:
                raise
            sleep(20)


def collect(folder=DEFAULT_FOLDER, *, scheduled_at=None):
    """Sample the latest batch. Neither skipped batches nor feed failures are no-news days."""
    folder = Path(folder)
    started = datetime.now(timezone.utc)
    run = folder / "snapshots" / started.strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "failed", "started_at": started.isoformat(), "index": INDEX,
                "coverage": "One 15-minute GKG batch; conservative English company/product title matching."}
    try:
        index = requests.get(INDEX, timeout=(5, 30))
        index.raise_for_status()
        name, size, index_digest = latest_batch(index.text)
        check_time = scheduled_at or started
        slot = check_time.replace(minute=(check_time.minute // 15) * 15, second=0, microsecond=0)
        indexed = gdelt_time(name[:14])
        if scheduled_at is not None:
            manifest.update(expected_batch_at=slot.isoformat(), indexed_batch_at=indexed.isoformat())
        if indexed > check_time or (scheduled_at is not None and indexed != slot):
            name = slot.strftime("%Y%m%d%H%M%S.gkg.csv.zip")
            index_digest = None
            manifest["archive_selection"] = "requested current slot directly; ZIP CRC checked while reading"
        manifest["url"] = ARCHIVE + name
        deadline = slot + timedelta(minutes=15) if scheduled_at is not None else None
        response = fetch_archive(manifest["url"], deadline)
        observed = datetime.now(timezone.utc)
        payload = response.content
        if index_digest is not None and (len(payload) != size or md5(payload).hexdigest() != index_digest):
            raise ValueError("Archive size or checksum differs from public index")
        manifest.update(observed_at=observed.isoformat(), batch_at=gdelt_time(name[:14]).isoformat(),
                        archive_sha256=sha256(payload).hexdigest())
        records, stats = parse_batch(payload, name, observed)
        # The shared ledger writer expects a single writer, including during retries.
        with (folder / ".ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            added = append_versions(folder / "articles.jsonl", records)
        status = "ok"
        if observed - gdelt_time(name[:14]) > timedelta(hours=2):
            status = "stale"
        manifest.update(status=status, **stats, added=added,
                        matches={ticker: sum(r["ticker"] == ticker for r in records) for ticker in HEADLINES},
                        published_time_known=sum(r["published_at"] is not None for r in records))
    except (requests.RequestException, ValueError, BadZipFile) as error:
        manifest["error"] = str(error)
    temporary = run / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(run / "manifest.json")
    return manifest


def healthy_batches(folder, end):
    healthy = set()
    for path in (folder / "snapshots").glob("*/manifest.json"):
        run = json.loads(path.read_text())
        if run["status"] == "ok" and utc(run["observed_at"]) <= end:
            healthy.add(utc(run["batch_at"]))
    return healthy


def headline_activity_at(cutoff, folder=DEFAULT_FOLDER):
    """Use batch timestamps for windows and observation timestamps for availability."""
    end = utc(cutoff)
    if end.minute % 15 or end.second or end.microsecond:
        raise ValueError("Supply a quarter-hour signal cutoff including its timezone")
    folder = Path(folder)
    recent = {end - timedelta(minutes=15 * i) for i in range(1, 5)}
    previous = {stamp - timedelta(hours=1) for stamp in recent}
    healthy = healthy_batches(folder, end)
    counts_ready = recent <= healthy
    changes_ready = counts_ready and previous <= healthy
    result = {"cutoff": end.isoformat(), "recent_batches": len(recent & healthy),
              "previous_batches": len(previous & healthy), "expected_batches_per_hour": 4,
              "tickers": {ticker: {"headline_urls_1h": None, "headline_urls_delta_1h": None}
                          for ticker in HEADLINES}}
    if not counts_ready:
        return result
    # A revised headline on the same URL only counts at the first observation for that ticker.
    first_seen = {}
    with (folder / "articles.jsonl").open() as ledger:
        for line in ledger:
            row = json.loads(line)
            observed = utc(row["available_at"])
            if observed <= end:
                key = (row["ticker"], row["url"])
                entry = (observed, utc(row["gdelt_batch_at"]))
                if key not in first_seen or entry < first_seen[key]:
                    first_seen[key] = entry
    for ticker, features in result["tickers"].items():
        batches = [batch for (company, _), (_, batch) in first_seen.items() if company == ticker]
        count = sum(batch in recent for batch in batches)
        features["headline_urls_1h"] = count
        if changes_ready:
            features["headline_urls_delta_1h"] = count - sum(batch in previous for batch in batches)
    return result
