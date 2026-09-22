"""Public company feeds for prospective research, never a historical backfill."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import requests


FEEDS = {
    "AAPL": "https://www.apple.com/newsroom/rss-feed.rss",
    "MSFT": "https://news.microsoft.com/feed/",
    "NVDA": "https://nvidianews.nvidia.com/releases.xml",
}
ATOM = "{http://www.w3.org/2005/Atom}"


def utc(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        stamp = parsedate_to_datetime(value)
    if stamp.tzinfo is None:
        raise ValueError("News timestamps must include a timezone")
    return stamp.astimezone(timezone.utc)


def parse_feed(payload, ticker, observed_at):
    """Return accepted versions and rejection counts; retain no article bodies."""
    observed = utc(observed_at)
    root = ET.fromstring(payload)
    atom = root.tag == ATOM + "feed"
    entries = root.findall(ATOM + "entry") if atom else root.findall("./channel/item")
    if not atom and root.tag != "rss":
        raise ValueError("Unsupported feed format")
    records, rejected = [], 0
    for entry in entries:
        prefix = ATOM if atom else ""
        title = (entry.findtext(prefix + "title") or "").strip()
        if atom:
            links = entry.findall(ATOM + "link")
            url = next((e.get("href", "") for e in links
                        if e.get("rel", "alternate") == "alternate"), "")
            published = entry.findtext(ATOM + "published") or entry.findtext(ATOM + "updated")
            updated = entry.findtext(ATOM + "updated") or published
        else:
            url = (entry.findtext("link") or "").strip()
            published = entry.findtext("pubDate")
            updated = published
        try:
            publication, revision = utc(published or ""), utc(updated or "")
            if not title or not url.startswith(("https://", "http://")):
                raise ValueError("Missing headline or URL")
            if max(publication, revision) > observed:
                raise ValueError("Future publication timestamp")
        except (ValueError, TypeError, OverflowError):
            rejected += 1
            continue
        version = [ticker, url, title, publication.isoformat(), revision.isoformat()]
        records.append({
            "version_id": sha256(json.dumps(version).encode()).hexdigest(),
            "ticker": ticker, "title": title, "url": url,
            "source_kind": "company_press_release",
            "published_at": publication.isoformat(), "updated_at": revision.isoformat(),
            "observed_at": observed.isoformat(), "available_at": observed.isoformat(),
        })
    return records, rejected


def append_versions(path, records):
    """Single-writer ledger: preserve first observation and retain revisions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = {json.loads(line)["version_id"] for line in path.read_text().splitlines()} if path.exists() else set()
    added = 0
    with path.open("a") as handle:
        for row in records:
            if row["version_id"] not in known:
                handle.write(json.dumps(row) + "\n")
                known.add(row["version_id"])
                added += 1
    return added


def available_before(records, cutoff):
    cutoff = utc(cutoff)
    return [row for row in records if utc(row["available_at"]) <= cutoff]


def feed_health(records, observed_at, rejected=0):
    """A 30-day inactivity flag is a collection warning, not a trading signal."""
    if not records:
        return {"status": "empty", "newest_publication": None}
    newest = max(utc(row["published_at"]) for row in records)
    age = (utc(observed_at) - newest).total_seconds() / 86400
    return {"status": "stale" if age > 30 else ("partial" if rejected else "ok"),
            "newest_publication": newest.isoformat(), "newest_age_days": round(age, 2)}


def collect(folder=Path("data/news_v9")):
    """Store raw evidence and explicit failures. Run with one collector at a time."""
    folder = Path(folder)
    run = folder / "snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    manifest = []
    for ticker, url in FEEDS.items():
        result = {"ticker": ticker, "url": url, "status": "failed"}
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "QuantLabAI research feed reader"})
            observed = datetime.now(timezone.utc).isoformat()
            result.update(http_status=response.status_code, observed_at=observed)
            response.raise_for_status()
            payload = response.content
            (run / (ticker + ".xml")).write_bytes(payload)
            result.update(sha256=sha256(payload).hexdigest(), raw_file=ticker + ".xml")
            records, rejected = parse_feed(payload, ticker, observed)
            added = append_versions(folder / "articles.jsonl", records)
            result.update(accepted=len(records), rejected=rejected, added=added)
            result.update(feed_health(records, observed, rejected))
        except (requests.RequestException, ValueError, ET.ParseError) as error:
            result["error"] = str(error)
        manifest.append(result)
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
