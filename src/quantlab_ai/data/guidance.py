"""Auditable earnings announcement content, independent of market outcomes."""
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd

from .context import COMPANY_CIKS


CONTENT_COLUMNS = ["guidance_numeric_revenue", "guidance_raised", "guidance_lowered", "guidance_withdrawn"]
GROWTH_COLUMNS = ["guidance_forward_growth_known", "guidance_forward_growth"]
EVENT_COLUMNS = ["earnings_release_recent"] + CONTENT_COLUMNS + GROWTH_COLUMNS
LEGAL_START = re.compile(
    r"(?im)^(?:forward[- ]looking statements\s*$|safe harbor\s*$|"
    r"(?:this|the) (?:press |news )?release contains forward[- ]looking statements|"
    r"(?:certain statements in this press release|statements in this press release that are not historical))")


def filing_index(root):
    """Use the existing hashed SEC index, never newer market outcomes."""
    folder = Path(root) / "data/context_v4"
    manifest = json.loads((folder / "manifest.json").read_text())
    events = []
    for source in manifest["sources"]:
        if source["kind"] != "earnings_filings":
            continue
        ticker = source["ticker"]
        for item in source["sources"]:
            path = folder / item["file"]
            if sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Cached SEC index hash mismatch")
            payload = json.loads(path.read_text())
            data = payload.get("filings", {}).get("recent", payload)
            for i, form in enumerate(data["form"]):
                date = data["filingDate"][i]
                if form != "8-K" or "2.02" not in {x.strip() for x in data["items"][i].split(",")} or not "2018-01-01" <= date <= "2026-05-22":
                    continue
                accession = data["accessionNumber"][i]
                base = f"https://www.sec.gov/Archives/edgar/data/{int(COMPANY_CIKS[ticker])}/{accession.replace('-', '')}/"
                events.append({"ticker": ticker, "accession": accession, "filing_date": date,
                               "accepted_at": data["acceptanceDateTime"][i],
                               "available_date": str((pd.Timestamp(date) + pd.Timedelta(days=1)).date()),
                               "primary_url": base + data["primaryDocument"][i]})
    frame = pd.DataFrame(events).drop_duplicates(["ticker", "accession"])
    return frame.sort_values(["ticker", "filing_date", "accession"]).to_dict("records")


def load_documents(root):
    """Require the entire predefined index and verify every saved document."""
    folder = Path(root) / "data/guidance_v11"
    manifest = json.loads((folder / "source_manifest.json").read_text())
    indexed = {r["accession"]: r for r in filing_index(root)}
    records = manifest["records"]
    if len(records) != len(indexed) or {r["accession"] for r in records} != set(indexed):
        raise ValueError("Announcement coverage does not match the full predefined filing index")
    extracted = []
    for record in records:
        reference = indexed[record["accession"]]
        if any(record[k] != v for k, v in reference.items()):
            raise ValueError("Announcement provenance differs from SEC metadata")
        if record["status"] != "complete":
            raise ValueError("Incomplete announcement coverage")
        expected_prefix = record["primary_url"].rsplit("/", 1)[0] + "/"
        if not record["document_url"].startswith(expected_prefix):
            raise ValueError("Exhibit does not belong to the indexed filing")
        path = folder / record["file"]
        if path.parent != folder or sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError("Announcement snapshot hash mismatch")
        extracted.append({**record, **extract_guidance(path.read_text())})
    return sorted(extracted, key=lambda row: (row["ticker"], row["filing_date"], row["accession"])), manifest


def extract_guidance(text):
    """Literal, conservative mentions, not sentiment or inferred revisions.

    An unflagged document means no supported phrase was detected, not that
    management never gave guidance elsewhere (for example on its earnings call).
    """
    text = text.replace("\u00a0", " ").replace("’", "'")
    stop = LEGAL_START.search(text)
    narrative = text[:stop.start()] if stop else text
    # Do not match across sentences; decimal points remain part of a sentence.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", narrative)
    result = {column: 0 for column in CONTENT_COLUMNS}
    evidence = {column: [] for column in CONTENT_COLUMNS}
    patterns = {
        "guidance_raised": r"\b(?:rais(?:e[sd]?|ing)|increas(?:e[sd]?|ing))\s+(?:(?:our|its|the|full-year|quarterly|annual|fiscal|revenue|sales|earnings|profit|eps|\d{4})\s+){0,6}(?:guidance|outlook)\b",
        "guidance_lowered": r"\b(?:lower(?:s|ed|ing)?|reduc(?:e[sd]?|ing)|cut(?:s|ting)?)\s+(?:(?:our|its|the|full-year|quarterly|annual|fiscal|revenue|sales|earnings|profit|eps|\d{4})\s+){0,6}(?:guidance|outlook)\b",
        "guidance_withdrawn": r"\b(?:withdraw(?:s|n|ing)?|withdrew|suspend(?:s|ed|ing)?)\s+(?:(?:our|its|the|full-year|quarterly|annual|fiscal|revenue|sales|earnings|profit|eps|\d{4})\s+){0,6}(?:guidance|outlook)\b",
    }
    numeric = re.compile(r"\brevenue\s+(?:(?:is\s+)?expected\s+to\s+be\s+|of\s+)?(?:between\s+|approximately\s+|about\s+)?\$[\d,.]+(?:\s*(?:to|and|[-–])\s*\$?[\d,.]+)?\s*(?:billion|million)\b", re.I)
    context = ""
    for sentence in sentences:
        clean = re.sub(r"\s+", " ", sentence).strip()
        if not clean:
            continue
        lower = clean.lower()
        # Numeric revenue only in an explicit outlook context or expected-revenue sentence.
        forward = (re.search(r"\b(?:outlook|guidance)\b", context[-300:])
                   or re.search(r"revenue (?:is )?expected to be", lower))
        match = numeric.search(clean) if forward else None
        if match:
            result["guidance_numeric_revenue"] = 1
            evidence["guidance_numeric_revenue"].append(match.group(0))
        for column, pattern in patterns.items():
            for match in re.finditer(pattern, clean, re.I):
                prefix = clean[max(0, match.start()-65):match.start()].lower()
                # Reject negated, hypothetical and attribution-to-expectations claims.
                if re.search(r"\b(?:not|never|no|may|might|could|would|will|if|expect(?:s|ed)? to|plans? to)\b", prefix):
                    continue
                result[column] = 1
                evidence[column].append(match.group(0))
        context = (context + " " + lower)[-500:]
    # Normalize a next-quarter revenue outlook by the just-reported quarter.
    # This measures management's sequential outlook, NOT an analyst surprise
    # and NOT a revision against a forecast for a different fiscal period.
    flat = re.sub(r"\s+", " ", narrative)
    amount = r"\$([\d,.]+)\s*(billion|million)"
    reported = re.search(
        r"(?:posted|reported)\s+(?:an? all-time |record )?(?:quarterly )?revenue(?: record)?"
        r"(?: for the (?:first|second|third|fourth) quarter ended [^$]{0,80}?)?[, ]*(?:of |was )?" + amount,
        flat[:2500], re.I)
    if not reported and re.search(r"results for the quarter ended", flat[:1500], re.I):
        reported = re.search(r"\bRevenue was " + amount, flat[:1500], re.I)
    outlook = re.search(r"(?:outlook|guidance) for (?:the |its )?(?:fiscal \d{4} )?(?:first|second|third|fourth) quarter", flat, re.I)
    forward = None
    if outlook:
        section = flat[outlook.end():outlook.end()+900]
        forward = re.search(r"\brevenue\s+(?:(?:is )?expected to be |of )?(?:between |approximately |about )?"
                            + amount + r"(?:\s*(?:and|to|[-–])\s*" + amount + r")?", section, re.I)
    result.update(guidance_forward_growth_known=0, guidance_forward_growth=0.0)
    if reported and forward:
        def value(number, unit):
            return float(number.replace(",", "")) * (1000 if unit.lower() == "billion" else 1)
        actual = value(reported.group(1), reported.group(2))
        low = value(forward.group(1), forward.group(2))
        high = value(forward.group(3), forward.group(4)) if forward.group(3) else low
        if actual > 0 and 0 < low <= high:
            result.update(guidance_forward_growth_known=1, guidance_forward_growth=((low+high)/2)/actual-1)
            evidence["guidance_forward_growth"] = [reported.group(0), forward.group(0)]
    return {**result, "evidence": evidence, "extraction_method": "literal_guidance_patterns_v1"}


def build_event_features(history, records):
    """All original rows retained; event window is five completed signal sessions."""
    result = history.copy()
    dates = pd.to_datetime(result.date)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Signal dates must be unique and chronological")
    for column in EVENT_COLUMNS:
        result[column] = 0.0 if column == "guidance_forward_growth" else 0
    if result.ticker.nunique() != 1:
        raise ValueError("Build one company's features at a time")
    ticker = result.ticker.iloc[0]
    for record in records:
        if record["ticker"] != ticker:
            continue
        if record.get("status") != "complete":
            raise ValueError("Missing or partial earnings documents: study cannot proceed")
        available = pd.Timestamp(record["available_date"])
        if available <= pd.Timestamp(record["filing_date"]):
            raise ValueError("Availability must be after the filing date")
        # Use signal sessions after publication, including weekends/holidays correctly.
        first = dates.searchsorted(available)
        # Older events before the start of the feature frame must not be revived.
        if available < dates.iloc[0]:
            continue
        for col, value in [("earnings_release_recent", 1)] + [(c, record[c]) for c in CONTENT_COLUMNS + GROWTH_COLUMNS]:
            if col != "guidance_forward_growth" and value not in (0, 1):
                raise ValueError("Event flags must be binary")
            if col == "guidance_forward_growth" and (not pd.notna(value) or not -1 < value < 100):
                raise ValueError("Invalid forward revenue growth")
            indices = result.index[first:first+5]
            result.loc[indices, col] = value
    return result
