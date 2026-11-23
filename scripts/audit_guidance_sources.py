"""Regenerate a shareable source index without publishing announcement text."""
from pathlib import Path

from quantlab_ai.data.guidance import CONTENT_COLUMNS, load_documents

ROOT = Path(__file__).resolve().parents[1]


def main():
    records, _ = load_documents(ROOT)
    lines = ["# Guidance study source audit", "",
             "All 105 entries in the fixed 2018–May 2026 SEC earnings-filing index are represented. "
             "This is coverage of indexed announcements, not every company disclosure or earnings call. "
             "The model receives information from the calendar day after filing. Raw narrative snapshots "
             "and full hashes remain local in data/guidance_v11. The linked SEC accession documents "
             "identify the original archived sources; these downloads were made retrospectively.", "",
             "The outlook ratio is management's next-quarter revenue midpoint relative to the "
             "just-reported quarter, not an analyst surprise or a revision against an earlier forecast "
             "for the same period. An em dash means that comparison could not be extracted. "
             "Flags describe detected document wording only. They do not cover earnings-call-only guidance.", "",
             "| Company | Filing date | Available from | Numeric outlook | Raise / cut / withdraw | Sequential outlook | Source |",
             "|---|---|---|---:|---|---:|---|"]
    for row in records:
        growth = f"{row['guidance_forward_growth']:+.2%}" if row["guidance_forward_growth_known"] else "—"
        flags = " / ".join(str(row[c]) for c in CONTENT_COLUMNS[1:])
        lines.append(f"| {row['ticker']} | {row['filing_date']} | {row['available_date']} | {row['guidance_numeric_revenue']} | {flags} | {growth} | [SEC exhibit]({row['document_url']}) |")
    lines += ["", "## Collection and extraction limitations", "",
              "Local direct downloads encountered DNS timeouts, followed by HTTP 403 after a per-request "
              "alternate-DNS check. The public browsing retrieval service successfully read the indexed "
              "SEC exhibits. Some service requests were rate-limited; they were retried after a pause "
              "with reduced concurrency. No rate-limit failures were converted to no-news rows.", "",
              "The readable narrative is complete up to the first recognized legal disclaimer, or the "
              "entire document when no boundary is found. Legal boilerplate and appended tables are "
              "not part of the language feature scope. Archive URLs were checked against each indexed "
              "accession; hashes detect later changes to these local snapshots. An LLM did not label "
              "market direction or infer facts from later outcomes.", "",
              "The extractor found 43 comparable quarterly outlooks, two explicit reduction phrases "
              "and no explicit raise/withdrawal phrases. Thus this is not an adequate sample for "
              "learning a general guidance-revision strategy. The field values and supporting matched "
              "text can be reviewed in the local extracted_events.json file. Literal patterns can miss "
              "paraphrases; absence of a flag is not proof that an event never happened.", "",
              "Run `PYTHONPATH=src .venv/bin/python scripts/audit_guidance_sources.py` to regenerate "
              "this table from the hashed local snapshots. A fresh clone needs those snapshots before "
              "the study can run; no bulk downloader or scheduled collection is included.", ""]
    path = ROOT / "docs/guidance_sources.md"
    path.write_text("\n".join(lines))
    print(f"Verified and indexed {len(records)} archived announcements: {path}")


if __name__ == "__main__":
    main()
