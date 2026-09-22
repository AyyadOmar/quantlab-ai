"""Export a small public snapshot from completed research, without private paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "baseline": "backtesting/open_to_close_v2/baseline_comparison.md",
    "features": "backtesting/feature_study_v3/feature_study.md",
    "context": "backtesting/context_study_v4/context_study.md",
    "earnings": "backtesting/earnings_study_v5/earnings_study.md",
    "direction": "backtesting/direction_study_v6/direction_study.md",
    "shared": "backtesting/pooled_direction_study_v7/pooled_direction_study.md",
    "protocol": "docs/research_protocol.md",
    "recent": "backtesting/recent_history_study_v8/recent_history_study.md",
    "confirmation": "backtesting/recent_history_study_v8/confirmation.md",
    "intraday": "docs/intraday_study.md",
    "news": "docs/news_research.md",
    "diagnostics": "docs/direction_diagnostic_audit.md",
}


def public_row(row):
    fields = ("accuracy", "always_up_accuracy", "accuracy_gain", "rows", "down_calls",
              "correct_down", "incorrect_down", "extra_correct_vs_always_up", "fixed_cutoff_accuracy")
    result = {key: row[key] for key in fields}
    result["always_down_accuracy"] = 1 - result["always_up_accuracy"]
    result["accuracy_gain_vs_always_down"] = result["accuracy"] - result["always_down_accuracy"]
    result["interval"] = row["uncertainty"]["interval_95"]
    assert result["rows"] > 0
    assert result["correct_down"] + result["incorrect_down"] == result["down_calls"]
    assert result["correct_down"] - result["incorrect_down"] == result["extra_correct_vs_always_up"]
    assert abs(result["accuracy"] - result["always_up_accuracy"] - result["accuracy_gain"]) < 1e-10
    return result


def export():
    destination = ROOT / "public/research"
    (destination / "reports").mkdir(parents=True, exist_ok=True)
    definitions = [
        ("confirmation", "Newer confirmation", "backtesting/recent_history_study_v8/confirmation.json",
         [("full", "Full-history model"), ("validation_selected", "Selected-history model")]),
        ("shared", "Shared training", "backtesting/pooled_direction_study_v7/pooled_direction_study.json",
         [("per_company", "Separate models"), ("shared_companies", "Shared model")]),
        ("direction", "Direction rules", "backtesting/direction_study_v6/direction_study.json",
         [("selected", "Selected direction rule")]),
    ]
    studies = []
    for key, title, source, scopes in definitions:
        raw = (ROOT / source).read_bytes()
        report = json.loads(raw)
        first = report["results"][0]
        confirmation = key == "confirmation"
        study = {"id": key, "title": title,
                 "evidence_type": "retrospective_confirmation" if confirmation else "historical_development",
                 "tickers": sorted({row["ticker"] for row in report["results"]}) if confirmation else report["plan"]["tickers"],
                 "sessions_per_asset": first["rows"], "start": first["execution_start"][:10],
                 "end": first["execution_end"][:10], "protocol": "recent_history_study_v8" if confirmation else report["plan"]["protocol"],
                 "source_sha256": hashlib.sha256(raw).hexdigest(),
                 "report": f"/research/reports/{key}.md", "models": {}}
        for model in ["xgboost", "logistic_regression"]:
            aggregates = []
            for scope, label in scopes:
                lookup = model if key == "direction" else f"{model}_{scope}"
                aggregates.append({"scope": scope, "label": label, **public_row(report["aggregates"][lookup])})
            assets = []
            for row in report["results"]:
                if row["model"] == model:
                    assert row["rows"] == first["rows"]
                    assert row["execution_start"][:10] == study["start"]
                    assert row["execution_end"][:10] == study["end"]
                    assets.append({"ticker": row["ticker"], "scope": row.get("scope", "selected"), **public_row(row)})
            study["models"][model] = {"aggregates": aggregates, "assets": assets}
        studies.append(study)
    snapshot = {"schema_version": 2, "status": "research_snapshot", "studies": studies,
                "reports": {key: f"/research/reports/{key}.md" for key in REPORTS}}
    (destination / "results.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    for key, source in REPORTS.items():
        content = (ROOT / source).read_text()
        content = content.replace("](direction_diagnostic_audit.md)", "](diagnostics.md)")
        content = content.replace("](intraday_study.md)", "](intraday.md)")
        (destination / "reports" / f"{key}.md").write_text(content)
    print(f"Exported {len(studies)} studies and {len(REPORTS)} reports.")


if __name__ == "__main__":
    export()
