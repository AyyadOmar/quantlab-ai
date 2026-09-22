"""Run from the project root: PYTHONPATH=src .venv/bin/python scripts/collect_public_news.py."""
import json
from quantlab_ai.data.news import collect


if __name__ == "__main__":
    results = collect()
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(row["status"] == "ok" for row in results) else 1)
