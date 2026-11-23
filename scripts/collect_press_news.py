"""Sample the latest GDELT batch: PYTHONPATH=src .venv/bin/python scripts/collect_press_news.py."""
import argparse
import json

from quantlab_ai.data.press_news import collect, headline_activity_at


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="Read saved headline counts and hour-over-hour changes at this ISO cutoff")
    args = parser.parse_args()
    if args.as_of:
        print(json.dumps(headline_activity_at(args.as_of), indent=2))
        raise SystemExit(0)
    manifest = collect()
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0 if manifest["status"] == "ok" else 1)
