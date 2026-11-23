"""Collect and score news locally, or inspect what was available at a past cutoff."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

from quantlab_ai.data.news_sentiment import score_pending, sentiment_at
from quantlab_ai.data.press_news import DEFAULT_FOLDER, collect, headline_activity_at


def in_collection_window(now):
    import exchange_calendars
    session = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    calendar = exchange_calendars.get_calendar("XNYS")
    if not calendar.is_session(session):
        return False
    close = calendar.session_close(session)
    return close - timedelta(hours=2) <= now < close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--as-of", help="Read saved activity and sentiment at an ISO quarter-hour cutoff")
    modes.add_argument("--scheduled", action="store_true", help="Collect only in the last two hours of a US equity session")
    modes.add_argument("--score-pending", action="store_true", help="Score previously collected headlines")
    args = parser.parse_args()
    if args.as_of:
        result = headline_activity_at(args.as_of)
        result["sentiment"] = sentiment_at(args.as_of, DEFAULT_FOLDER, result)
    elif args.score_pending:
        result = {"new_scores": score_pending(DEFAULT_FOLDER)}
    elif args.scheduled and not in_collection_window(datetime.now(timezone.utc)):
        result = {"status": "outside_collection_window"}
    else:
        result = collect()
        result["new_scores"] = score_pending(DEFAULT_FOLDER)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {None, "ok", "outside_collection_window"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
