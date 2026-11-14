"""Collect GPU prices and events from Exa into data/.

    python scripts/collect.py discover                       # find pricing pages, update configs/sources.yaml
    python scripts/collect.py history --start 2025-06-01 --every-days 7
    python scripts/collect.py live                           # run on a schedule during the event
    python scripts/collect.py articles --start 2025-06-01 --every-days 30
    python scripts/collect.py events --start 2025-06-01 --every-days 30
    python scripts/collect.py summary                        # counts and daily medians

Every command except ``summary`` and ``discover`` accepts ``--dry-run``,
which prints how many Exa requests it would make and stops.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr.data import GPUS, PRICING_TYPES, PriceStore  # noqa: E402
from gpr.data.exa_source import ExaSource  # noqa: E402
from gpr.data.schema import parse_time  # noqa: E402


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def date_range(start: datetime, end: datetime, every_days: int) -> list[datetime]:
    out, day = [], start
    while day < end:
        out.append(day)
        day += timedelta(days=every_days)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["discover", "history", "live", "articles", "events", "summary"])
    parser.add_argument("--sources", default=str(ROOT / "configs" / "sources.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--start", help="ISO date, e.g. 2025-06-01")
    parser.add_argument("--end", help="ISO date; default now")
    parser.add_argument("--every-days", type=int, default=7)
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources_path = Path(args.sources)
    sources = yaml.safe_load(sources_path.read_text()) or {}
    urls = list(dict.fromkeys((sources.get("seed_urls") or []) + (sources.get("discovered_urls") or [])))
    store = PriceStore.load(args.data)

    if args.command == "summary":
        print(f"{len(store)} price observations, {len(store.events())} events in {args.data}")
        for gpu in GPUS:
            for kind in PRICING_TYPES:
                series = store.daily_median(gpu, kind, carry_days=args.every_days)
                rows = store.prices(gpu, kind)
                if rows:
                    days = sorted(series)
                    print(f"  {gpu:5s} {kind:9s} {len(rows):5d} obs, {len({r.provider for r in rows})} providers, "
                          f"{days[0]} .. {days[-1]}, latest median ${series[days[-1]]:.2f}/GPU-h")
        return

    now = datetime.now(timezone.utc)
    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else now
    if args.command in {"history", "articles", "events"} and start is None:
        parser.error(f"{args.command} needs --start")
    windows = date_range(start, end, args.every_days) if start else []

    planned = {
        "discover": len(sources.get("discovery_queries") or []) or 4,
        "history": len(windows),  # one get_contents call per date, covering every URL
        "live": 1,
        "articles": len(windows) * len(sources.get("article_queries") or []),
        "events": len(windows) * len(sources.get("event_queries") or []),
    }[args.command]
    print(f"{args.command}: about {planned} Exa requests"
          + (f" for {len(urls)} pages" if args.command in {"history", "live"} else ""))
    if args.dry_run:
        return

    load_env()
    if not os.environ.get("EXA_API_KEY"):
        sys.exit("EXA_API_KEY is not set: copy .env.example to .env and add the key from the Exa dashboard.")
    source = ExaSource()

    added = 0
    if args.command == "discover":
        found = source.discover_pricing_pages(num_results=args.num_results)
        known = set(urls)
        new = [u for u in found if u not in known]
        sources["discovered_urls"] = (sources.get("discovered_urls") or []) + new
        sources_path.write_text(yaml.safe_dump(sources, sort_keys=False))
        print(f"found {len(found)} pages, {len(new)} new; review them in {sources_path}")
    elif args.command == "history":
        added = sum(store.add_price(p) for p in source.snapshot_prices(urls, windows))
    elif args.command == "live":
        added = sum(store.add_price(p) for p in source.live_prices(urls))
    elif args.command == "articles":
        for i, window in enumerate(windows):
            stop = windows[i + 1] if i + 1 < len(windows) else end
            for query in sources.get("article_queries") or []:
                added += sum(store.add_price(p) for p in source.article_prices(query, window, stop, args.num_results))
    elif args.command == "events":
        for i, window in enumerate(windows):
            stop = windows[i + 1] if i + 1 < len(windows) else end
            for query in sources.get("event_queries") or []:
                added += sum(store.add_event(e) for e in source.events(query, window, stop, args.num_results))

    if args.command != "discover":
        store.save(args.data)
        print(f"added {added} new records; store now has {len(store)} prices and {len(store.events())} events")
    print(f"Exa calls: {source.log.calls}, failures: {source.log.failures}")
    for note in source.log.notes[-5:]:
        print(f"  {note}")


if __name__ == "__main__":
    main()
