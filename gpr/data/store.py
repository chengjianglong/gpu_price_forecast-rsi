"""The time-sliced store every agent reads from.

``PriceStore.as_of(t)`` returns a view containing only what was knowable at
``t``. Agents get that view and nothing else, which is how the backtest stays
free of look-ahead.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from .schema import Event, PriceObservation, parse_time


class PriceStore:
    def __init__(self, prices: Iterable[PriceObservation] = (), events: Iterable[Event] = (),
                 cutoff: datetime | None = None):
        self._prices: dict[tuple, PriceObservation] = {}
        self._events: dict[str, Event] = {}
        self.cutoff = cutoff
        for p in prices:
            self.add_price(p)
        for e in events:
            self.add_event(e)

    # ---- writing ---------------------------------------------------------
    def add_price(self, obs: PriceObservation) -> bool:
        """Add an observation; return False if it was already there.

        When the same quote arrives twice, keep the copy that became available
        first: that is when a forecaster could really have known it.
        """
        self._check_writable()
        key = obs.key()
        current = self._prices.get(key)
        if current is None or obs.available_at < current.available_at:
            self._prices[key] = obs
            return current is None
        return False

    def add_event(self, event: Event) -> bool:
        self._check_writable()
        if event.event_id in self._events:
            return False
        self._events[event.event_id] = event
        return True

    def _check_writable(self) -> None:
        if self.cutoff is not None:
            raise RuntimeError("an as_of() view is read-only")

    # ---- reading ---------------------------------------------------------
    def as_of(self, t: datetime | str) -> "PriceStore":
        """A read-only view with only the records available at or before ``t``."""
        t = parse_time(t)
        if t is None:
            raise ValueError("as_of needs a valid time")
        if self.cutoff is not None and t > self.cutoff:
            raise ValueError("cannot widen a view past its own cutoff")
        view = PriceStore()
        view._prices = {k: p for k, p in self._prices.items() if p.available_at <= t}
        view._events = {k: e for k, e in self._events.items() if e.published_at <= t}
        view.cutoff = t
        return view

    def prices(self, gpu: str | None = None, pricing_type: str | None = None) -> list[PriceObservation]:
        rows = [p for p in self._prices.values()
                if (gpu is None or p.gpu == gpu) and (pricing_type is None or p.pricing_type == pricing_type)]
        return sorted(rows, key=lambda p: (p.observed_at, p.provider))

    def events(self) -> list[Event]:
        return sorted(self._events.values(), key=lambda e: e.published_at)

    def daily_median(self, gpu: str, pricing_type: str, carry_days: int = 0) -> dict[date, float]:
        """Median per-GPU price across providers for each day.

        A provider's last quote counts for up to ``carry_days`` days after it
        was observed, because list prices persist between snapshots. With
        ``carry_days=0`` only same-day quotes count.
        """
        latest: dict[str, list[PriceObservation]] = defaultdict(list)
        for p in self.prices(gpu, pricing_type):
            latest[p.provider].append(p)
        if not latest:
            return {}
        days = sorted({p.observed_at.date() for ps in latest.values() for p in ps})
        last_day = days[-1] + timedelta(days=carry_days)
        if self.cutoff is not None:
            last_day = min(last_day, self.cutoff.date())
        out: dict[date, float] = {}
        day = days[0]
        while day <= last_day:
            quotes = []
            for ps in latest.values():
                live = [p for p in ps if 0 <= (day - p.observed_at.date()).days <= carry_days]
                if live:
                    newest = max(live, key=lambda p: p.observed_at)
                    same_time = [p.usd_per_gpu_hour for p in live if p.observed_at == newest.observed_at]
                    quotes.append(min(same_time))  # a provider's cheapest variant at that time
            if quotes:
                out[day] = statistics.median(quotes)
            day += timedelta(days=1)
        return out

    def __len__(self) -> int:
        return len(self._prices)

    # ---- files -----------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        self._check_writable()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_jsonl(directory / "prices.jsonl", (p.to_json() for p in self.prices()))
        _write_jsonl(directory / "events.jsonl", (e.to_json() for e in self.events()))

    @classmethod
    def load(cls, directory: str | Path) -> "PriceStore":
        directory = Path(directory)
        prices = [PriceObservation.from_json(r) for r in _read_jsonl(directory / "prices.jsonl")]
        events = [Event.from_json(r) for r in _read_jsonl(directory / "events.jsonl")]
        return cls(prices, events)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
