"""Build the price and event data from Exa.

Three ways prices come in:

1. **Page snapshots (the history).** ``get_contents(url, snapshot_as_of=t)``
   returns the newest stored version of a provider's pricing page at or
   before ``t``. Asking for a series of past dates rebuilds that page's price
   history. The snapshot's own crawl time (``snapshot_at``) is used as both
   ``observed_at`` and ``available_at``; results without it are dropped,
   because their timing can't be trusted.
2. **Live pages (going forward).** The same pages fetched fresh
   (``max_age_hours=0``), run on a schedule during the event.
3. **Articles.** News and blog posts that quote prices, found with
   published-date filters. Their prices are evidence for the agents, not the
   forecast target, because the quoted price may be older than the article.

Prices are pulled out of each page by Exa's structured summary
(``summary={"query": ..., "schema": ...}``) and then checked by ``clean.py``.
Events (launches, price cuts, supply news) come from news search and feed the
shared board's evidence pack.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from .clean import clean_quote, provider_from_url
from .schema import Event, PriceObservation, iso, parse_time

QUOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "quotes": {
            "type": "array",
            "description": "Every GPU rental price listed on the page.",
            "items": {
                "type": "object",
                "properties": {
                    "gpu": {"type": "string", "description": "GPU model as written, e.g. 'H100 SXM 80GB'"},
                    "price_usd": {"type": "number", "description": "Price in US dollars"},
                    "price_unit": {
                        "type": "string",
                        "enum": ["per_gpu_hour", "per_instance_hour", "per_gpu_month", "per_instance_month"],
                    },
                    "gpus_per_instance": {"type": "integer", "description": "GPUs in the priced instance; 1 if per GPU"},
                    "pricing_type": {
                        "type": "string",
                        "enum": ["on_demand", "spot", "reserved"],
                        "description": "spot also covers interruptible or community-cloud prices",
                    },
                },
                "required": ["gpu", "price_usd", "price_unit"],
            },
        }
    },
    "required": ["quotes"],
}

QUOTE_QUERY = (
    "List every NVIDIA A100, H100, H200 and B200 rental price on this page, "
    "with its unit, the number of GPUs it covers, and whether it is on-demand, spot or reserved."
)

DEFAULT_DISCOVERY_QUERIES = (
    "H100 GPU cloud pricing per hour",
    "H200 GPU cloud rental price per hour",
    "B200 GPU cloud pricing on-demand",
    "A100 80GB cloud GPU pricing per hour",
)

DEFAULT_EVENT_QUERIES = (
    "GPU cloud price cut H100",
    "NVIDIA B200 shipments supply data center",
    "H100 rental prices falling",
    "GPU shortage cloud capacity",
)


@dataclass
class CallLog:
    calls: int = 0
    failures: int = 0
    notes: list[str] = field(default_factory=list)


def parse_summary(summary: str | None) -> list[dict[str, Any]]:
    """Read the quotes list out of a structured summary string."""
    if not summary:
        return []
    text = summary.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    quotes = data.get("quotes") if isinstance(data, dict) else data
    return [q for q in quotes if isinstance(q, dict)] if isinstance(quotes, list) else []


def event_id(url: str) -> str:
    return "N" + hashlib.sha1(url.encode()).hexdigest()[:8]


class ExaSource:
    """Wraps an ``exa_py.Exa`` client (or any object with the same methods)."""

    def __init__(self, client: Any | None = None, retries: int = 3, pause_s: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] | None = None):
        if client is None:
            from exa_py import Exa  # reads EXA_API_KEY

            client = Exa()
        self.client = client
        self.retries = retries
        self.pause_s = pause_s
        self.sleep = sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.log = CallLog()

    # ---- plumbing --------------------------------------------------------
    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any | None:
        for attempt in range(1, self.retries + 1):
            self.log.calls += 1
            try:
                result = fn(*args, **kwargs)
                if self.pause_s:
                    self.sleep(self.pause_s)
                return result
            except Exception as error:  # the SDK raises several types; all are retried the same way
                self.log.failures += 1
                self.log.notes.append(f"{getattr(fn, '__name__', 'call')} attempt {attempt}: {error}")
                if attempt < self.retries:
                    self.sleep(2 ** attempt)
        return None

    def _observations(self, result: Any, observed: datetime, available: datetime,
                      kind: str) -> list[PriceObservation]:
        out = []
        for quote in parse_summary(getattr(result, "summary", None)):
            clean = clean_quote(quote)
            if clean is None:
                continue
            out.append(PriceObservation(
                provider=provider_from_url(result.url),
                observed_at=observed,
                available_at=available,
                source_url=result.url,
                source_kind=kind,
                raw=quote,
                **clean,
            ))
        return out

    # ---- discovery -------------------------------------------------------
    def discover_pricing_pages(self, queries: Sequence[str] = DEFAULT_DISCOVERY_QUERIES,
                               num_results: int = 10, exclude_domains: Sequence[str] = ()) -> list[str]:
        """Find candidate pricing pages; one URL per domain, in relevance order."""
        seen: dict[str, str] = {}
        for query in queries:
            response = self._call(self.client.search, query, num_results=num_results, contents=False,
                                  exclude_domains=list(exclude_domains) or None)
            for result in getattr(response, "results", None) or []:
                provider = provider_from_url(result.url)
                if provider not in seen:
                    seen[provider] = result.url
        return list(seen.values())

    # ---- prices ----------------------------------------------------------
    def snapshot_prices(self, urls: Sequence[str], dates: Iterable[datetime]) -> list[PriceObservation]:
        """Price history from stored snapshots of each page at each date."""
        summary = {"query": QUOTE_QUERY, "schema": QUOTE_SCHEMA}
        out: list[PriceObservation] = []
        for when in dates:
            response = self._call(self.client.get_contents, list(urls), summary=summary,
                                  snapshot_as_of=iso(when), filter_empty_results=True)
            for result in getattr(response, "results", None) or []:
                taken = parse_time(getattr(result, "snapshot_at", None))
                if taken is None or taken > when:
                    continue  # no trustworthy snapshot time
                out += self._observations(result, taken, taken, "page_snapshot")
        return out

    def live_prices(self, urls: Sequence[str]) -> list[PriceObservation]:
        """Fresh prices from each page, timestamped now."""
        summary = {"query": QUOTE_QUERY, "schema": QUOTE_SCHEMA}
        response = self._call(self.client.get_contents, list(urls), summary=summary, max_age_hours=0,
                              filter_empty_results=True)
        now = self.now()
        out: list[PriceObservation] = []
        for result in getattr(response, "results", None) or []:
            out += self._observations(result, now, now, "live_page")
        return out

    def article_prices(self, query: str, start: datetime, end: datetime,
                       num_results: int = 10) -> list[PriceObservation]:
        """Prices quoted in articles published in [start, end)."""
        response = self._call(
            self.client.search, query, num_results=num_results,
            start_published_date=iso(start), end_published_date=iso(end),
            contents={"summary": {"query": QUOTE_QUERY, "schema": QUOTE_SCHEMA}},
        )
        out: list[PriceObservation] = []
        for result in getattr(response, "results", None) or []:
            published = parse_time(getattr(result, "published_date", None))
            if published is None or not (start <= published < end):
                continue
            out += self._observations(result, published, published, "article")
        return out

    # ---- events ----------------------------------------------------------
    def events(self, query: str, start: datetime, end: datetime, num_results: int = 10) -> list[Event]:
        response = self._call(
            self.client.search, query, num_results=num_results, category="news",
            start_published_date=iso(start), end_published_date=iso(end),
            contents={"highlights": True},
        )
        out = []
        for result in getattr(response, "results", None) or []:
            published = parse_time(getattr(result, "published_date", None))
            if published is None or not (start <= published < end):
                continue
            highlights = getattr(result, "highlights", None) or []
            out.append(Event(
                event_id=event_id(result.url),
                title=getattr(result, "title", None) or "",
                url=result.url,
                published_at=published,
                snippet=" … ".join(highlights)[:1200],
                query=query,
            ))
        return out
