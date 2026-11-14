"""Records the data layer stores.

Every record carries two times:

- ``observed_at``: when the price (or event) was true in the world.
- ``available_at``: when a forecaster could first have known it. Agents only
  ever see records with ``available_at <= t`` (see ``PriceStore.as_of``).

For a page snapshot both are the snapshot's crawl time. For an article, the
quoted price may be older than the article, but it only became knowable when
the article was published.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

GPUS = ("H100", "H200", "B200", "A100")
PRICING_TYPES = ("on_demand", "spot", "reserved")
SOURCE_KINDS = ("page_snapshot", "live_page", "article")


def parse_time(value: str | datetime | None) -> datetime | None:
    """Parse an ISO 8601 string (or a date) into an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class PriceObservation:
    gpu: str                   # one of GPUS
    provider: str              # e.g. "lambda", normalised from the domain
    pricing_type: str          # one of PRICING_TYPES
    usd_per_gpu_hour: float    # per single GPU, even if the offer is an 8-GPU node
    observed_at: datetime
    available_at: datetime
    source_url: str
    source_kind: str           # one of SOURCE_KINDS
    gpus_per_instance: int = 1
    variant: str = ""          # e.g. "SXM 80GB", kept for analysis
    raw: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple:
        """Identity used for de-duplication: the same quote seen twice is one record."""
        return (self.gpu, self.provider, self.pricing_type, round(self.usd_per_gpu_hour, 4),
                self.variant, self.observed_at.date().isoformat())

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["observed_at"] = iso(self.observed_at)
        row["available_at"] = iso(self.available_at)
        return row

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "PriceObservation":
        row = dict(row)
        row["observed_at"] = parse_time(row["observed_at"])
        row["available_at"] = parse_time(row["available_at"])
        return cls(**row)


@dataclass
class Event:
    event_id: str              # stable ID agents cite, e.g. "N3f2a…"
    title: str
    url: str
    published_at: datetime     # also the time it became available
    snippet: str = ""
    query: str = ""

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["published_at"] = iso(self.published_at)
        return row

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Event":
        row = dict(row)
        row["published_at"] = parse_time(row["published_at"])
        return cls(**row)
