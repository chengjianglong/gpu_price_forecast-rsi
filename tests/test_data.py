"""Offline tests for the data layer: cleaning, the Exa collector (with a fake
client), and the no-look-ahead guarantee of PriceStore.as_of."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from gpr.data import PriceStore
from gpr.data.clean import clean_quote, normalise_gpu, normalise_pricing_type, provider_from_url
from gpr.data.exa_source import ExaSource, parse_summary
from gpr.data.schema import PriceObservation

UTC = timezone.utc


def day(n: int) -> datetime:
    return datetime(2025, 9, 1, tzinfo=UTC) + timedelta(days=n)


# ---- cleaning ---------------------------------------------------------------
def test_normalise_gpu_and_variants():
    assert normalise_gpu("NVIDIA H100 SXM5 80GB") == ("H100", "SXM5 80GB")
    assert normalise_gpu("h200 141GB")[0] == "H200"
    assert normalise_gpu("GH200 Grace Hopper") == (None, "")  # a different product
    assert normalise_gpu("RTX 4090") == (None, "")


def test_pricing_type_and_provider():
    assert normalise_pricing_type("Interruptible") == "spot"
    assert normalise_pricing_type("1-year commitment") == "reserved"
    assert normalise_pricing_type(None) == "on_demand"
    assert provider_from_url("https://www.lambda.ai/pricing") == "lambda"
    assert provider_from_url("https://cloud.example.co.uk/gpus") == "example"


def test_clean_quote_converts_and_rejects():
    node = clean_quote({"gpu": "H100 SXM", "price_usd": 23.92, "price_unit": "per_instance_hour",
                        "gpus_per_instance": 8})
    assert node["usd_per_gpu_hour"] == pytest.approx(2.99)
    monthly = clean_quote({"gpu": "A100", "price_usd": 1095, "price_unit": "per_gpu_month"})
    assert monthly["usd_per_gpu_hour"] == pytest.approx(1.5)
    # an 8-GPU node price mislabelled as per GPU falls outside the bounds
    assert clean_quote({"gpu": "H100", "price_usd": 23.92, "price_unit": "per_gpu_hour"}) is None
    assert clean_quote({"gpu": "H100", "price_usd": "n/a", "price_unit": "per_gpu_hour"}) is None


def test_parse_summary_tolerates_fences_and_noise():
    payload = {"quotes": [{"gpu": "H100", "price_usd": 2.5, "price_unit": "per_gpu_hour"}]}
    assert parse_summary(json.dumps(payload)) == payload["quotes"]
    assert parse_summary("```json\n" + json.dumps(payload) + "\n```") == payload["quotes"]
    assert parse_summary("Here you go: " + json.dumps(payload)) == payload["quotes"]
    assert parse_summary("no prices here") == []


# ---- a fake Exa client --------------------------------------------------------
@dataclass
class FakeResult:
    url: str
    summary: str | None = None
    snapshot_at: str | None = None
    published_date: str | None = None
    title: str | None = None
    highlights: list | None = None


@dataclass
class FakeResponse:
    results: list


class FakeExa:
    """Serves a pricing page whose H100 price drops on 2025-09-10."""

    def __init__(self):
        self.calls = []

    def get_contents(self, urls, **kwargs):
        self.calls.append(("get_contents", kwargs))
        when = datetime.fromisoformat(kwargs.get("snapshot_as_of", "2025-09-20T00:00:00Z").replace("Z", "+00:00"))
        price = 2.99 if when < day(9) else 2.49
        taken = when - timedelta(hours=6)  # the stored snapshot is a little older than requested
        summary = json.dumps({"quotes": [
            {"gpu": "H100 SXM", "price_usd": price, "price_unit": "per_gpu_hour", "pricing_type": "on_demand"},
            {"gpu": "RTX 4090", "price_usd": 0.4, "price_unit": "per_gpu_hour"},
        ]})
        return FakeResponse([FakeResult(url=u, summary=summary, snapshot_at=taken.isoformat()) for u in urls])

    def search(self, query, **kwargs):
        self.calls.append(("search", kwargs))
        if kwargs.get("category") == "news":
            return FakeResponse([
                FakeResult(url="https://news.example.com/h100-cut", title="H100 prices cut",
                           published_date=day(9).isoformat(), highlights=["Provider cuts H100 to $2.49"]),
                FakeResult(url="https://news.example.com/old", title="Out of window",
                           published_date=day(-40).isoformat()),
            ])
        return FakeResponse([FakeResult(url="https://www.lambda.ai/pricing"),
                             FakeResult(url="https://lambda.ai/other"),
                             FakeResult(url="https://www.runpod.io/pricing")])


def make_source() -> tuple[ExaSource, FakeExa]:
    fake = FakeExa()
    return ExaSource(client=fake, sleep=lambda s: None, now=lambda: day(20)), fake


def test_snapshot_history_uses_snapshot_time():
    source, fake = make_source()
    obs = source.snapshot_prices(["https://www.lambda.ai/pricing"], [day(0), day(7), day(14)])
    assert [o.usd_per_gpu_hour for o in obs] == [2.99, 2.99, 2.49]  # the 4090 is dropped
    assert all(o.observed_at == o.available_at for o in obs)
    assert obs[0].observed_at == day(0) - timedelta(hours=6)
    assert obs[0].provider == "lambda" and obs[0].source_kind == "page_snapshot"
    assert fake.calls[0][1]["snapshot_as_of"] == "2025-09-01T00:00:00Z"


def test_discovery_keeps_one_url_per_provider():
    source, _ = make_source()
    assert source.discover_pricing_pages(queries=["h100 pricing"]) == [
        "https://www.lambda.ai/pricing", "https://www.runpod.io/pricing"]


def test_events_are_filtered_to_the_window():
    source, _ = make_source()
    events = source.events("h100 price cut", day(0), day(30))
    assert [e.title for e in events] == ["H100 prices cut"]
    assert events[0].event_id.startswith("N") and "2.49" in events[0].snippet


def test_failed_calls_are_retried_then_skipped():
    class Broken:
        def get_contents(self, urls, **kwargs):
            raise RuntimeError("rate limited")

    source = ExaSource(client=Broken(), retries=2, sleep=lambda s: None)
    assert source.live_prices(["https://x.example.com"]) == []
    assert source.log.calls == 2 and source.log.failures == 2


# ---- the store ----------------------------------------------------------------
def obs(provider: str, price: float, observed: datetime, available: datetime | None = None) -> PriceObservation:
    return PriceObservation(gpu="H100", provider=provider, pricing_type="on_demand", usd_per_gpu_hour=price,
                            observed_at=observed, available_at=available or observed,
                            source_url=f"https://{provider}.example.com", source_kind="page_snapshot")


def test_as_of_hides_the_future():
    store = PriceStore([obs("a", 3.0, day(0)), obs("a", 2.5, day(10)),
                        obs("b", 2.8, day(1), available=day(12))])  # known only on day 12
    view = store.as_of(day(11))
    assert [p.usd_per_gpu_hour for p in view.prices()] == [3.0, 2.5]
    assert max(view.daily_median("H100", "on_demand", carry_days=30)) <= day(11).date()
    with pytest.raises(RuntimeError):
        view.add_price(obs("c", 1.0, day(2)))
    with pytest.raises(ValueError):
        view.as_of(day(20))


def test_dedup_keeps_the_earliest_availability():
    store = PriceStore()
    assert store.add_price(obs("a", 3.0, day(0), available=day(5)))
    assert not store.add_price(obs("a", 3.0, day(0), available=day(2)))
    assert store.prices()[0].available_at == day(2)
    assert len(store) == 1


def test_daily_median_carries_quotes_forward():
    store = PriceStore([obs("a", 3.0, day(0)), obs("b", 2.0, day(0)), obs("c", 2.6, day(1)),
                        obs("a", 2.4, day(3))])
    series = store.daily_median("H100", "on_demand", carry_days=2)
    assert series[day(0).date()] == 2.5            # a, b
    assert series[day(1).date()] == 2.6            # a, b, c
    assert series[day(3).date()] == pytest.approx(2.5)  # a's new 2.4, c's 2.6 (b expired)


def test_save_and_load_round_trip(tmp_path):
    source, _ = make_source()
    store = PriceStore()
    for p in source.snapshot_prices(["https://www.lambda.ai/pricing"], [day(0), day(14)]):
        store.add_price(p)
    for e in source.events("cut", day(0), day(30)):
        store.add_event(e)
    store.save(tmp_path)
    loaded = PriceStore.load(tmp_path)
    assert [p.to_json() for p in loaded.prices()] == [p.to_json() for p in store.prices()]
    assert [e.event_id for e in loaded.events()] == [e.event_id for e in store.events()]
