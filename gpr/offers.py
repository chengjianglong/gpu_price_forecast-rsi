"""The offers behind each posted rental series.

Every posted rental series in the pack is the daily median of one collector's list of offers for
one GPU and pricing tier. ``OfferTable`` holds that list through time: one row per day, one column
per offer (``provider|sku|region|s<snapshot>|d<duplicate>``). The median of a day's row
reproduces the official value (``verify``).

The engine and the board read offers through ``Offers.table(series_id, origin)``, which returns
rows dated on or before the origin only. Ported from the sister project ``gpu-price``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .pack import DATA_CUTOFF, PACK_DIR, Pack

KEY = ["instrument_class", "gpu_model", "pricing_tier"]


@dataclass
class OfferTable:
    series_id: str
    instrument_class: str
    days: np.ndarray       # int64 day ordinals, one per row of the official series
    value: np.ndarray      # the official daily value (the pack's median), USD per GPU-hour
    prices: np.ndarray     # float64 [n_days, n_slots]; NaN where the offer was not listed
    slots: list[str]

    @property
    def last_day(self) -> date:
        return date.fromordinal(int(self.days[-1]))

    def cut(self, origin: date) -> "OfferTable | None":
        """A copy holding nothing dated after ``origin``; None if nothing is left."""
        k = int(np.searchsorted(self.days, origin.toordinal(), side="right"))
        if k == 0:
            return None
        prices = self.prices[:k]
        keep = ~np.all(np.isnan(prices), axis=0)
        return OfferTable(self.series_id, self.instrument_class, self.days[:k].copy(), self.value[:k].copy(),
                          prices[:, keep].copy(), [s for s, kept in zip(self.slots, keep) if kept])


def row_median(prices: np.ndarray) -> np.ndarray:
    """Median over each row's listed offers; NaN for a row with none (without numpy's warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(prices, axis=1)


def offer_name(slot: str) -> str:
    """The slot without its snapshot and duplicate numbers: ``provider|sku|region``."""
    return slot.rsplit("|", 2)[0].rstrip("|")


def build_table(series_id: str, instrument_class: str, series: pd.DataFrame, rows: pd.DataFrame) -> OfferTable:
    """One table from the official series (``date``, ``value``) and its offer rows (``day``
    ordinal, ``snapshot``, ``provider``, ``sku``, ``region``, ``price``)."""
    days = np.array([d.toordinal() for d in series["date"]], dtype=np.int64)
    value = series["value"].to_numpy(dtype=float, copy=True)
    rows = rows.copy()
    rows["snap"] = rows.groupby("day")["snapshot"].rank(method="dense").astype(int)
    rows = rows.sort_values(["day", "snap", "provider", "sku", "region", "price"], kind="stable")
    rows["dup"] = rows.groupby(["day", "snap", "provider", "sku", "region"]).cumcount()
    rows["slot"] = (rows["provider"] + "|" + rows["sku"] + "|" + rows["region"] + "|s" + rows["snap"].astype(str)
                    + "|d" + rows["dup"].astype(str))
    wide = rows.pivot(index="day", columns="slot", values="price").reindex(days)
    return OfferTable(series_id, instrument_class, days, value, wide.to_numpy(dtype=float, copy=True), list(wide.columns))


class Offers:
    """Offer tables for the pack's posted rental series, read once and sliced by origin."""

    def __init__(self, pack: Pack, pack_dir: Path = PACK_DIR, cutoff: date = DATA_CUTOFF):
        self.pack = pack
        self.pack_dir = Path(pack_dir)
        self.cutoff = cutoff
        self._rows: dict[str, dict[tuple, pd.DataFrame]] = {}      # source -> series key -> offer rows
        self._tables: dict[str, OfferTable | None] = {}

    def _key(self, series_id: str) -> tuple[str, tuple] | None:
        cat = self.pack.catalog
        if series_id not in cat.index:
            return None
        row = cat.loc[series_id]
        if str(row.get("table", "daily_statistics")) != "daily_statistics":
            return None      # AWS spot and other series have no offer list in rental_observations
        return str(row["source_id"]), tuple(str(row[k]) for k in KEY)

    def _source_rows(self, source: str) -> dict[tuple, pd.DataFrame]:
        if source not in self._rows:
            import duckdb

            folder = self.pack_dir / "rental_observations" / f"source_id={source}"
            if not folder.is_dir():
                self._rows[source] = {}
                return self._rows[source]
            df = duckdb.connect().sql(f"""
                select dt, snapshot_ts_utc as snapshot, instrument_class, gpu_model, pricing_tier,
                       coalesce(provider, '') as provider, coalesce(sku_raw, '') as sku,
                       coalesce(region_raw, '') as region, usd_per_gpu_hour as price
                from read_parquet('{folder}/*/*.parquet', hive_partitioning=true)
                where dt <= DATE '{self.cutoff.isoformat()}' and usd_per_gpu_hour is not null
            """).df()
            df["day"] = pd.to_datetime(df["dt"]).map(pd.Timestamp.toordinal).astype(np.int64)
            self._rows[source] = {tuple(map(str, key)): g.drop(columns=KEY + ["dt"])
                                  for key, g in df.groupby(KEY, sort=False)}
        return self._rows[source]

    def full_table(self, series_id: str) -> OfferTable | None:
        """The whole table up to the pack's cutoff. Forecast code uses ``table``, which cuts it."""
        if series_id not in self._tables:
            key = self._key(series_id)
            rows = self._source_rows(key[0]).get(key[1]) if key else None
            series = self.pack.history(series_id, self.cutoff) if rows is not None else None
            self._tables[series_id] = (None if series is None or series.empty
                                       else build_table(series_id, key[1][0], series, rows))
        return self._tables[series_id]

    def table(self, series_id: str, origin: date) -> OfferTable | None:
        """Offers dated on or before ``origin`` only; None if the series has no offer list."""
        full = self.full_table(series_id)
        return None if full is None else full.cut(origin)


def shared(pack: Pack) -> Offers:
    """One ``Offers`` per pack, so the offer rows are read once."""
    if getattr(pack, "_offers", None) is None:
        pack._offers = Offers(pack)
    return pack._offers


def verify(tables: Sequence[OfferTable]) -> pd.DataFrame:
    """How well the median over each day's offers reproduces the official daily value."""
    out = []
    for t in tables:
        rebuilt = np.round(row_median(t.prices), 4)
        listed = ~np.isnan(rebuilt)
        ok = np.isclose(rebuilt, t.value, atol=1.01e-4)
        out.append({"series_id": t.series_id, "n_days": len(t.days), "n_slots": len(t.slots),
                    "days_with_offers": float(listed.mean()) if len(listed) else 0.0,
                    "match_share": float(ok[listed].mean()) if listed.any() else float("nan")})
    return pd.DataFrame(out)


def make_placebo(table: OfferTable, seed: int) -> OfferTable:
    """A shuffled copy: the same offers and listing pattern, with the real day-to-day price
    changes applied in a shuffled order. One shuffle is shared by all offers of the series, so
    co-movement survives. A change that lands on a day its offer is not listed is dropped, so
    the copy holds fewer changes than the real data (about four in five on the validation scope).
    The official value becomes the median of the shuffled offers.
    """
    with np.errstate(all="ignore"):
        log_p = np.log(table.prices)
    n = len(log_p)
    perm = np.arange(1, n)                      # step i is the move from day i-1 to day i
    np.random.default_rng(seed).shuffle(perm)
    fake = np.full_like(log_p, np.nan)
    fake[0] = last = log_p[0].copy()
    for i, j in zip(range(1, n), perm.tolist()):
        change = log_p[j] - log_p[j - 1]                      # the change that really happened at step j
        change = np.where(np.isnan(change), 0.0, change)      # not listed on both days there: no change
        present = ~np.isnan(log_p[i])                         # the listing pattern stays as it really was
        fake[i] = np.where(present, np.where(np.isnan(last), log_p[i], last + change), np.nan)
        last = np.where(present, fake[i], last)
    prices = np.exp(fake)
    return OfferTable(table.series_id, table.instrument_class, table.days.copy(), np.round(row_median(prices), 4),
                      prices, list(table.slots))


def list_evidence(table: OfferTable, around: int = 5, full_up_to: int = 14, window_days: int = 30,
                  changes_days: int = 14) -> list[str]:
    """The list behind the median on the table's last day, as board evidence (``table`` already
    cut at the origin). Empty when no offers are listed that day. A collector's same-day
    snapshots are merged, so each offer appears once."""
    if np.isnan(table.prices[-1]).all():
        return []
    days = table.days
    name = [offer_name(s) for s in table.slots]
    dup = [s.rsplit("|", 1)[1] for s in table.slots]
    merged = pd.DataFrame(table.prices.T).groupby([name, dup], sort=False).first()   # a day's first snapshot
    names, prices = merged.index.get_level_values(0).to_numpy(), merged.to_numpy().T
    today = prices[-1]
    here = np.flatnonzero(~np.isnan(today))
    order = here[np.argsort(today[here], kind="stable")]
    n = len(order)
    middle = {(n - 1) // 2, n // 2}
    lo = int(np.searchsorted(days, days[-1] - window_days))
    show = range(n) if n <= full_up_to else range(max(0, (n - 1) // 2 - around), min(n, n // 2 + around + 1))
    head = (f"The list behind this median on {table.last_day.isoformat()}: {n} offers, sorted by price; "
            "'<<' marks the middle.")
    if n > full_up_to:
        head += (f" Showing the offers around the middle; {min(show)} cheaper ones from {today[order[0]]:.4g} and "
                 f"{n - 1 - max(show)} dearer ones up to {today[order[-1]]:.4g} are not shown.")
    lines = [head]
    for pos in show:
        j = order[pos]
        col = prices[lo:, j]
        listed = ~np.isnan(col)
        steps = np.diff(col[listed])
        moved = np.flatnonzero(np.abs(steps) > 1e-9)
        last = ""
        if len(moved):
            idx = np.flatnonzero(listed)
            k = moved[-1]
            when = date.fromordinal(int(days[lo + idx[k + 1]])).isoformat()[5:]
            last = f"; last change {when}: {col[idx[k]]:.4g} -> {col[idx[k + 1]]:.4g}"
        lines.append(f"  {names[j][:46]} {today[j]:.4g}{' <<' if pos in middle else ''} "
                     f"(listed {int(listed.sum())} of the last {len(col)} days; {len(moved)} price changes{last})")
    lo = int(np.searchsorted(days, days[-1] - changes_days))
    now, then = set(names[here]), set(names[~np.isnan(prices[lo])])

    def some(items: set) -> str:
        items = sorted(items)
        return (", ".join(items[:6]) + (f" and {len(items) - 6} more" if len(items) > 6 else "")) or "none"

    return ["\n".join(lines), f"List changes since {date.fromordinal(int(days[lo])).isoformat()[5:]}: "
                              f"joined: {some(now - then)}; left: {some(then - now)}."]


FREQUENT_SHARE = 0.30   # a frequent repricer repriced on at least this share of the days it was listed


def median_moves(table: OfferTable) -> dict:
    """What moved this median day to day, split in order into ``list`` (offers joining or
    leaving), ``frequent`` (repricing by frequent repricers) and ``rare`` (all other repricing).
    Each is the summed absolute log move of the median it explains."""
    prices = table.prices
    n, m = prices.shape
    listed = ~np.isnan(prices)
    both = listed[1:] & listed[:-1]
    changed = both & (np.abs(prices[1:] - prices[:-1]) > 1e-9)
    n_both = both.sum(axis=0)
    share = np.divide(changed.sum(axis=0), n_both, out=np.zeros(m), where=n_both > 0)
    frequent = (share >= FREQUENT_SHARE) & (n_both >= 5)

    def med(v: np.ndarray) -> float:
        v = v[~np.isnan(v)]
        return float(np.median(v)) if len(v) else float("nan")

    total = {"list": 0.0, "frequent": 0.0, "rare": 0.0}
    moves = 0
    for i in range(1, n):
        a, b = prices[i - 1], prices[i]
        common = listed[i - 1] & listed[i]
        if not common.any():
            continue
        before, after = med(a), med(b)
        if not (before > 0 and after > 0) or abs(np.log(after / before)) < 1e-12:
            continue
        moves += 1
        stayed = med(np.where(common, a, np.nan))                                   # exits removed
        some = med(np.where(common & frequent, b, np.where(common, a, np.nan)))     # frequent repricers moved
        everyone = med(np.where(common, b, np.nan))                                 # everyone repriced
        total["list"] += abs(np.log(stayed / before)) + abs(np.log(after / everyone))
        total["frequent"] += abs(np.log(some / stayed))
        total["rare"] += abs(np.log(everyone / some))
    ever = listed.any(axis=0)
    return {"series_id": table.series_id, "instrument_class": table.instrument_class, "days": n,
            "offers": int(ever.sum()), "frequent_offers": int((frequent & ever).sum()),
            "share_of_days_moved": moves / max(n - 1, 1), **total}
