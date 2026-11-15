"""The list-replay engine: replay each offer's own past, then take the median again.

Pacific's ``drift`` replays how the median moved. For a horizon of h days the engine takes every
past pair of days (d, d + h) in the lookback window, applies to today's list what each offer did
between them (repriced, stayed, left, joined) and takes the median again. The spread of those
replayed medians is the range. Nothing is trained and no LLM is called.

Ported from the sister project ``gpu-price`` (``gpuval/engine.py``) with its settings; tuning
them there did not hold up. ``forecast`` takes a table already cut at the origin.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .offers import OfferTable, row_median
from .pack import QUANTILES, SCORED_HORIZONS

METHOD = "engine"
LOOKBACK_DAYS = 60    # replayed pairs (d, d + h) start within this many days of the last day
MIN_SCENARIOS = 10    # with fewer usable pairs there is no forecast


def replay(table: OfferTable, h: int) -> np.ndarray:
    """Replayed h-day log changes of the median, one per usable past pair of days."""
    days = table.days
    i0 = len(days) - 1
    with np.errstate(all="ignore"):
        log_p = np.log(table.prices)
    x = log_p[i0]
    today = ~np.isnan(x)
    a = np.arange(int(np.searchsorted(days, days[i0] - LOOKBACK_DAYS)), i0 + 1)
    b = np.minimum(np.searchsorted(days, days[a] + h), i0)
    ok = days[b] == days[a] + h
    a, b = a[ok], b[ok]
    if len(a) == 0 or not today.any():
        return np.empty(0)
    was, still = ~np.isnan(log_p[a]), ~np.isnan(log_p[b])
    new = x + np.where(was & still, log_p[b] - log_p[a], 0.0)     # listed on both days: its repricing
    new[was & ~still & today] = np.nan                            # left the list between d and d + h
    valid = ~np.isnan(log_p)
    last_known = log_p[i0 - np.argmax(valid[::-1], axis=0), np.arange(log_p.shape[1])]
    new = np.where(~was & still & ~today, last_known, new)        # joined, absent today: last known price
    medians = row_median(np.exp(new))
    good = np.isfinite(medians) & (medians > 0)
    return np.log(medians[good] / np.nanmedian(np.exp(x)))


def forecast(table: OfferTable) -> dict[int, tuple[float, float, float]]:
    """(q10, q50, q90) per horizon from the table's last day; horizons it cannot replay are left out."""
    last = float(table.value[-1])
    out = {}
    for h in SCORED_HORIZONS if last > 0 else ():
        changes = replay(table, h)
        if len(changes) >= MIN_SCENARIOS:
            out[h] = tuple((last * np.exp(np.quantile(changes, QUANTILES))).tolist())
    return out


def forecast_rows(table_for: Callable[[str, date], OfferTable | None], origin: date, series_ids: Sequence[str],
                  fallback: pd.DataFrame) -> pd.DataFrame:
    """Engine forecasts in the starter kit's forecast-file format.

    ``table_for`` is ``Offers.table``. ``fallback`` holds Pacific's ``drift`` rows for the origin.
    They are used, with ``method_note`` ``drift_fallback``, where the engine has no forecast: no
    list on the origin day, or too few pairs to replay.
    """
    by_cell = {(r.series_id, int(r.horizon_days)): (float(r.q10), float(r.q50), float(r.q90))
               for r in fallback[fallback["origin_date"].astype(str) == origin.isoformat()].itertuples(index=False)}
    rows = []
    for sid in series_ids:
        table = table_for(sid, origin)
        ours = forecast(table) if table is not None and table.last_day == origin else {}
        for h in SCORED_HORIZONS:
            q, note = (ours[h], METHOD) if h in ours else (by_cell.get((sid, h)), "drift_fallback")
            if q is not None:
                rows.append({"series_id": sid, "origin_date": origin.isoformat(), "horizon_days": h,
                             "q10": round(q[0], 6), "q50": round(q[1], 6), "q90": round(q[2], 6),
                             "target_date": (origin + timedelta(days=h)).isoformat(), "method": METHOD,
                             "method_note": note})
    return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90", "target_date",
                                       "method", "method_note"])

