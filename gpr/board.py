"""The shared board: one per (series, origin), read by every agent.

The board holds the evidence pack (every item has an ID agents cite), the
track record and lessons (filled in by the loop in block 3), the positions
the Forecasters post, and the challenges the Critic and Forecasters post.

Everything on the board is built from ``Pack.history(series_id, origin)`` and
from events published on or before the origin, so nothing on it is dated
after the origin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .pack import HORIZONS, Pack


@dataclass
class Evidence:
    eid: str
    text: str


@dataclass
class Position:
    agent: str
    quantiles: dict[int, tuple[float, float, float]]   # horizon -> (q10, q50, q90)
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""
    problems: list[str] = field(default_factory=list)
    fallback: bool = False                              # True if this is the baseline, not the agent's answer
    raw: str = ""


@dataclass
class Challenge:
    author: str
    target: str
    claim: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class Board:
    series_id: str
    origin: date
    info: dict[str, Any]
    last_value: float
    evidence: list[Evidence]
    baselines: dict[str, dict[int, tuple[float, float, float]]]   # method -> horizon -> quantiles
    events: list[Evidence] = field(default_factory=list)
    track_record: dict[str, str] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)
    market_note: str = ""
    positions: dict[str, Position] = field(default_factory=dict)        # independent round
    challenges: list[Challenge] = field(default_factory=list)
    revised: dict[str, Position] = field(default_factory=dict)          # revision round

    @property
    def evidence_ids(self) -> set[str]:
        return {e.eid for e in self.evidence} | {e.eid for e in self.events}

    def render_evidence(self) -> str:
        lines = [f"Series: {self.series_id}", f"Forecast origin: {self.origin.isoformat()} "
                 "(you know nothing dated after it)", "", "Evidence:"]
        lines += [f"[{e.eid}] {e.text}" for e in self.evidence]
        if self.events:
            lines += ["", "News events published on or before the origin:"]
            lines += [f"[{e.eid}] {e.text}" for e in self.events]
        if self.market_note:
            lines += ["", f"Market Analyst note: {self.market_note}"]
        if self.track_record:
            lines += ["", "Track record (recent pinball loss, lower is better; voting weight):"]
            lines += [f"- {agent}: {text}" for agent, text in sorted(self.track_record.items())]
        if self.lessons:
            lines += ["", "Lessons from past mistakes:"] + [f"- {lesson}" for lesson in self.lessons]
        return "\n".join(lines)

    def render_positions(self, positions: dict[str, Position], exclude: str | None = None) -> str:
        lines = []
        for agent, pos in positions.items():
            if agent == exclude:
                continue
            q = "; ".join(f"h={h}: {fmt(*pos.quantiles[h])}" for h in sorted(pos.quantiles))
            cites = ", ".join(pos.evidence) or "none"
            lines.append(f"- {agent}: {q}. Cites {cites}. Reasoning: {pos.reasoning[:400]}")
        return "\n".join(lines) or "(none)"

    def render_challenges(self, target: str | None = None) -> str:
        rows = [c for c in self.challenges if target is None or c.target == target]
        return "\n".join(f"- from {c.author} to {c.target}: {c.claim} (cites {', '.join(c.evidence) or 'none'})"
                         for c in rows) or "(none)"


def fmt(q10: float, q50: float, q90: float) -> str:
    return f"[{q10:.4g}, {q50:.4g}, {q90:.4g}]"


def _pct(a: float, b: float) -> str:
    return "n/a" if not (a and b) else f"{100 * (b / a - 1):+.1f}%"


def _changes(values: np.ndarray, h: int) -> np.ndarray:
    if len(values) <= h:
        return np.array([])
    a, b = values[:-h], values[h:]
    ok = (a > 0) & (b > 0)
    return np.log(b[ok] / a[ok])


def build_board(pack: Pack, series_id: str, origin: date,
                baselines: dict[str, dict[int, tuple[float, float, float]]],
                events: Iterable[Any] = (), max_events: int = 6, path_days: int = 60) -> Board | None:
    """Assemble the evidence pack for one series at one origin; None if there is no history."""
    hist = pack.history(series_id, origin)
    if hist.empty:
        return None
    values = hist["value"].to_numpy(dtype=float)
    dates = list(hist["date"])
    last, last_date = float(values[-1]), dates[-1]
    info = pack.info(series_id)

    ev: list[Evidence] = []

    def add(text: str) -> None:
        ev.append(Evidence(f"E{len(ev) + 1}", text))

    contract = info.get("benchmark_contract") or ""
    add(f"Instrument: {info.get('instrument_class')}; GPU {info.get('gpu_model') or info.get('instance_type')}; "
        f"tier {info.get('pricing_tier') or 'spot'}; source {info.get('source_id')}. Contract: {contract[:300]}")

    changed = np.flatnonzero(np.diff(values) != 0)
    since = (last_date - dates[changed[-1] + 1]).days if len(changed) else (last_date - dates[0]).days
    add(f"Last value {last:.4g} USD per GPU-hour on {last_date.isoformat()} "
        f"({(origin - last_date).days} days before the origin); unchanged for {since} days; "
        f"{len(hist)} days of history from {dates[0].isoformat()}.")

    recent = hist[hist["date"] > origin - pd.Timedelta(days=path_days)]
    step = max(1, len(recent) // 15)
    path = ", ".join(f"{d.isoformat()[5:]}: {v:.4g}" for d, v in zip(recent["date"][::step], recent["value"][::step]))
    add(f"Recent path (every {step} days, MM-DD: value): {path}")

    def back(days: int) -> float | None:
        older = hist[hist["date"] <= origin - pd.Timedelta(days=days)]
        return float(older["value"].iloc[-1]) if len(older) else None

    add("Change to the last value: " + "; ".join(
        f"over {d} days {_pct(back(d), last)}" for d in (7, 30, 90)))

    distinct = len(np.unique(values[-90:]))
    moves = len(changed[changed >= len(values) - 91]) if len(changed) else 0
    add(f"In the last 90 days: {distinct} distinct values, {moves} days with a change.")

    parts = []
    for h in HORIZONS:
        ch = _changes(values, h)
        if len(ch) >= 20:
            p10, p50, p90 = np.percentile(ch, [10, 50, 90])
            parts.append(f"h={h}: p10 {100 * (math.exp(p10) - 1):+.1f}%, p50 {100 * (math.exp(p50) - 1):+.1f}%, "
                         f"p90 {100 * (math.exp(p90) - 1):+.1f}%, share unchanged {np.mean(ch == 0):.0%}")
    if parts:
        add("Historical h-day changes: " + "; ".join(parts))

    if "n_providers" in hist and hist["n_providers"].notna().any():
        add(f"Providers in the median, last 7 days: {hist['n_providers'].tail(7).median():.0f}")

    for method, by_h in baselines.items():
        add(f"Baseline {method} (q10, q50, q90): " + "; ".join(f"h={h}: {fmt(*by_h[h])}" for h in sorted(by_h)))

    cutoff = datetime.combine(origin, time(23, 59, 59), tzinfo=timezone.utc)
    news = [e for e in events if e.published_at <= cutoff]
    news = sorted(news, key=lambda e: e.published_at, reverse=True)[:max_events]
    event_items = [Evidence(e.event_id, f"{e.published_at.date().isoformat()} {e.title}: {e.snippet[:240]}")
                   for e in news]

    return Board(series_id=series_id, origin=origin, info=info, last_value=last, evidence=ev,
                 baselines=baselines, events=event_items)


def baseline_table(df: pd.DataFrame) -> dict[tuple[str, date], dict[str, dict[int, tuple[float, float, float]]]]:
    """Index starter-kit baseline rows as (series, origin) -> method -> horizon -> quantiles."""
    out: dict = {}
    for row in df.itertuples(index=False):
        origin = date.fromisoformat(str(row.origin_date))
        out.setdefault((row.series_id, origin), {}).setdefault(row.method, {})[int(row.horizon_days)] = (
            float(row.q10), float(row.q50), float(row.q90))
    return out


def board_to_json(board: Board) -> dict[str, Any]:
    def pos(p: Position) -> dict[str, Any]:
        return {"agent": p.agent, "quantiles": {str(h): list(q) for h, q in p.quantiles.items()},
                "evidence": p.evidence, "reasoning": p.reasoning, "problems": p.problems, "fallback": p.fallback,
                "raw": p.raw[:4000]}

    return {
        "series_id": board.series_id, "origin": board.origin.isoformat(),
        "evidence": [e.__dict__ for e in board.evidence], "events": [e.__dict__ for e in board.events],
        "market_note": board.market_note,
        "positions": {k: pos(v) for k, v in board.positions.items()},
        "challenges": [c.__dict__ for c in board.challenges],
        "revised": {k: pos(v) for k, v in board.revised.items()},
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)
