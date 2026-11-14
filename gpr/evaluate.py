"""Evaluate a team on a list of origins with Pacific's scorer.

Origins run in date order. After each origin, the voting weights get the
Hedge update from each Forecaster's own losses on cells whose target is
known before the next origin (and on or before the truth cap). So the weights
used at an origin only ever learned from outcomes dated before it.

All origins' forecasts go into one file, scored once in backtest mode against
a truth file truncated at the cap, so validation never sees the test window.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from . import pack as packlib
from .rewards import FLOOR_SHARE, VALIDATION_TRUTH_CAP, pinball
from .team import OriginResult, Team


@dataclass
class EvalResult:
    report: dict
    cells: pd.DataFrame          # Pacific's per-cell output, scored cells only
    results: list[OriginResult]
    directory: Path

    @property
    def pooled(self) -> float:
        value = (self.report.get("overall") or {}).get("skill_pooled")
        return float("nan") if value is None else float(value)

    def summary(self) -> dict:
        overall = self.report.get("overall") or {}
        keys = ("skill_pooled", "skill_median_of_series", "skill_mean_of_series", "skill_geomean", "coverage")
        return {k: overall.get(k) for k in keys} | {"cells": (self.report.get("cells") or {}).get("scored")}


def capped_truth(files: packlib.PackFiles, cap: date) -> Path:
    out = files.directory / f"truth_until_{cap.isoformat()}.csv"
    if not out.is_file():
        df = pd.read_csv(files.series)
        df[df["date"] <= cap.isoformat()].to_csv(out, index=False)
    return out


def update_weights(team: Team, result: OriginResult, known_before: date, cap: date) -> None:
    """Hedge update from each Forecaster's own (revised) forecasts on cells already known."""
    for board in result.boards:
        cls = str(board.info.get("instrument_class"))
        final = board.revised or board.positions
        for h in packlib.SCORED_HORIZONS:
            target = board.origin + timedelta(days=h)
            if target >= known_before or target > cap:
                continue
            y = team.pack.truth(board.series_id, target)
            if y is None:
                continue
            last = board.last_value
            base = max(pinball(y, (last, last, last)), FLOOR_SHARE * abs(last), 1e-9)
            losses = {a: min(3.0, pinball(y, p.quantiles[h]) / base) for a, p in final.items()}
            team.weights.update((cls, h), losses)


def evaluate(team: Team, origins: Sequence[date], out_dir: Path,
             series_for: Callable[[date], Sequence[str]] | None = None,
             cap: date = VALIDATION_TRUTH_CAP) -> EvalResult:
    """Run and score ``origins`` in date order; default series are the rental scope."""
    files = team.pack.files
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(origins)
    results, frames = [], []
    for i, origin in enumerate(ordered):
        series = (series_for or (lambda o: packlib.rental_scope(files, o)))(origin)
        result = team.run(origin, series)
        results.append(result)
        frames.append(result.rows)
        nxt = ordered[i + 1] if i + 1 < len(ordered) else cap + timedelta(days=1)
        update_weights(team, result, nxt, cap)
    forecast = out_dir / "forecast.csv"
    pd.concat(frames, ignore_index=True).to_csv(forecast, index=False)
    with (out_dir / "transcripts.jsonl").open("w") as handle:
        from .board import board_to_json

        for result in results:
            for b in result.boards:
                handle.write(json.dumps(board_to_json(b), default=str) + "\n")
    (out_dir / "weights.json").write_text(json.dumps(team.weights.to_json(), indent=1))
    report = packlib.score(forecast, capped_truth(files, cap), files.series, files.catalog,
                           out_dir / "report.json", out_dir / "cells.csv", tier="all")
    cells = pd.read_csv(out_dir / "cells.csv") if (out_dir / "cells.csv").is_file() else pd.DataFrame()
    if not cells.empty:
        cells = cells[cells["status"] == "scored"].reset_index(drop=True)
    return EvalResult(report, cells, results, out_dir)


def paired_bootstrap(new: pd.DataFrame, old: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """P(new pooled skill > old) by resampling (series, origin) blocks of matched cells."""
    key = ["series_id", "origin_date", "horizon_days"]
    m = new[key + ["loss", "baseline_loss"]].merge(old[key + ["loss"]], on=key, suffixes=("_new", "_old"))
    if m.empty:
        return {"p_better": 0.0, "delta": 0.0, "cells": 0}
    blocks = [g for _, g in m.groupby(["series_id", "origin_date"])]
    sums = [(g["loss_new"].sum(), g["loss_old"].sum(), g["baseline_loss"].sum()) for g in blocks]

    def delta(sample) -> float:
        ln = sum(s[0] for s in sample)
        lo = sum(s[1] for s in sample)
        lb = sum(s[2] for s in sample) or 1e-12
        return (lo - ln) / lb   # = skill_new - skill_old

    rng = random.Random(seed)
    observed = delta(sums)
    wins = sum(delta(rng.choices(sums, k=len(sums))) > 0 for _ in range(n))
    return {"p_better": wins / n, "delta": observed, "cells": len(m)}
