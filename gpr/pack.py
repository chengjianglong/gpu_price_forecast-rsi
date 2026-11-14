"""Pacific's data pack: series files, the train/validation/test/submission
splits, and thin wrappers around the starter kit's scripts.

The starter kit's ``build_series.py``, ``baselines.py`` and ``scorer.py`` are
run as subprocesses (``python -I -B``, as the kit's own ``run_demo.sh`` does)
rather than imported: they are the official definitions of the series, the
baselines and the score, and running them unchanged keeps our numbers equal to
Pacific's.

``Pack.history(series_id, origin)`` is the only way agents see prices, and it
returns rows dated on or before the origin only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "data" / "raw" / "gpu-price-pack-v0.1.1-handover"
DERIVED = ROOT / "data" / "derived"
DATA_CUTOFF = date(2026, 11, 6)        # the pack's last day; also the submission origin
HORIZONS = (1, 3, 7, 30)
QUANTILES = (0.1, 0.5, 0.9)
BASELINE_METHODS = ("last_value", "last_value_empirical", "drift", "seasonal_naive")


def _d(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def weekly(start: str | date, end: str | date, step_days: int = 7) -> list[date]:
    out, day = [], _d(start)
    while day <= _d(end):
        out.append(day)
        day += timedelta(days=step_days)
    return out


@dataclass(frozen=True)
class Splits:
    """Forecast origins per split (docs/PLAN.md section 2.6)."""

    train: tuple[date, ...] = tuple(weekly("2026-02-04", "2026-09-30"))
    validation: tuple[date, ...] = tuple(weekly("2026-10-01", "2026-10-15"))
    test: date = date(2026, 10, 24)
    submission: date = DATA_CUTOFF

    def check(self) -> None:
        """Origins must be ordered train < validation < test < submission with no overlap."""
        assert max(self.train) < min(self.validation), "train overlaps validation"
        assert max(self.validation) < self.test < self.submission, "splits out of order"


class StarterError(RuntimeError):
    pass


def run_starter(script: str, *args: str | Path, pack_dir: Path = PACK_DIR, check: bool = True,
                python: str = sys.executable) -> subprocess.CompletedProcess:
    """Run one starter-kit script in isolated mode and return the finished process."""
    path = pack_dir / "starter" / script
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: unpack Pacific's pack into {pack_dir.parent}")
    proc = subprocess.run([python, "-I", "-B", str(path), *map(str, args)], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise StarterError(f"{script} exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


@dataclass
class PackFiles:
    """Series files built by the starter kit for one catalog date."""

    as_of: date
    directory: Path
    series: Path = field(init=False)
    catalog: Path = field(init=False)

    def __post_init__(self) -> None:
        self.series = self.directory / "series_daily.csv"
        self.catalog = self.directory / "series_catalog.csv"

    def history_file(self, cutoff: date) -> Path:
        return self.directory / f"history_{cutoff.isoformat()}.csv"

    def history_catalog(self, cutoff: date) -> Path:
        return self.directory / f"catalog_{cutoff.isoformat()}.csv"


def build(as_of: str | date = DATA_CUTOFF, out_root: Path = DERIVED, pack_dir: Path = PACK_DIR,
          force: bool = False) -> PackFiles:
    """Build ``series_daily.csv`` and ``series_catalog.csv`` with the catalog dated ``as_of``."""
    as_of = _d(as_of)
    files = PackFiles(as_of, out_root / f"pack_{as_of.isoformat()}")
    files.directory.mkdir(parents=True, exist_ok=True)
    if force or not (files.series.is_file() and files.catalog.is_file()):
        run_starter("build_series.py", "--pack", pack_dir, "--out", files.series, "--catalog", files.catalog,
                    "--as-of", as_of.isoformat(), "--report", files.directory / "build_series_report.json",
                    pack_dir=pack_dir)
    return files


def build_history(files: PackFiles, cutoff: str | date, pack_dir: Path = PACK_DIR, force: bool = False) -> Path:
    """The snapshot truncated at ``cutoff``, as frozen-cutoff scoring requires."""
    cutoff = _d(cutoff)
    out = files.history_file(cutoff)
    if force or not out.is_file():
        run_starter("build_series.py", "--pack", pack_dir, "--cutoff", cutoff.isoformat(),
                    "--members-from", files.catalog, "--out", out,
                    "--catalog", files.history_catalog(cutoff),
                    "--report", files.directory / f"build_history_{cutoff.isoformat()}.json", pack_dir=pack_dir)
    return out


def baselines(history: Path, catalog: Path, method: str, out: Path, origins: Sequence[date] | None = None,
              origin: date | None = None, tier: str = "headline", frozen: bool = False,
              pack_dir: Path = PACK_DIR) -> pd.DataFrame:
    """Pacific's baseline forecasts for one origin or a weekly range of origins."""
    args: list[str | Path] = ["--history", history, "--method", method, "--catalog", catalog, "--tier", tier,
                              "--out", out]
    if origin is not None:
        args += ["--origin", origin.isoformat()]
    elif origins:
        ordered = sorted(origins)
        steps = {(b - a).days for a, b in zip(ordered, ordered[1:])} or {7}
        if len(steps) != 1:
            raise ValueError("origins must be evenly spaced for the starter's --origin-step")
        args += ["--origin-start", ordered[0].isoformat(), "--origin-end", ordered[-1].isoformat(),
                 "--origin-step", str(steps.pop())]
    else:
        raise ValueError("give origin or origins")
    if frozen:
        args.append("--frozen")
    run_starter("baselines.py", *args, pack_dir=pack_dir)
    return pd.read_csv(out)


def score(forecast: Path, truth: Path, history: Path, catalog: Path, out: Path, per_cell: Path | None = None,
          frozen_cutoff: date | None = None, tier: str = "headline", extra: Sequence[str] = (),
          pack_dir: Path = PACK_DIR) -> dict:
    """Run Pacific's scorer and return its JSON report (status ``refused`` on a refusal)."""
    args: list[str | Path] = ["--forecast", forecast, "--truth", truth, "--history", history,
                              "--catalog", catalog, "--tier", tier, "--out", out]
    if per_cell is not None:
        args += ["--per-cell", per_cell]
    if frozen_cutoff is not None:
        args += ["--frozen-cutoff", frozen_cutoff.isoformat()]
    proc = run_starter("scorer.py", *args, *extra, pack_dir=pack_dir, check=False)
    if proc.returncode not in (0, 2):
        raise StarterError(f"scorer.py exited {proc.returncode}: {proc.stderr.strip()}")
    report = json.loads(out.read_text()) if out.is_file() else {"status": "error", "stderr": proc.stderr}
    report["_exit_code"] = proc.returncode
    return report


def series_at(files: PackFiles, origin: str | date, tier: str | None = "headline",
              daily_truth_only: bool = False, pack_dir: Path = PACK_DIR) -> list[str]:
    """Series in a tier according to the catalog dated ``origin``.

    The tier gates (recent value, enough history, enough movement) depend on the
    date, so the headline set differs by origin: 97 series at 24 Oct (80 AWS
    spot), 37 at 6 Nov (the AWS archive ends 31 Oct). Scoring at an origin uses
    that origin's catalog, so series must be chosen from it too.
    """
    origin = _d(origin)
    build_history(files, origin, pack_dir=pack_dir)
    cat = pd.read_csv(files.history_catalog(origin))
    if tier:
        cat = cat[cat["tier"] == tier]
    if daily_truth_only:
        cat = cat[cat["truth_availability"].astype(str).str.startswith("daily")]
    return sorted(cat["series_id"])


class Pack:
    """Read access to the series and catalog, sliced by origin."""

    def __init__(self, files: PackFiles):
        self.files = files

    @cached_property
    def series(self) -> pd.DataFrame:
        df = pd.read_csv(self.files.series, usecols=["series_id", "date", "value", "n_providers", "is_filled",
                                                     "instrument_class"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values(["series_id", "date"]).reset_index(drop=True)

    @cached_property
    def catalog(self) -> pd.DataFrame:
        return pd.read_csv(self.files.catalog).set_index("series_id")

    @cached_property
    def _by_series(self) -> dict[str, pd.DataFrame]:
        return {sid: g.reset_index(drop=True) for sid, g in self.series.groupby("series_id", sort=False)}

    def series_ids(self, tier: str | None = "headline", daily_truth_only: bool = False) -> list[str]:
        """Series in a tier. ``daily_truth_only`` keeps the ones with daily truth (scoreable on 7–13 Nov)."""
        cat = self.catalog
        if tier:
            cat = cat[cat["tier"] == tier]
        if daily_truth_only:
            cat = cat[cat["truth_availability"].astype(str).str.startswith("daily")]
        return sorted(cat.index)

    def history(self, series_id: str, origin: date, days: int | None = None) -> pd.DataFrame:
        """Rows dated on or before ``origin`` only; optionally the last ``days`` calendar days."""
        rows = self._by_series.get(series_id)
        if rows is None:
            return self.series.iloc[0:0]
        rows = rows[rows["date"] <= origin]
        if days is not None:
            rows = rows[rows["date"] > origin - timedelta(days=days)]
        return rows

    def truth(self, series_id: str, target: date) -> float | None:
        rows = self._by_series.get(series_id)
        if rows is None:
            return None
        hit = rows[rows["date"] == target]
        return float(hit["value"].iloc[0]) if len(hit) else None

    def info(self, series_id: str) -> dict:
        row = self.catalog.loc[series_id]
        keys = ["instrument_class", "gpu_model", "pricing_tier", "source_id", "instance_type", "benchmark_contract"]
        return {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in keys}
