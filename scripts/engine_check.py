"""Check the list-replay engine against Pacific's baselines. No LLM, and no locked test.

    python scripts/engine_check.py                 # a few minutes
    python scripts/engine_check.py --placebos 0    # skip the shuffled-data control

1. Offer lists: the median over each day's offers must reproduce the pack's official value.
2. Validation and training windows, rental scope, h = 1, 3, 7: the engine, ``drift`` and
   ``last_value_empirical`` on the same cells, scored by Pacific's scorer.
3. Shuffled-data control (validation origins): the order of every list's day-to-day changes is
   scrambled. If the engine beats drift there as on the real data, its edge is not timing. Rough:
   a copy keeps about four in five of the real changes.
4. What moves a median (data to 30 Sep): offers joining or leaving, frequent repricers, the rest.

Writes ``results/engine.md``.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr import engine  # noqa: E402
from gpr import pack as packlib  # noqa: E402
from gpr.evaluate import capped_truth  # noqa: E402
from gpr.offers import make_placebo, median_moves, shared, verify  # noqa: E402
from gpr.rewards import TRAIN_TRUTH_CAP, VALIDATION_TRUTH_CAP  # noqa: E402

METHODS = ("drift", "last_value_empirical")
KEY = ["series_id", "origin_date", "horizon_days"]


def in_scope(df: pd.DataFrame, scope: dict[date, list[str]]) -> pd.DataFrame:
    keep = {(s, o.isoformat()) for o, ids in scope.items() for s in ids}
    return df[[(s, o) in keep for s, o in zip(df["series_id"], df["origin_date"].astype(str))]]


def score(forecast: pd.DataFrame, name: str, truth: Path, history: Path, catalog: Path, out: Path) -> pd.DataFrame:
    """Pacific's scorer in backtest mode; returns the scored cells."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    forecast.to_csv(path, index=False)
    report = packlib.score(path, truth, history, catalog, out / f"{name}_report.json", out / f"{name}_cells.csv",
                           tier="all")
    if report.get("status") != "scored":
        raise SystemExit(f"scorer did not score {name}: {report.get('status')} {report.get('refusals')}")
    cells = pd.read_csv(out / f"{name}_cells.csv")
    return cells[cells["status"] == "scored"].reset_index(drop=True)


def run_world(label: str, origins: list[date], scope: dict[date, list[str]], table_for, history: Path, truth: Path,
              catalog: Path, out: Path) -> dict[str, pd.DataFrame]:
    """Forecast and score one world: Pacific's baselines from ``history``, the engine from ``table_for``."""
    cells, drift = {}, None
    for method in METHODS:
        rows = in_scope(packlib.baselines(history, catalog, method, out / f"{method}_all.csv", origins=origins,
                                          tier="all"), scope)
        if method == "drift":
            drift = rows
        cells[method] = score(rows, method, truth, history, catalog, out)
    rows = pd.concat([engine.forecast_rows(table_for, o, scope[o], drift) for o in origins], ignore_index=True)
    cells[engine.METHOD] = score(rows, engine.METHOD, truth, history, catalog, out).merge(
        rows[KEY + ["method_note"]].astype({"origin_date": str}), on=KEY, how="left")
    print(f"  {label}: {len(cells[engine.METHOD])} scored cells, "
          f"{cells[engine.METHOD]['series_id'].nunique()} series, {len(origins)} origins", flush=True)
    return cells


def bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """``a`` against ``b`` on matched cells: 1 - loss_a / loss_b, its 90% interval and P(a better),
    resampling (series, origin) blocks."""
    m = a[KEY + ["loss"]].merge(b[KEY + ["loss"]], on=KEY, suffixes=("_a", "_b"))
    sums = [(g["loss_a"].sum(), g["loss_b"].sum()) for _, g in m.groupby(["series_id", "origin_date"])]
    rng = random.Random(seed)

    def stat(sample) -> float:
        return 1 - sum(x for x, _ in sample) / max(sum(y for _, y in sample), 1e-12)

    draws = sorted(stat(rng.choices(sums, k=len(sums))) for _ in range(n))
    return {"less_error": stat(sums), "lo90": draws[int(0.05 * n)], "hi90": draws[int(0.95 * n)],
            "p_better": sum(d > 0 for d in draws) / n, "cells": len(m)}


def pooled(cells: pd.DataFrame) -> float | None:
    base = cells["baseline_loss"].sum()
    return float(1 - cells["loss"].sum() / base) if base > 0 else None


def summarise(cells: dict[str, pd.DataFrame], classes: pd.Series, headline: set[str]) -> dict:
    eng, drift = cells[engine.METHOD], cells["drift"]
    out: dict = {"cells": len(eng), "series": int(eng["series_id"].nunique()),
                 "origins": int(eng["origin_date"].nunique()),
                 "engine_cells_share": float((eng["method_note"] == engine.METHOD).mean()), "methods": {}}
    for name, c in cells.items():
        out["methods"][name] = {
            "pooled_skill": pooled(c), "inside": float(((c["y"] >= c["q10"]) & (c["y"] <= c["q90"])).mean()),
            "by_horizon": {int(h): pooled(g) for h, g in c.groupby("horizon_days")}}
    out["engine_vs_drift"] = bootstrap(eng, drift)
    out["engine_vs_last_value_empirical"] = bootstrap(eng, cells["last_value_empirical"])
    out["engine_vs_drift_by_horizon"] = {int(h): bootstrap(g, drift[drift["horizon_days"] == h])
                                         for h, g in eng.groupby("horizon_days")}
    cls = eng["series_id"].map(classes)
    out["engine_vs_drift_by_class"] = {str(k): bootstrap(g, drift) | {"series": int(g["series_id"].nunique())}
                                       for k, g in eng.groupby(cls)}
    sunday = eng[eng["series_id"].isin(headline)]
    if len(sunday):
        out["engine_vs_drift_sunday_series"] = bootstrap(sunday, drift) | {
            "series": int(sunday["series_id"].nunique()), "of_series": len(headline)}
    return out


def placebo_world(seed: int, sids: list[str], offers, cap: date, series: pd.DataFrame, out: Path):
    """Shuffled offer tables (cut at the truth cap first) and a series file holding their medians."""
    tables = {}
    for sid in sids:
        full = offers.full_table(sid)
        cut = full.cut(cap) if full is not None else None
        if cut is not None and cut.prices.shape[1]:
            tables[sid] = make_placebo(cut, seed)
    value = {(sid, date.fromordinal(int(d)).isoformat()): float(v)
             for sid, t in tables.items() for d, v in zip(t.days, t.value)}
    rows = series[series["series_id"].isin(tables) & (series["date"] <= cap.isoformat())].copy()
    rows["value"] = [value[(s, d)] for s, d in zip(rows["series_id"], rows["date"])]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "series.csv"
    rows.to_csv(path, index=False)
    return path, (lambda sid, origin: tables[sid].cut(origin) if sid in tables else None)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def vs_line(b: dict) -> str:
    return f"{pct(b['less_error'])} ({pct(b['lo90'])} to {pct(b['hi90'])}), P(better) {b['p_better']:.2f}"


def markdown(res: dict) -> str:
    lines = ["# The list-replay engine", "",
             "Generated by `scripts/engine_check.py`: the pack, the starter kit's scripts and `gpr/engine.py` "
             "only. No LLM, nothing trained, and the locked test is not touched.", "",
             "Skill is Pacific's pooled skill against the last value. 'Less error than drift' is "
             "1 − pinball(engine) / pinball(drift) on the same cells, with a 90% interval from resampling "
             "(series, origin) blocks. 'Inside' is the share of truths in [q10, q90] (target 0.80).", ""]
    v = res["offer_lists"]
    lines += ["## Offer lists", "",
              f"{v['series']} posted rental series have an offer list. The median over a day's offers reproduces the "
              f"official value on {100 * v['match_share']:.2f}% of series-days "
              f"({v['perfect_series']} of {v['series']} series on every day).", ""]
    for key, title in (("validation", "Validation: origins 1, 8 and 15 Oct 2026, truth to 24 Oct"),
                       ("train", "Training window: weekly origins 4 Feb to 30 Sep 2026, truth to 30 Sep")):
        r = res[key]
        lines += [f"## {title}", "",
                  f"Rental scope, h = 1, 3, 7: {r['cells']} cells, {r['series']} series, {r['origins']} origins. "
                  f"The engine itself forecast {100 * r['engine_cells_share']:.0f}% of the cells; the rest fall back to "
                  "drift (no list on the origin day, or fewer than 10 replayable pairs of days).", "",
                  "| Forecaster | Pooled skill | h=1 | h=3 | h=7 | Inside |", "|---|---|---|---|---|---|"]
        for name, m in r["methods"].items():
            bh = m["by_horizon"]
            lines.append(f"| {name} | {pct(m['pooled_skill'])} | {pct(bh.get(1))} | {pct(bh.get(3))} | "
                         f"{pct(bh.get(7))} | {m['inside']:.2f} |")
        lines += ["", "| Engine against | Less error |", "|---|---|",
                  f"| drift | {vs_line(r['engine_vs_drift'])} |",
                  f"| last_value_empirical | {vs_line(r['engine_vs_last_value_empirical'])} |"]
        for h, b in r["engine_vs_drift_by_horizon"].items():
            lines.append(f"| drift, h={h} | {vs_line(b)} |")
        if "engine_vs_drift_sunday_series" in r:
            b = r["engine_vs_drift_sunday_series"]
            lines.append(f"| drift, series scored on Sunday only ({b['series']} of {b['of_series']} in this window) | "
                         f"{vs_line(b)} |")
        lines += ["", "| Instrument class | Series | Cells | Engine against drift |", "|---|---|---|---|"]
        for cls, b in sorted(r["engine_vs_drift_by_class"].items()):
            lines.append(f"| {cls} | {b['series']} | {b['cells']} | {vs_line(b)} |")
        lines.append("")
    if res.get("placebo"):
        real = res["validation"]["engine_vs_drift"]
        lines += ["## Shuffled-data control (validation origins)", "",
                  "Each shuffled copy keeps every offer and its listing pattern, and applies the real day-to-day price "
                  "changes in a scrambled order. A change that lands on a day its offer is not listed is dropped, so a "
                  "copy holds about four in five of the real changes. Baselines, engine and truth are all recomputed "
                  "inside the copy. This is a rough control.", "", "| World | Cells | Engine against drift |", "|---|---|---|",
                  f"| real | {real['cells']} | {vs_line(real)} |"]
        for p in res["placebo"]:
            lines.append(f"| shuffled, seed {p['seed']} | {p['engine_vs_drift']['cells']} | {vs_line(p['engine_vs_drift'])} |")
        lines.append("")
    if res.get("median_moves"):
        m = res["median_moves"]
        lines += ["## What moves a median (data to 30 Sep 2026)", "",
                  f"Each day-to-day move of a median is split, in order, into offers joining or leaving the list, "
                  f"repricing by frequent repricers (offers that changed price on at least 30% of their listed days) and "
                  f"repricing by every other offer. Shares are of the summed absolute log move over {m['series']} series.",
                  "", "| Instrument class | Series | List changes | Frequent repricers | Other repricing | Typical offers | Days the median moved |",
                  "|---|---|---|---|---|---|---|"]
        for row in m["by_class"] + [m["all"]]:
            lines.append(f"| {row['instrument_class']} | {row['series']} | {100 * row['list']:.0f}% | "
                         f"{100 * row['frequent']:.0f}% | {100 * row['rare']:.0f}% | {row['median_offers']:.0f} | "
                         f"{100 * row['share_of_days_moved']:.0f}% |")
        lines.append("")
    return "\n".join(lines)


def moves_table(tables: list) -> dict:
    df = pd.DataFrame([median_moves(t) for t in tables])
    df = df[df[["list", "frequent", "rare"]].sum(axis=1) > 0]

    def row(name: str, g: pd.DataFrame) -> dict:
        total = g[["list", "frequent", "rare"]].sum().sum()
        return {"instrument_class": name, "series": len(g), "median_offers": float(g["offers"].median()),
                "share_of_days_moved": float(g["share_of_days_moved"].median()),
                **{k: float(g[k].sum() / total) for k in ("list", "frequent", "rare")}}

    return {"series": len(df), "by_class": [row(str(k), g) for k, g in df.groupby("instrument_class")],
            "all": row("all", df)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--placebos", type=int, default=5, help="number of shuffled copies (0 to skip)")
    args = parser.parse_args()

    splits = packlib.Splits()
    splits.check()
    files = packlib.build(packlib.DATA_CUTOFF)
    pack = packlib.Pack(files)
    offers = shared(pack)
    out = ROOT / "outputs" / "engine_check"
    classes = pack.catalog["instrument_class"]
    headline = set(packlib.series_at(files, packlib.DATA_CUTOFF, "headline"))
    res: dict = {}

    listed = [t for t in (offers.full_table(s) for s in pack.catalog.index) if t is not None]
    v = verify(listed)
    days = v["n_days"] * v["days_with_offers"]
    res["offer_lists"] = {"series": len(v), "perfect_series": int((v["match_share"] == 1).sum()),
                          "match_share": float((v["match_share"] * days).sum() / days.sum())}
    print(f"offer lists: {len(v)} series, median reproduced on {100 * res['offer_lists']['match_share']:.2f}% of "
          f"series-days", flush=True)

    scopes = {}
    for name, origins, cap in (("validation", list(splits.validation), VALIDATION_TRUTH_CAP),
                               ("train", list(splits.train), TRAIN_TRUTH_CAP)):
        scopes[name] = {o: packlib.rental_scope(files, o) for o in origins}
        cells = run_world(name, origins, scopes[name], offers.table, files.series, capped_truth(files, cap),
                          files.catalog, out / name)
        res[name] = summarise(cells, classes, headline)
        print(f"  engine against drift: {vs_line(res[name]['engine_vs_drift'])}", flush=True)

    if args.placebos:
        origins, scope = list(splits.validation), scopes["validation"]
        sids = sorted({s for ids in scope.values() for s in ids})
        series = pd.read_csv(files.series, dtype=str)
        res["placebo"] = []
        for seed in range(1, args.placebos + 1):
            path, table_for = placebo_world(seed, sids, offers, VALIDATION_TRUTH_CAP, series, out / f"placebo{seed}")
            cells = run_world(f"shuffled copy {seed}", origins, scope, table_for, path, path, files.catalog,
                              out / f"placebo{seed}")
            res["placebo"].append({"seed": seed, "engine_vs_drift": bootstrap(cells[engine.METHOD], cells["drift"])})
            print(f"  engine against drift: {vs_line(res['placebo'][-1]['engine_vs_drift'])}", flush=True)

    scope_ids = sorted({s for ids in scopes["validation"].values() for s in ids})
    cut = [t for t in (offers.table(s, TRAIN_TRUTH_CAP) for s in scope_ids) if t is not None and len(t.days) > 1]
    res["median_moves"] = moves_table(cut)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "engine.md").write_text(markdown(res))
    print(f"wrote results/engine.md (working files in {out})")


if __name__ == "__main__":
    main()
