"""Final evaluation and the submission, from the loop's accepted incumbent.

    python scripts/final.py --best outputs/rsi/<run>/best.json
    python scripts/final.py --base            # the untrained team (no checkpoint needed)

Steps:

1. **Locked test** (24, 27 and 30 Oct; nothing in training or validation saw
   it). Each origin runs in date order and is scored by Pacific's scorer in
   frozen-cutoff mode against the pack's truth. Variants on the same cells:
   - the trained team (the incumbent's checkpoint, team config and learned voting weights);
   - the same team without research findings (``use_findings`` off);
   - the untrained team (base model);
   - Pacific's ``drift`` and ``last_value_empirical`` baselines.

   Pooled skill gets a 90% bootstrap interval over (series, origin) blocks, and
   each variant a paired bootstrap against ``drift``.
2. **Submission:** the trained team at the 6 Nov origin on the headline tier
   (the 37 posted rental series Sunday scores), checked in frozen-cutoff mode.
   Written to ``submission/forecast_2026-11-06.csv`` with its debate transcript.
3. ``results/final.json`` collects everything for ``scripts/report.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr import pack as packlib  # noqa: E402
from gpr.evaluate import paired_bootstrap, update_weights  # noqa: E402
from gpr.llm import TinkerLLM  # noqa: E402
from gpr.research import Notebook  # noqa: E402
from gpr.team import Team, TeamConfig, write_outputs  # noqa: E402
from gpr.weights import VotingWeights, WeightRule  # noqa: E402


def load_weights(path: Path | None, agents: list[str], rule: WeightRule) -> VotingWeights:
    w = VotingWeights(agents, rule)
    if path and Path(path).is_file():
        for key, table in json.loads(Path(path).read_text()).items():
            cls, h = key.rsplit("|", 1)
            w.table[(cls, int(h))] = {a: float(v) for a, v in table.items()}
    return w


def pooled(cells: pd.DataFrame) -> float:
    return float(1 - cells["loss"].sum() / max(cells["baseline_loss"].sum(), 1e-12))


def interval(cells: pd.DataFrame, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    blocks = [g for _, g in cells.groupby(["series_id", "origin_date"])]
    sums = [(g["loss"].sum(), g["baseline_loss"].sum()) for g in blocks]
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        s = rng.choices(sums, k=len(sums))
        draws.append(1 - sum(a for a, _ in s) / max(sum(b for _, b in s), 1e-12))
    draws.sort()
    return draws[int(0.05 * n)], draws[int(0.95 * n)]


def score_frozen(files, forecast: Path, origin: date, out: Path) -> pd.DataFrame:
    history = packlib.build_history(files, origin)
    report = packlib.score(forecast, files.series, history, files.history_catalog(origin), out / "report.json",
                           out / "cells.csv", frozen_cutoff=origin, tier="all")
    if report.get("status") == "refused":
        raise SystemExit(f"scorer refused the {origin} forecast: {report.get('refusals')}")
    cells = pd.read_csv(out / "cells.csv")
    return cells[cells["status"] == "scored"]


def run_team(label: str, cfg: TeamConfig, llm, pack, files, findings, weights, out_root: Path) -> pd.DataFrame:
    team = Team(cfg, llm, pack, findings=findings, weights=weights)
    origins = sorted(packlib.Splits().test)
    frames = []
    for i, origin in enumerate(origins):
        out = out_root / label / origin.isoformat()
        result = team.run(origin, packlib.rental_scope(files, origin))
        paths = write_outputs(result, out, cfg)
        frames.append(score_frozen(files, paths["forecast"], origin, out).assign(variant=label))
        nxt = origins[i + 1] if i + 1 < len(origins) else packlib.DATA_CUTOFF
        update_weights(team, result, nxt, packlib.DATA_CUTOFF)
        print(f"  {label} {origin}: {len(result.boards)} series, {result.calls} calls", flush=True)
    return pd.concat(frames, ignore_index=True)


def run_baseline(method: str, files, out_root: Path) -> pd.DataFrame:
    frames = []
    for origin in sorted(packlib.Splits().test):
        out = out_root / method / origin.isoformat()
        out.mkdir(parents=True, exist_ok=True)
        history = packlib.build_history(files, origin)
        forecast = out / "forecast.csv"
        df = packlib.baselines(history, files.history_catalog(origin), method, forecast, origin=origin, tier="all",
                               frozen=True)
        keep = set(packlib.rental_scope(files, origin))
        df[df["series_id"].isin(keep)].to_csv(forecast, index=False)
        frames.append(score_frozen(files, forecast, origin, out).assign(variant=method))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best", help="best.json written by scripts/rsi_loop.py")
    parser.add_argument("--base", action="store_true", help="use the untrained team only (no checkpoint)")
    parser.add_argument("--team", default=str(ROOT / "configs" / "team_v0.yaml"))
    parser.add_argument("--skip-test", action="store_true", help="only write the submission")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    files = packlib.build(packlib.DATA_CUTOFF)
    pack = packlib.Pack(files)
    base_cfg = TeamConfig.load(args.team)
    best = json.loads(Path(args.best).read_text()) if args.best else {}
    cfg, checkpoint, weights_path, findings = base_cfg, None, None, []
    if best.get("checkpoint") and not args.base:
        cfg = TeamConfig.load(args.team)
        team_json = best.get("team") or {}
        cfg = replace(cfg, **{k: v for k, v in team_json.items() if k in ("temperature", "debate_rounds",
                                                                             "debate_min_disagreement", "use_findings")})
        if "rule" in team_json:
            cfg.rule = WeightRule(**team_json["rule"])
        checkpoint, weights_path = best.get("sampler_checkpoint") or best["checkpoint"], best.get("weights")
        notebook = Path(args.best).parent / "notebook.jsonl"
        findings = Notebook(notebook).findings() if notebook.is_file() else []
    agents = [l.name for l in cfg.lenses]

    trained = TinkerLLM(cfg.base_model, model_path=checkpoint) if checkpoint else TinkerLLM(cfg.base_model)
    label = "trained_team" if checkpoint else "base_team"
    results: dict = {"checkpoint": checkpoint, "team": cfg.to_json(), "findings": [r.finding_text() for r in findings]}
    out_root = ROOT / "results" / "test"

    if not args.skip_test:
        print("locked test origins:", [str(o) for o in packlib.Splits().test], flush=True)
        frames = [run_team(label, cfg, trained, pack, files, findings, load_weights(weights_path, agents, cfg.rule), out_root)]
        if checkpoint and not args.skip_ablations:
            frames.append(run_team("trained_no_findings", replace(cfg, use_findings=False), trained, pack, files, [],
                                   load_weights(weights_path, agents, cfg.rule), out_root))
            frames.append(run_team("base_team", base_cfg, TinkerLLM(base_cfg.base_model), pack, files, [],
                                   VotingWeights(agents, base_cfg.rule), out_root))
        for method in ("drift", "last_value_empirical"):
            frames.append(run_baseline(method, files, out_root))
        cells = pd.concat(frames, ignore_index=True)
        cells.to_csv(out_root / "all_cells.csv", index=False)
        drift = cells[cells["variant"] == "drift"]
        headline = set(packlib.series_at(files, packlib.DATA_CUTOFF, "headline"))
        table = {}
        for variant, g in cells.groupby("variant"):
            lo, hi = interval(g)
            vs = paired_bootstrap(g, drift) if variant != "drift" else {}
            inside = float(((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean())
            sunday = g[g["series_id"].isin(headline)]
            table[variant] = {"pooled": pooled(g), "pooled_90ci": [lo, hi], "cells": len(g), "coverage_80": inside,
                              "vs_drift": vs, "sunday_series_pooled": pooled(sunday) if len(sunday) else None,
                              "by_horizon": {int(h): pooled(x) for h, x in g.groupby("horizon_days")}}
            print(f"  {variant:22s} pooled {table[variant]['pooled']:+.4f} [{lo:+.4f}, {hi:+.4f}], "
                  f"coverage {inside:.2f}, vs drift {vs}", flush=True)
        results["test"] = table

    # Submission: origin 6 Nov, the headline tier Sunday scores.
    origin = packlib.DATA_CUTOFF
    series = packlib.series_at(files, origin, "headline")
    team = Team(cfg, trained, pack, findings=findings, weights=load_weights(weights_path, agents, cfg.rule))
    result = team.run(origin, series)
    rows = result.rows[result.rows["horizon_days"].isin(packlib.SCORED_HORIZONS)].copy()
    rows["team_id"] = "gpu-pricing-rsi"
    result.rows = rows
    sub = ROOT / "submission"
    paths = write_outputs(result, sub, cfg)
    history = packlib.build_history(files, origin)
    check = packlib.score(paths["forecast"], files.series, history, files.history_catalog(origin), sub / "frozen_check.json",
                          frozen_cutoff=origin, tier="headline")
    status = check.get("status")
    print(f"submission: {len(rows)} rows, {rows['series_id'].nunique()} series; frozen check status {status} "
          f"(exit {check.get('_exit_code')}); invalid rows {((check.get('forecast_validation') or {}).get('problems') or [])[:3]}")
    if status == "refused":
        raise SystemExit("the frozen-cutoff check refused the submission")
    results["submission"] = {"rows": len(rows), "series": int(rows["series_id"].nunique()), "frozen_check": status,
                             "file": str(paths["forecast"].relative_to(ROOT))}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "final.json").write_text(json.dumps(results, indent=1, default=str))
    if args.best:  # keep the run's record next to the results (skipped when already there)
        src = Path(args.best).resolve().parent
        for name in ("best.json", "notebook.jsonl", "archive.jsonl", "summary.json"):
            if (src / name).is_file() and src != (ROOT / "results").resolve():
                shutil.copy(src / name, ROOT / "results" / name)
    print(f"wrote {sub} and results/final.json")


if __name__ == "__main__":
    main()
