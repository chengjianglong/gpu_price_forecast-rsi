"""Run the team for one origin, write the forecast file and score it with Pacific's scorer.

    python scripts/run_origin.py --origin 2026-10-24 --llm fake
    python scripts/run_origin.py --origin 2026-10-24 --llm tinker --limit 5
    python scripts/run_origin.py --origin 2026-11-06 --llm tinker          # the submission origin

Earlier origins are scored against the pack's own truth at the 1-, 3- and
7-day horizons Sunday scores. By default they use the rental scope (posted
rental series with daily truth, `gpr.pack.rental_scope`). The three locked
test origins (24, 27, 30 Oct) and the 6 Nov submission origin are checked in
Pacific's frozen-cutoff mode; for 6 Nov the truth is withheld, so only the
validation and refusal checks run, on the headline tier Sunday scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr import pack as packlib  # noqa: E402
from gpr.data import PriceStore  # noqa: E402
from gpr.team import Team, TeamConfig, write_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "team_v0.yaml"))
    parser.add_argument("--llm", choices=["fake", "tinker"], default="fake")
    parser.add_argument("--model", help="override base_model")
    parser.add_argument("--model-path", help="tinker:// checkpoint to sample from")
    parser.add_argument("--tier", default="headline")
    parser.add_argument("--scope", choices=["rental", "tier"], default="rental",
                        help="rental: posted rental series with daily truth (what Sunday scores); tier: the --tier set")
    parser.add_argument("--limit", type=int, help="first N series only (for quick checks)")
    parser.add_argument("--series", nargs="*", help="explicit series ids")
    parser.add_argument("--events", default=str(ROOT / "data"), help="directory with Exa events.jsonl")
    parser.add_argument("--out", help="output directory (default outputs/<timestamp>_<origin>)")
    args = parser.parse_args()

    origin = date.fromisoformat(args.origin)
    cfg = TeamConfig.load(args.config)
    if args.model:
        cfg.base_model = args.model

    files = packlib.build(packlib.DATA_CUTOFF)
    pack = packlib.Pack(files)
    if args.series:
        series = args.series
    elif args.scope == "rental" and origin != packlib.DATA_CUTOFF:
        series = packlib.rental_scope(files, origin)
    else:  # the submission: Sunday's scored set is the headline tier at the cutoff (all rental)
        series = packlib.series_at(files, origin, args.tier)
    if args.limit:
        series = series[: args.limit]
    events = PriceStore.load(args.events).as_of(f"{origin.isoformat()}T23:59:59Z").events()

    if args.llm == "fake":
        from gpr.fake import fake_llm

        llm = fake_llm()
    else:
        from gpr.llm import TinkerLLM

        llm = TinkerLLM(cfg.base_model, model_path=args.model_path)

    out = Path(args.out) if args.out else ROOT / "outputs" / f"{datetime.now():%Y%m%d-%H%M%S}_{origin}_{args.llm}"
    t0 = time.time()
    result = Team(cfg, llm, pack, events).run(origin, series)
    paths = write_outputs(result, out, cfg)
    elapsed = time.time() - t0
    fallbacks = sum(p.fallback for b in result.boards for p in list(b.positions.values()) + list(b.revised.values()))
    problems = sum(len(p.problems) for b in result.boards for p in b.positions.values())
    print(f"{len(result.boards)} series, {len(result.rows)} rows, {result.calls} LLM calls, "
          f"{elapsed:.0f} s; {fallbacks} fallback positions, {problems} parse problems; {len(events)} events")

    frozen = origin in (*packlib.Splits().test, packlib.DATA_CUTOFF)
    if frozen:
        history = packlib.build_history(files, origin)
        report = packlib.score(paths["forecast"], files.series, history, files.history_catalog(origin),
                               out / "report.json", out / "cells.csv", frozen_cutoff=origin,
                               tier=args.tier if origin == packlib.DATA_CUTOFF else "all")
    else:
        report = packlib.score(paths["forecast"], files.series, files.series, files.history_catalog(origin),
                               out / "report.json", out / "cells.csv", tier="all")
    overall = report.get("overall") or {}
    keys = ("skill_pooled", "skill_median_of_series", "skill_mean_of_series", "skill_geomean")
    print(f"scorer status {report.get('status')} (exit {report.get('_exit_code')}), mode {report.get('mode')}, "
          f"cells {json.dumps(report.get('cells'))}")
    print("  " + ", ".join(f"{k} {overall.get(k)}" for k in keys))
    print(f"  coverage {overall.get('coverage')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
