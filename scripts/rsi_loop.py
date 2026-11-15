"""Run the RL-driven self-improvement loop.

    python scripts/rsi_loop.py --deadline 2026-11-15T09:00
    python scripts/rsi_loop.py --generations 1 --steps 2 --population 2   # a short check

Writes outputs/rsi/<run>/: archive.jsonl (every member and generation),
best.json (the accepted incumbent and its Tinker checkpoint), summary.json,
and each member's validation forecasts, transcripts, weights and scores.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr.loop import LoopConfig, run  # noqa: E402
from gpr.rl import RLConfig  # noqa: E402
from gpr.team import TeamConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", default=str(ROOT / "configs" / "team_v0.yaml"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "rsi.yaml"))
    parser.add_argument("--out")
    parser.add_argument("--deadline", help="local ISO time, e.g. 2026-11-15T09:00")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model", help="Tinker base model, e.g. deepseek-ai/DeepSeek-V4.1-Flash:peft:262144")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    rl = RLConfig(**raw.get("rl", {}))
    loop_raw = raw.get("loop", {})
    for key in ("learning_rates", "lr_bounds"):
        if key in loop_raw:
            loop_raw[key] = tuple(loop_raw[key])
    loop = LoopConfig(**loop_raw)
    if args.deadline:
        loop.deadline = args.deadline
    if args.generations:
        loop.max_generations = args.generations
    if args.steps:
        loop.steps_per_generation = args.steps
    if args.population:
        loop.population = args.population
    if args.batch_size:
        rl.batch_size = args.batch_size
    if args.model:
        rl.base_model = args.model
    team = TeamConfig.load(args.team)
    team.base_model = rl.base_model
    out = Path(args.out) if args.out else ROOT / "outputs" / "rsi" / f"{datetime.now():%Y%m%d-%H%M%S}"
    run(team, rl, loop, out)


if __name__ == "__main__":
    main()
