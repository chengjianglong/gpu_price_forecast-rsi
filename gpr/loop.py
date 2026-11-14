"""The recursive self-improvement loop (docs/PLAN.md section 2.2).

Each generation:

1. **Train:** every population member runs ``steps_per_generation`` RL steps on
   its own LoRA adapter.
2. **Evaluate:** every member's full team (debate and voting) runs the
   validation origins and is scored with Pacific's scorer. Voting weights
   restart equal and update along the origins (``evaluate.update_weights``).
3. **Accept:** the best member replaces the incumbent only if a paired
   bootstrap says it beats it with ``p_better >= accept_p``. Generation 0's
   incumbent is Pacific's ``last_value_empirical`` baseline.
4. **Exploit and explore:** the bottom half copy the best member's adapter
   (``save_state`` then a new training client from it) and perturb its
   learning rate, sampling temperature and voting rule. One of them instead
   takes the Improver's edits to the best member's team config.
5. **Archive** every member and generation; **stop** on a plateau, on
   overfitting, or at the budget.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import pack as packlib
from .evaluate import EvalResult, capped_truth, evaluate, paired_bootstrap
from .improve import breakdown, propose
from .research import Notebook
from .rl import RLConfig, Trainer, build_pool
from .rewards import TRAIN_TRUTH_CAP, VALIDATION_TRUTH_CAP
from .team import Team, TeamConfig
from .weights import VotingWeights


@dataclass
class LoopConfig:
    population: int = 3
    steps_per_generation: int = 6
    max_generations: int = 20
    plateau_generations: int = 3
    overfit_generations: int = 2
    accept_p: float = 0.9
    deadline: str | None = None           # local ISO time; stop starting new generations after it
    max_series_per_origin: int = 40
    learning_rates: tuple[float, ...] = (5e-6, 1e-5, 2e-5)
    lr_bounds: tuple[float, float] = (1e-6, 1e-4)
    use_improver: bool = True
    use_research: bool = True
    hypotheses_per_generation: int = 2
    seed: int = 0


@dataclass
class Member:
    mid: str
    parent: str | None
    team_cfg: TeamConfig
    rl_cfg: RLConfig
    trainer: Trainer
    note: str = ""
    val: float = math.nan
    train_reward: float = math.nan
    result: EvalResult | None = None


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def baseline_cells(files: packlib.PackFiles, origins: list[date], out_dir: Path,
                   method: str = "last_value_empirical") -> tuple[dict, pd.DataFrame]:
    """Pacific's baseline on the validation origins, scored like the members."""
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast = out_dir / f"baseline_{method}.csv"
    packlib.baselines(files.series, files.catalog, method, forecast, origins=origins, tier="all")
    df = pd.read_csv(forecast)
    keep = set()
    for o in origins:
        keep |= {(s, o.isoformat()) for s in packlib.rental_scope(files, o)}
    df = df[[(s, o) in keep for s, o in zip(df["series_id"], df["origin_date"])]]
    df.to_csv(forecast, index=False)
    report = packlib.score(forecast, capped_truth(files, VALIDATION_TRUTH_CAP), files.series, files.catalog,
                           out_dir / "report_baseline.json", out_dir / "cells_baseline.csv", tier="all")
    cells = pd.read_csv(out_dir / "cells_baseline.csv")
    return report, cells[cells["status"] == "scored"].reset_index(drop=True)


def run(team_cfg: TeamConfig, rl_cfg: RLConfig, loop_cfg: LoopConfig, out_dir: Path,
        llm_factory: Any = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(loop_cfg.seed)
    splits = packlib.Splits()
    splits.check()
    files = packlib.build(packlib.DATA_CUTOFF)
    pack = packlib.Pack(files)
    val_origins = list(splits.validation)
    archive = (out_dir / "archive.jsonl").open("a")

    def record(**row: Any) -> None:
        archive.write(json.dumps(row, default=str) + "\n")
        archive.flush()

    # Reference: Pacific's best simple baseline on the same validation cells.
    base_report, incumbent_cells = baseline_cells(files, val_origins, out_dir / "baseline")
    incumbent = {"mid": "baseline:last_value_empirical",
                 "val": (base_report.get("overall") or {}).get("skill_pooled"), "checkpoint": None}
    log(f"baseline last_value_empirical on validation (rental scope, h=1/3/7): pooled {incumbent['val']}")
    record(kind="baseline", **incumbent)

    # Research notebook: hypotheses tested on the training window only.
    notebook = Notebook(out_dir / "notebook.jsonl")

    # Training pool: truth dated on or before TRAIN_TRUTH_CAP only.
    helper = Team(team_cfg, None, pack)

    def make_pool():
        return build_pool(pack, list(splits.train), lambda o: packlib.rental_scope(files, o),
                          helper.baselines_for, team_cfg.baseline_methods, cap=TRAIN_TRUTH_CAP,
                          max_series_per_origin=loop_cfg.max_series_per_origin, seed=loop_cfg.seed,
                          findings=notebook.findings() if team_cfg.use_findings else ())

    if loop_cfg.use_research:
        for r in notebook.iterate(pack, None, iteration=0):
            log(f"research {r.hid} [{'survived' if r.survived else 'rejected'}] {r.hypothesis.statement} -> {r.summary}, p={r.p_value:.3f}")
    pool = make_pool()
    log(f"training pool: {len(pool)} examples from {len(splits.train)} origins (truth <= {TRAIN_TRUTH_CAP})")

    lenses = team_cfg.lenses
    members = []
    for i in range(loop_cfg.population):
        rc = replace(rl_cfg, learning_rate=loop_cfg.learning_rates[i % len(loop_cfg.learning_rates)])
        members.append(Member(f"m{i}", None, copy.deepcopy(team_cfg), rc, Trainer(rc, lenses, seed=loop_cfg.seed + i),
                              note="initial"))

    plateau, history, best_val_trend, best_train_trend = 0, [], [], []
    deadline = datetime.fromisoformat(loop_cfg.deadline) if loop_cfg.deadline else None
    stop_reason = "max_generations"
    for gen in range(loop_cfg.max_generations):
        if deadline and datetime.now() >= deadline:
            stop_reason = "deadline"
            break
        t0 = time.time()
        if loop_cfg.use_research and gen > 0:
            before = len(notebook.findings())
            for r in notebook.iterate(pack, members[0].trainer.sampler_llm(), iteration=gen,
                                      n=loop_cfg.hypotheses_per_generation):
                log(f"research {r.hid} [{'survived' if r.survived else 'rejected'}] {r.hypothesis.family} "
                    f"{json.dumps(r.hypothesis.params)}: {r.hypothesis.statement} -> {r.summary}, p={r.p_value:.3f}")
                record(kind="hypothesis", gen=gen, **r.to_json())
            if len(notebook.findings()) > before:
                pool = make_pool()
        for m in members:
            stats = [m.trainer.step(pool) for _ in range(loop_cfg.steps_per_generation)]
            m.train_reward = sum(s.mean_reward for s in stats) / max(1, len(stats))
            team = Team(m.team_cfg, m.trainer.sampler_llm(), pack, findings=notebook.findings(),
                        weights=VotingWeights([l.name for l in m.team_cfg.lenses], m.team_cfg.rule))
            m.result = evaluate(team, val_origins, out_dir / f"gen{gen:02d}" / m.mid)
            m.val = m.result.pooled
            log(f"gen {gen} {m.mid} ({m.note}): train reward {m.train_reward:+.3f}, validation pooled {m.val:+.4f} "
                f"| {json.dumps(m.result.summary(), default=str)}")
            record(kind="member", gen=gen, mid=m.mid, parent=m.parent, note=m.note, val=m.val,
                   summary=m.result.summary(), train_reward=m.train_reward, rl=asdict(m.rl_cfg),
                   team=m.team_cfg.to_json(), steps=[asdict(s) for s in stats])

        ranked = sorted(members, key=lambda m: -m.val if math.isfinite(m.val) else math.inf)
        best = ranked[0]
        test = paired_bootstrap(best.result.cells, incumbent_cells)
        accepted = test["p_better"] >= loop_cfg.accept_p and test["delta"] > 0
        if accepted:
            path = best.trainer.save_state(f"gen{gen:02d}_{best.mid}")
            sampler_path = best.trainer.save_sampler(f"gen{gen:02d}_{best.mid}_sampler")
            incumbent = {"mid": f"gen{gen}:{best.mid}", "val": best.val, "checkpoint": path,
                         "sampler_checkpoint": sampler_path,
                         "team": best.team_cfg.to_json(), "weights": best.result.directory / "weights.json"}
            incumbent_cells = best.result.cells
            plateau = 0
            (out_dir / "best.json").write_text(json.dumps(incumbent, indent=1, default=str))
        else:
            plateau += 1
        history.append(f"gen {gen}: best {best.mid} ({best.note}) val {best.val:+.4f}; "
                       f"{'accepted' if accepted else 'not accepted'} (p_better {test['p_better']:.2f})")
        log(history[-1] + f"; generation took {time.time() - t0:.0f} s")
        record(kind="generation", gen=gen, best=best.mid, accepted=accepted, bootstrap=test, incumbent=incumbent)

        best_val_trend.append(best.val)
        best_train_trend.append(best.train_reward)
        k = loop_cfg.overfit_generations
        if len(best_val_trend) > k and all(best_train_trend[-i] > best_train_trend[-i - 1] for i in range(1, k + 1)) \
                and all(best_val_trend[-i] < best_val_trend[-i - 1] for i in range(1, k + 1)):
            stop_reason = "overfitting"
            break
        if plateau >= loop_cfg.plateau_generations:
            stop_reason = "plateau"
            break

        # Exploit and explore: the bottom half restart from the best member.
        losers = ranked[len(ranked) - len(ranked) // 2:]
        if losers:
            state = best.trainer.save_state(f"gen{gen:02d}_{best.mid}_seed")
            for j, m in enumerate(losers):
                lr = best.rl_cfg.learning_rate * rng.choice((0.5, 2.0))
                lr = min(loop_cfg.lr_bounds[1], max(loop_cfg.lr_bounds[0], lr))
                temp = min(1.1, max(0.5, best.rl_cfg.temperature + rng.choice((-0.1, 0.1))))
                rc = replace(best.rl_cfg, learning_rate=lr, temperature=temp)
                tc = copy.deepcopy(best.team_cfg)
                tc.rule = replace(tc.rule, eta=min(10.0, max(0.1, tc.rule.eta * rng.choice((0.5, 2.0)))),
                                  widen=min(2.0, max(0.0, tc.rule.widen + rng.choice((-0.2, 0.2)))))
                note = f"copy of {best.mid}: lr {lr:.1e}, temp {temp:.1f}, eta {tc.rule.eta:g}, widen {tc.rule.widen:g}"
                if loop_cfg.use_improver and j == 0 and best.result is not None:
                    table = breakdown(best.result.cells, pack.catalog)
                    tc, edits, why = propose(best.trainer.sampler_llm(), best.team_cfg, best.result.summary(), table, history)
                    note = f"Improver on {best.mid}: {', '.join(edits) or 'no valid edit'} ({why[:120]})"
                m.parent, m.note, m.rl_cfg, m.team_cfg = best.mid, note, rc, tc
                m.trainer = Trainer(rc, tc.lenses, from_state=state, seed=loop_cfg.seed + 100 * gen + j)
                m.mid = f"m{gen + 1}_{j}"
    archive.close()
    summary = {"stop_reason": stop_reason, "incumbent": incumbent, "history": history,
               "findings": [r.to_json() for r in notebook.findings()], "hypotheses_tested": len(notebook.results)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"stopped: {stop_reason}; incumbent {incumbent['mid']} validation pooled {incumbent['val']}")
    return summary
