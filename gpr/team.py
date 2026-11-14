"""The team: runs the agents and the debate for one origin over many series.

Flow per origin (docs/PLAN.md section 1):

1. Pacific's baselines for the origin, then one shared board per series.
2. Market Analyst: one note from the news events, posted on every board.
3. Independent round: each Forecaster posts a position without seeing the others.
4. Up to ``debate_rounds`` rounds of Critic challenges and Forecaster revisions,
   only on series where the Forecasters disagree enough (``debate_min_disagreement``).
5. Aggregator: weighted vote per quantile, widened by the remaining disagreement.

Calls are batched across series: every phase is one ``LLM.generate`` call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from . import pack as packlib
from .agents import (DEFAULT_LENSES, Lens, critic_chat, forecaster_chat, market_chat, parse_challenges,
                     parse_forecast, revision_chat)
from .board import Board, baseline_table, board_to_json, build_board
from .llm import LLM
from .pack import HORIZONS, Pack
from .weights import VotingWeights, WeightRule, aggregate, disagreement

FALLBACK_METHOD = "last_value_empirical"


@dataclass
class TeamConfig:
    name: str = "team_v0"
    base_model: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.7
    max_tokens: int = 600
    baseline_methods: tuple[str, ...] = ("last_value_empirical", "drift")
    lenses: tuple[Lens, ...] = DEFAULT_LENSES
    debate_rounds: int = 1
    debate_min_disagreement: float = 0.0
    use_market_analyst: bool = True
    max_events: int = 6
    rule: WeightRule = field(default_factory=WeightRule)

    @classmethod
    def load(cls, path: str | Path) -> "TeamConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        lenses = tuple(Lens(**l) for l in raw.pop("lenses", [])) or DEFAULT_LENSES
        rule = WeightRule(**raw.pop("rule", {}))
        methods = tuple(raw.pop("baseline_methods", cls.baseline_methods))
        return cls(lenses=lenses, rule=rule, baseline_methods=methods, **raw)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["lenses"] = [asdict(l) for l in self.lenses]
        return out


@dataclass
class OriginResult:
    origin: date
    rows: pd.DataFrame
    boards: list[Board]
    calls: int


class Team:
    def __init__(self, config: TeamConfig, llm: LLM, pack: Pack, events: Iterable[Any] = (),
                 weights: VotingWeights | None = None, cache_dir: Path | None = None):
        self.cfg = config
        self.llm = llm
        self.pack = pack
        self.events = list(events)
        self.agents = [l.name for l in config.lenses]
        self.weights = weights or VotingWeights(self.agents, config.rule)
        self.cache_dir = cache_dir or pack.files.directory / "baselines"
        self.calls = 0

    # ---- baselines ----------------------------------------------------------
    def baselines_for(self, origin: date) -> dict:
        methods = tuple(dict.fromkeys((FALLBACK_METHOD,) + tuple(self.cfg.baseline_methods)))
        frames = []
        for method in methods:
            out = self.cache_dir / f"{method}_{origin.isoformat()}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.is_file():
                frames.append(pd.read_csv(out))
            else:
                frames.append(packlib.baselines(self.pack.files.series, self.pack.files.catalog, method, out,
                                                origin=origin, tier="all"))
        return baseline_table(pd.concat(frames, ignore_index=True))

    # ---- one origin -----------------------------------------------------------
    def _generate(self, chats: list) -> list[str]:
        if not chats:
            return []
        self.calls += len(chats)
        samples = self.llm.generate(chats, n=1, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
        return [s[0].text for s in samples]

    def run(self, origin: date, series_ids: Sequence[str]) -> OriginResult:
        table = self.baselines_for(origin)
        boards: list[Board] = []
        for sid in series_ids:
            by_method = table.get((sid, origin))
            if not by_method or FALLBACK_METHOD not in by_method:
                continue
            shown = {m: by_method[m] for m in self.cfg.baseline_methods if m in by_method}
            board = build_board(self.pack, sid, origin, shown, self.events, self.cfg.max_events)
            if board is not None:
                board.baselines[FALLBACK_METHOD] = by_method[FALLBACK_METHOD]
                boards.append(board)

        # Market Analyst: one note per origin from the events on the boards.
        if self.cfg.use_market_analyst:
            items = {e.eid: e for b in boards for e in b.events}
            if items:
                text = "\n".join(f"[{e.eid}] {e.text}" for e in items.values())
                note = self._generate([market_chat(text, origin.isoformat())])[0].strip()
                for b in boards:
                    b.market_note = note[:800]

        # Independent round.
        chats, keys = [], []
        for b in boards:
            for lens in self.cfg.lenses:
                chats.append(forecaster_chat(b, lens))
                keys.append((b, lens))
        for (b, lens), text in zip(keys, self._generate(chats)):
            b.positions[lens.name] = parse_forecast(text, lens.name, b.evidence_ids, b.baselines[FALLBACK_METHOD])

        # Debate rounds. The Critic sees the latest positions; the independent
        # round is restored on the boards afterwards for the transcript.
        independent = {id(b): dict(b.positions) for b in boards}
        current = {id(b): dict(b.positions) for b in boards}
        for _ in range(self.cfg.debate_rounds):
            active = [b for b in boards
                      if max(disagreement({a: p.quantiles[h] for a, p in current[id(b)].items()})
                             for h in HORIZONS) >= self.cfg.debate_min_disagreement]
            if not active:
                break
            for b in active:
                b.positions = current[id(b)]
            critiques = self._generate([critic_chat(b, self.agents) for b in active])
            for b, text in zip(active, critiques):
                found, _ = parse_challenges(text, "critic", self.agents, b.evidence_ids)
                b.challenges.extend(found)
            chats, keys = [], []
            for b in active:
                for lens in self.cfg.lenses:
                    chats.append(revision_chat(b, lens))
                    keys.append((b, lens))
            for (b, lens), text in zip(keys, self._generate(chats)):
                prior = current[id(b)][lens.name]
                revised = parse_forecast(text, lens.name, b.evidence_ids, prior.quantiles)
                b.revised[lens.name] = revised
                current[id(b)][lens.name] = revised
        for b in boards:
            b.positions = independent[id(b)]

        rows = []
        for b in boards:
            final = current[id(b)]
            cls = str(b.info.get("instrument_class"))
            for h in HORIZONS:
                w = self.weights.get((cls, h))
                q10, q50, q90 = aggregate({a: p.quantiles[h] for a, p in final.items()}, w, self.cfg.rule.widen)
                rows.append({
                    "series_id": b.series_id, "origin_date": origin.isoformat(), "horizon_days": h,
                    "q10": round(q10, 6), "q50": round(q50, 6), "q90": round(q90, 6),
                    "target_date": (origin + timedelta(days=h)).isoformat(), "method": self.cfg.name,
                    "method_note": "fallbacks=" + ",".join(a for a, p in final.items() if p.fallback),
                })
        return OriginResult(origin, pd.DataFrame(rows), boards, self.calls)


def write_outputs(result: OriginResult, out_dir: Path, config: TeamConfig) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast = out_dir / f"forecast_{result.origin.isoformat()}.csv"
    result.rows.to_csv(forecast, index=False)
    transcript = out_dir / f"transcript_{result.origin.isoformat()}.jsonl"
    with transcript.open("w") as handle:
        for b in result.boards:
            handle.write(json.dumps(board_to_json(b), default=str) + "\n")
    (out_dir / "team_config.json").write_text(json.dumps(config.to_json(), indent=2, default=str))
    return {"forecast": forecast, "transcript": transcript}
