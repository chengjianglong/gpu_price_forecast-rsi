"""The team: runs the agents and the debate for one origin over many series.

Flow per origin (docs/PLAN.md section 1):

1. Pacific's baselines for the origin, then one shared board per series.
2. Market Analyst: one note from the news events, posted on every board.
3. Independent round: each Forecaster posts a position without seeing the others.
4. Up to ``debate_rounds`` rounds of Critic challenges and Forecaster revisions,
   only on series where the Forecasters disagree enough (``debate_min_disagreement``).
5. Aggregator: weighted vote per quantile, widened by the remaining disagreement,
   then stretched per horizon (``rule.stretch_<h>``).

Off by default, on in ``configs/team_list.yaml``: ``engine`` in ``baseline_methods`` posts the
list-replay engine's forecast on the board, and ``list_evidence`` shows the offer list.

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

from . import engine
from . import pack as packlib
from .agents import (DEFAULT_LENSES, Lens, critic_chat, forecaster_chat, market_chat, parse_challenges,
                     parse_forecast, revision_chat)
from .board import Board, baseline_table, board_to_json, build_board
from .llm import LLM
from .offers import OfferTable, shared as shared_offers
from .pack import HORIZONS, Pack
from .weights import VotingWeights, WeightRule, aggregate, disagreement, stretch

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
    use_findings: bool = True
    list_evidence: bool = False       # show the offer list behind the median on the board
    seed: int | None = 20261106       # fixed sampling seeds, so a rerun gives the same forecasts
    rule: WeightRule = field(default_factory=WeightRule)

    @classmethod
    def load(cls, path: str | Path) -> "TeamConfig":
        return cls.from_json(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_json(cls, raw: dict) -> "TeamConfig":
        """The inverse of ``to_json``."""
        raw = dict(raw)
        lenses = tuple(Lens(**l) for l in raw.pop("lenses", [])) or DEFAULT_LENSES
        rule = WeightRule(**raw.pop("rule", {}))
        methods = tuple(raw.pop("baseline_methods", cls.baseline_methods))
        return cls(lenses=lenses, rule=rule, baseline_methods=methods, **raw)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["lenses"] = [asdict(l) for l in self.lenses]
        return out


def baseline_rows(pack: Pack, method: str, origin: date, cache_dir: Path) -> pd.DataFrame:
    """One of Pacific's baselines for every series at one origin, kept in ``cache_dir``."""
    out = cache_dir / f"{method}_{origin.isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        return pd.read_csv(out)
    return packlib.baselines(pack.files.series, pack.files.catalog, method, out, origin=origin, tier="all")


@dataclass
class OriginResult:
    origin: date
    rows: pd.DataFrame
    boards: list[Board]
    calls: int


class Team:
    def __init__(self, config: TeamConfig, llm: LLM, pack: Pack, events: Iterable[Any] = (),
                 weights: VotingWeights | None = None, cache_dir: Path | None = None,
                 findings: Sequence[Any] = ()):
        self.cfg = config
        self.llm = llm
        self.pack = pack
        self.events = list(events)
        self.agents = [l.name for l in config.lenses]
        self.weights = weights or VotingWeights(self.agents, config.rule)
        self.cache_dir = cache_dir or pack.files.directory / "baselines"
        self.findings = list(findings)   # research.Result objects that survived
        self.calls = 0
        self._phase = 0

    # ---- baselines ----------------------------------------------------------
    def baselines_for(self, origin: date) -> dict:
        """Pacific's baselines, plus the engine where the list replay itself produced the forecast."""
        shown = tuple(self.cfg.baseline_methods)
        methods = [m for m in dict.fromkeys((FALLBACK_METHOD,) + shown) if m != engine.METHOD]
        if engine.METHOD in shown and "drift" not in methods:
            methods.append("drift")      # the engine's fallback
        by_method = {m: baseline_rows(self.pack, m, origin, self.cache_dir) for m in methods}
        frames = list(by_method.values())
        if engine.METHOD in shown:
            drift = by_method["drift"]
            rows = engine.forecast_rows(shared_offers(self.pack).table, origin, sorted(set(drift["series_id"])), drift)
            frames.append(rows[rows["method_note"] == engine.METHOD])
        return baseline_table(pd.concat(frames, ignore_index=True))

    def offer_table(self, series_id: str, origin: date) -> OfferTable | None:
        """The offer list behind a series, cut at the origin; None unless ``list_evidence`` is on."""
        return shared_offers(self.pack).table(series_id, origin) if self.cfg.list_evidence else None

    # ---- one origin -----------------------------------------------------------
    def _generate(self, chats: list) -> list[str]:
        if not chats:
            return []
        seed = None if self.cfg.seed is None else self.cfg.seed + 100_000 * self._phase
        self._phase += 1
        self.calls += len(chats)
        samples = self.llm.generate(chats, n=1, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
                                    seed=seed)
        return [s[0].text for s in samples]

    def run(self, origin: date, series_ids: Sequence[str]) -> OriginResult:
        self._phase = origin.toordinal() % 1000 * 10   # phase counter: distinct, repeatable seeds per call batch
        table = self.baselines_for(origin)
        boards: list[Board] = []
        for sid in series_ids:
            by_method = table.get((sid, origin))
            if not by_method or FALLBACK_METHOD not in by_method:
                continue
            shown = {m: by_method[m] for m in self.cfg.baseline_methods if m in by_method}
            found = ()
            if self.cfg.use_findings and self.findings:
                from .research import findings_for

                found = findings_for(self.findings, str(self.pack.info(sid).get("instrument_class")))
            board = build_board(self.pack, sid, origin, shown, self.events, self.cfg.max_events, findings=found,
                                offers=self.offer_table(sid, origin))
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
                q10, q50, q90 = stretch((q10, q50, q90), self.cfg.rule.stretch_for(h))
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
