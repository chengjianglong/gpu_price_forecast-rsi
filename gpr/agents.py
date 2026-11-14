"""Agent prompts and answer parsing.

Every LLM agent answers in JSON. Parsing is strict about what the scorer
would reject and lenient about formatting: crossed quantiles are sorted and
non-positive values floored, and each repair is recorded in
``Position.problems`` so the RL rewards (block 3) can penalise it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .board import Board, Challenge, Position
from .llm import Messages
from .pack import HORIZONS

FORECAST_FORMAT = (
    'Answer with JSON only, no other text:\n'
    '{"forecasts": {"1": [q10, q50, q90], "3": [q10, q50, q90], "7": [q10, q50, q90], "30": [q10, q50, q90]}, '
    '"evidence": ["E2", "E6"], "reasoning": "two or three sentences"}\n'
    "Keys are horizons in days after the origin. Values are USD per GPU-hour, with q10 <= q50 <= q90, all > 0. "
    "q10 and q90 are the 10th and 90th percentiles: the true value should fall below q10 one time in ten and "
    "above q90 one time in ten. Cite the evidence IDs your forecast relies on."
)

SYSTEM = (
    "You are one of several agents forecasting posted GPU rental prices. Agents share the same evidence and "
    "debate it from different perspectives. You are scored by pinball loss at quantiles 0.1, 0.5 and 0.9, "
    "against a baseline that repeats the last value. Posted prices often stay flat for weeks, then move in "
    "steps. Use only the evidence given: nothing dated after the forecast origin exists for you."
)


@dataclass(frozen=True)
class Lens:
    name: str
    view: str


DEFAULT_LENSES = (
    Lens("momentum", "You are the momentum forecaster: you believe recent trends in this price tend to continue, "
         "and you look for the direction and pace of the latest moves."),
    Lens("reversion", "You are the mean-reversion forecaster: you believe prices drift back towards their usual "
         "level after a move, and you look for moves that overshot."),
    Lens("supply_demand", "You are the supply-and-demand forecaster: you reason from GPU generations, provider "
         "counts, capacity and news events about why this price should move or hold."),
)


def forecaster_chat(board: Board, lens: Lens) -> Messages:
    user = (f"{lens.view}\n\n{board.render_evidence()}\n\nGive your own forecast without seeing the other agents. "
            f"{FORECAST_FORMAT}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def critic_chat(board: Board, agents: Sequence[str]) -> Messages:
    user = (
        "You are the Critic. Read the evidence and the Forecasters' positions, then challenge the weakest "
        "parts: ranges too narrow or too wide for the history of changes, medians that ignore the evidence, "
        "reasoning that doesn't hold, citations that don't support the claim, or any use of information "
        "dated after the origin. Only challenge what the evidence lets you argue.\n\n"
        f"{board.render_evidence()}\n\nPositions:\n{board.render_positions(board.positions)}\n\n"
        'Answer with JSON only: {"challenges": [{"target": "<agent name>", "claim": "one sentence", '
        '"evidence": ["E4"]}]}\n'
        f"Agent names: {', '.join(agents)}. At most two challenges per agent; an empty list is allowed."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def revision_chat(board: Board, lens: Lens) -> Messages:
    own = board.positions.get(lens.name)
    user = (
        f"{lens.view}\n\n{board.render_evidence()}\n\n"
        f"Your independent forecast:\n{board.render_positions({lens.name: own} if own else {})}\n\n"
        f"The other Forecasters:\n{board.render_positions(board.positions, exclude=lens.name)}\n\n"
        f"Challenges to you:\n{board.render_challenges(lens.name)}\n\n"
        "Revise your forecast or keep it. Change it only where an argument or the evidence persuades you, "
        "and say in the reasoning what changed your mind or why you hold. You are scored on your own revised "
        f"forecast, so copying the others gains you nothing. {FORECAST_FORMAT}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def market_chat(events_text: str, origin: str) -> Messages:
    user = (
        "You are the Market Analyst. From the news events below, all published on or before "
        f"{origin}, write at most four sentences on what they imply for GPU rental prices over the next days "
        "and weeks: which GPUs, which direction, how confident. Cite event IDs in brackets. If the events say "
        "nothing about prices, say so.\n\n" + events_text
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---- parsing ---------------------------------------------------------------
def extract_json(text: str) -> Any | None:
    """The first JSON object in ``text`` (handles code fences and leading prose)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        else:
            # Ran out of text with braces still open: models sometimes stop just
            # before the closing brace. Close them and try once.
            if depth > 0 and not in_string:
                try:
                    return json.loads(text[start:].rstrip().rstrip(",") + "}" * depth)
                except json.JSONDecodeError:
                    pass
        start = text.find("{", start + 1)
    return None


def parse_forecast(text: str, agent: str, valid_ids: set[str], fallback: dict[int, tuple[float, float, float]],
                   horizons: Sequence[int] = HORIZONS, floor: float = 1e-4) -> Position:
    """Parse a forecaster's JSON; any missing horizon falls back to ``fallback`` and is recorded."""
    data = extract_json(text)
    problems: list[str] = []
    if not isinstance(data, dict):
        return Position(agent, dict(fallback), problems=["unparseable"], fallback=True, raw=text)
    raw_q = data.get("forecasts") if isinstance(data.get("forecasts"), dict) else {}
    quantiles: dict[int, tuple[float, float, float]] = {}
    for h in horizons:
        q = raw_q.get(str(h), raw_q.get(h))
        try:
            vals = [float(x) for x in q][:3]
            if len(vals) != 3 or any(v != v for v in vals):
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"missing_h{h}")
            quantiles[h] = fallback[h]
            continue
        if any(v <= 0 for v in vals):
            problems.append(f"nonpositive_h{h}")
            vals = [max(v, floor) for v in vals]
        if not (vals[0] <= vals[1] <= vals[2]):
            problems.append(f"crossed_h{h}")
            vals = sorted(vals)
        quantiles[h] = (vals[0], vals[1], vals[2])
    cites = [str(c) for c in data.get("evidence") or [] if isinstance(c, (str, int))]
    bad = [c for c in cites if c not in valid_ids]
    if bad:
        problems.append("bad_citation:" + ",".join(bad[:5]))
    if not cites:
        problems.append("no_citation")
    return Position(agent, quantiles, evidence=[c for c in cites if c in valid_ids],
                    reasoning=str(data.get("reasoning") or "")[:1200], problems=problems,
                    fallback=all(p.startswith("missing_") for p in problems) and len(problems) == len(horizons),
                    raw=text)


def parse_challenges(text: str, author: str, agents: Sequence[str], valid_ids: set[str],
                     per_agent: int = 2) -> tuple[list[Challenge], list[str]]:
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("challenges"), list):
        return [], ["unparseable"]
    out, problems, count = [], [], {a: 0 for a in agents}
    for item in data["challenges"]:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or "")
        if target not in count:
            problems.append(f"unknown_target:{target[:30]}")
            continue
        if count[target] >= per_agent:
            continue
        cites = [str(c) for c in item.get("evidence") or []]
        if any(c not in valid_ids for c in cites):
            problems.append("bad_citation")
        claim = str(item.get("claim") or "").strip()
        if claim:
            out.append(Challenge(author, target, claim[:400], [c for c in cites if c in valid_ids]))
            count[target] += 1
    return out, problems
