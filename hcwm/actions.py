"""PlayWorld's shared action space and the parsing around it.

The agent picks one token per step from ``AGENT_ACTIONS`` (or ``END``). Each
backend maps a token to its own native control; see ``hcwm/backends``.
"""

from __future__ import annotations

import re
from typing import Any

MOVE = ("W", "A", "S", "D")
LOOK = ("LEFT", "RIGHT", "UP", "DOWN")
AGENT_ACTIONS = MOVE + LOOK + ("WAIT",)
END = "END"

_TOKEN_NORMALIZE = {"←": "LEFT", "→": "RIGHT", "↑": "UP", "↓": "DOWN"}
_HOLD_RE = re.compile(r"hold\(\s*([A-Za-z+]+)\s*,\s*(\d+)\s*ms\s*\)", re.IGNORECASE)
# PlayWorld writes one dataset token as roughly 450 ms of key hold
# ("d*3" <-> "hold(D,1350ms)").
DATASET_TOKEN_MS = 450


def _normalize_action_str(text: str) -> str:
    text = re.sub(r"interact\([^)]*\)", "W", text or "")
    text = re.sub(r"\bu(?=\*)", "UP", text)
    text = re.sub(r"\bj(?=\*)", "DOWN", text)
    for raw, std in _TOKEN_NORMALIZE.items():
        text = text.replace(raw, std)
    return text


def dataset_segments(task: dict[str, Any]) -> list[tuple[str, int]]:
    """Return the human-written reference plan as ``[(action, tokens), ...]``.

    Reads the compact ``action`` field ("d*3 -> w*12") and falls back to
    ``action_sequence_steps`` ("hold(D,1350ms)").
    """
    segments: list[tuple[str, int]] = []
    for part in re.split(r"->|→", _normalize_action_str(str(task.get("action") or ""))):
        match = re.match(r"\s*([A-Za-z]+)\s*\*\s*(\d+)", part)
        if match:
            segments.append((match.group(1).upper(), max(1, int(match.group(2)))))
    if not segments:
        for step in task.get("action_sequence_steps") or []:
            match = _HOLD_RE.fullmatch(str(step).strip())
            if match:
                key = match.group(1).upper().split("+")[0]
                tokens = max(1, round(int(match.group(2)) / DATASET_TOKEN_MS))
                segments.append((key, tokens))
    return [(a if a in AGENT_ACTIONS else "W", n) for a, n in segments]


def reference_plan_length(task: dict[str, Any], tokens_per_step: int) -> int:
    """Number of agent steps the reference plan covers before padding."""
    unit = max(1, tokens_per_step)
    return max(1, sum(-(-count // unit) for _, count in dataset_segments(task)))


def reference_plan(task: dict[str, Any], num_steps: int, tokens_per_step: int) -> list[str]:
    """Spread the reference plan over ``num_steps`` agent steps.

    Every ``tokens_per_step`` dataset tokens become one step. The agent sees the
    recommendation for each step but decides for itself.
    """
    plan: list[str] = []
    unit = max(1, tokens_per_step)
    for action, count in dataset_segments(task):
        plan.extend([action] * max(1, -(-count // unit)))
    plan = plan or ["W"]
    if len(plan) < num_steps:
        plan += [plan[-1]] * (num_steps - len(plan))
    return plan[:num_steps]


def is_insight_wait_only(task: dict[str, Any]) -> bool:
    """Insight-evolution cases hold the camera still and only observe."""
    if str(task.get("_split", "")).lower() == "insight":
        return True
    action = str(task.get("action") or "").strip().lower().replace(" ", "")
    steps = [str(s).strip().lower().replace(" ", "") for s in task.get("action_sequence_steps") or []]
    return bool(
        re.fullmatch(r"wait\*\d+", action)
        or action in {"wait", "waitandobserve"}
        or (steps and all(s.startswith("wait(") for s in steps))
    )


def parse_agent_reply(text: str, allowed: tuple[str, ...]) -> str | None:
    """Pick the first allowed token in the reply, longest names first."""
    upper = str(text or "").upper()
    for token in sorted(allowed, key=len, reverse=True):
        if re.search(rf"\b{token}\b", upper):
            return token
    return None
