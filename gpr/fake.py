"""A rule-based stand-in for the LLM agents, for offline tests and dry runs.

It reads the board from the prompt and answers in the agents' JSON format:
Forecasters echo the ``last_value_empirical`` baseline, nudged by their lens
(momentum up, reversion down) so the debate has something to argue about; the
Critic challenges the momentum forecaster; the Market Analyst says nothing
useful. Its forecasts say nothing about real skill.
"""

from __future__ import annotations

import json
import re

from .llm import FakeLLM, Messages

_BASELINE = re.compile(r"\[(E\d+)\] Baseline last_value_empirical \(q10, q50, q90\): (.*)")
_PAIR = re.compile(r"h=(\d+): \[([\d.eE+-]+), ([\d.eE+-]+), ([\d.eE+-]+)\]")
_NUDGE = {"momentum": 1.01, "reversion": 0.99, "supply_demand": 1.0}


def respond(chat: Messages, i: int) -> str:
    text = chat[-1]["content"]
    if "You are the Critic" in text:
        return json.dumps({"challenges": [{"target": "momentum", "claim": "The range ignores the flat history.",
                                           "evidence": ["E2"]}]})
    if "You are the Market Analyst" in text:
        return "The events say nothing specific about prices."
    match = _BASELINE.search(text)
    if not match:
        return "I cannot tell."
    eid, rest = match.groups()
    lens = None
    if "momentum forecaster" in text:
        lens = "momentum"
    elif "mean-reversion forecaster" in text:
        lens = "reversion"
    elif "supply-and-demand forecaster" in text:
        lens = "supply_demand"
    k = _NUDGE.get(lens or "", 1.0)
    forecasts = {h: [float(a), float(b) * k, max(float(c), float(b) * k)] for h, a, b, c in _PAIR.findall(rest)}
    for h, (a, b, c) in forecasts.items():
        forecasts[h] = [min(a, b), b, max(b, c)]
    return json.dumps({"forecasts": forecasts, "evidence": [eid, "E2"],
                       "reasoning": f"Echo of the baseline with a {lens} nudge."})


def fake_llm() -> FakeLLM:
    return FakeLLM(respond)
