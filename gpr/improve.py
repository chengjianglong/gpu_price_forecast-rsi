"""The Improver: proposes edits to the team config from the error breakdown.

It may only touch a whitelist of settings, each with bounds, so an edit can
change how the team argues and votes but can never reach the data, the scorer
or the truth caps. Every edit is applied to a copy; the loop keeps it only if
it passes the acceptance test on the validation origins.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pandas as pd

from .agents import Lens, extract_json
from .llm import LLM
from .team import TeamConfig

BOUNDS = {
    "temperature": (0.2, 1.0),
    "debate_rounds": (0, 2),
    "debate_min_disagreement": (0.0, 0.2),
    "rule.eta": (0.1, 10.0),
    "rule.decay": (0.0, 0.5),
    "rule.w_min": (0.0, 0.3),
    "rule.widen": (0.0, 2.0),
    "rule.stretch_1": (0.5, 3.0),
    "rule.stretch_3": (0.5, 3.0),
    "rule.stretch_7": (0.5, 3.0),
}
MAX_VIEW_CHARS = 600


def breakdown(cells: pd.DataFrame, catalog: pd.DataFrame) -> str:
    """Pooled skill by instrument class and horizon, plus range coverage, as text for the Improver."""
    if cells.empty:
        return "no scored cells"
    df = cells.copy()
    df["cls"] = df["series_id"].map(catalog["instrument_class"])
    lines = []
    for (cls, h), g in df.groupby(["cls", "horizon_days"]):
        skill = 1 - g["loss"].sum() / max(g["baseline_loss"].sum(), 1e-12)
        below = (g["y"] < g["q10"]).mean()
        above = (g["y"] > g["q90"]).mean()
        lines.append(f"{cls} h={h}: skill {skill:+.3f}, {len(g)} cells, below q10 {below:.0%}, above q90 {above:.0%}")
    return "\n".join(lines)


def improver_chat(cfg: TeamConfig, summary: dict, table: str, history: list[str]) -> list[dict[str, str]]:
    lenses = "\n".join(f'- {l.name}: "{l.view}"' for l in cfg.lenses)
    user = (
        "You improve a team of LLM agents that forecast GPU rental prices (q10/q50/q90, pinball loss, skill "
        "versus repeating the last value; the target coverage is 10% below q10 and 10% above q90).\n\n"
        f"Validation summary: {json.dumps(summary, default=str)}\n\nBy instrument class and horizon:\n{table}\n\n"
        f"Current settings: temperature {cfg.temperature}, debate_rounds {cfg.debate_rounds}, "
        f"debate_min_disagreement {cfg.debate_min_disagreement}, rule.eta {cfg.rule.eta}, rule.decay "
        f"{cfg.rule.decay}, rule.w_min {cfg.rule.w_min}, rule.widen {cfg.rule.widen}, range stretch around the "
        f"median (1 = as voted) rule.stretch_1 {cfg.rule.stretch_1}, rule.stretch_3 {cfg.rule.stretch_3}, "
        f"rule.stretch_7 {cfg.rule.stretch_7}.\nForecaster lenses:\n{lenses}\n\n"
        f"Edits tried before and their outcome:\n" + ("\n".join(history[-8:]) or "(none)") + "\n\n"
        "Propose ONE or TWO edits most likely to raise validation skill. Allowed paths: "
        f"{', '.join(BOUNDS)} (numbers within bounds {json.dumps(BOUNDS)}), and lenses.<name>.view (rewrite a "
        f"lens, at most {MAX_VIEW_CHARS} characters). Answer with JSON only: "
        '{"edits": [{"path": "rule.widen", "value": 0.8}], "rationale": "one sentence"}'
    )
    return [{"role": "user", "content": user}]


def apply_edits(cfg: TeamConfig, edits: list[dict[str, Any]]) -> tuple[TeamConfig, list[str]]:
    """Apply whitelisted, bounded edits to a copy; return it and a log of what changed."""
    new = copy.deepcopy(cfg)
    log = []
    for edit in edits or []:
        path, value = str(edit.get("path", "")), edit.get("value")
        if path.startswith("lenses.") and path.endswith(".view"):
            name = path.split(".")[1]
            if isinstance(value, str) and value.strip() and any(l.name == name for l in new.lenses):
                text = value.strip()[:MAX_VIEW_CHARS]
                new.lenses = tuple(Lens(l.name, text) if l.name == name else l for l in new.lenses)
                log.append(f"{path} rewritten")
            continue
        if path not in BOUNDS:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = BOUNDS[path]
        number = min(hi, max(lo, number))
        if path.startswith("rule."):
            new.rule = replace(new.rule, **{path.split(".", 1)[1]: number})
        elif path == "debate_rounds":
            number = int(round(number))
            new.debate_rounds = number
        else:
            setattr(new, path, number)
        log.append(f"{path}={number:g}")
    return new, log


def propose(llm: LLM, cfg: TeamConfig, summary: dict, table: str, history: list[str]) -> tuple[TeamConfig, list[str], str]:
    text = llm.generate([improver_chat(cfg, summary, table, history)], n=1, temperature=0.7, max_tokens=700)[0][0].text
    data = extract_json(text) or {}
    new, log = apply_edits(cfg, data.get("edits") if isinstance(data, dict) else [])
    return new, log, str(data.get("rationale", "") if isinstance(data, dict) else "")[:300]
