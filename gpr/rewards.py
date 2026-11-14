"""Rewards and the time rules for truth (docs/PLAN.md sections 2.1 and 2.6).

**Truth caps.** Rewards may only use truth dated on or before a cap:

- RL training: the day before the first validation origin (``TRAIN_TRUTH_CAP``),
  so the weights never learn from the validation or test windows.
- Validation scoring: the first locked test origin (``VALIDATION_TRUTH_CAP``),
  so validation never sees what the test scores.

Only the horizons Sunday scores (1, 3 and 7 days) earn reward.

**Cell skill.** For one (series, origin, horizon):
``skill = 1 - pinball(forecast) / max(pinball(last_value), floor)``, clipped to
``[-3, 1]``. The floor (a small share of the last value) stops near-flat
series, where the last value is almost perfect, from producing huge negative
rewards; a tight forecast on a flat series still earns close to 1.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Mapping, Sequence

from .board import Position
from .pack import QUANTILES, SCORED_HORIZONS, Splits

SPLITS = Splits()
TRAIN_TRUTH_CAP = min(SPLITS.validation) - timedelta(days=1)
VALIDATION_TRUTH_CAP = min(SPLITS.test)

FLOOR_SHARE = 0.002          # baseline-loss floor as a share of the last value
SKILL_CLIP = (-3.0, 1.0)
PENALTY = {"unparseable": 1.0, "missing": 0.25, "crossed": 0.1, "nonpositive": 0.25,
           "bad_citation": 0.05, "no_citation": 0.05}


def pinball(y: float, q: Sequence[float], quantiles: Sequence[float] = QUANTILES) -> float:
    """Mean pinball loss over the quantiles, as Pacific's scorer computes it."""
    return sum(max(t * (y - f), (t - 1) * (y - f)) for t, f in zip(quantiles, q)) / len(quantiles)


def cell_skill(q: Sequence[float], y: float, last: float) -> float:
    base = pinball(y, (last, last, last))
    denom = max(base, FLOOR_SHARE * abs(last), 1e-9)
    lo, hi = SKILL_CLIP
    return max(lo, min(hi, 1.0 - pinball(y, q) / denom))


def truths_for(pack, series_id: str, origin: date, cap: date,
               horizons: Sequence[int] = SCORED_HORIZONS) -> dict[int, float]:
    """Truth per horizon, only for targets dated on or before ``cap``."""
    out = {}
    for h in horizons:
        target = origin + timedelta(days=h)
        if target <= cap:
            y = pack.truth(series_id, target)
            if y is not None:
                out[h] = y
    return out


def problem_penalty(problems: Sequence[str]) -> float:
    total = 0.0
    for p in problems:
        key = p.split(":")[0]
        key = "missing" if key.startswith("missing_") else key
        key = "crossed" if key.startswith("crossed_") else key
        key = "nonpositive" if key.startswith("nonpositive_") else key
        total += PENALTY.get(key, 0.0)
    return total


def forecast_reward(position: Position, truths: Mapping[int, float], last: float) -> float:
    """Mean cell skill over horizons with truth, minus penalties for repaired output."""
    if "unparseable" in position.problems:
        return SKILL_CLIP[0] - PENALTY["unparseable"]
    skills = [cell_skill(position.quantiles[h], y, last) for h, y in truths.items() if h in position.quantiles]
    base = sum(skills) / len(skills) if skills else 0.0
    return base - problem_penalty(position.problems)
