"""Voting weights and the Aggregator (docs/PLAN.md section 2.5).

Weights are kept per context (instrument class x horizon) and per agent. The
fast update is Hedge: ``w <- w * exp(-eta * loss)``, then decay towards equal
weights and a floor, then normalise. The rule's settings (``eta``, ``decay``,
``w_min``) are what the outer loop varies in block 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Mapping, Sequence


@dataclass
class WeightRule:
    eta: float = 2.0          # how fast weights react to loss
    decay: float = 0.05       # pull towards equal weights per update
    w_min: float = 0.05       # floor, so no perspective is dropped
    widen: float = 0.5        # how much forecaster disagreement widens the final range


@dataclass
class VotingWeights:
    agents: Sequence[str]
    rule: WeightRule = field(default_factory=WeightRule)
    table: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)

    def get(self, context: tuple[str, int]) -> dict[str, float]:
        return dict(self.table.get(context) or {a: 1 / len(self.agents) for a in self.agents})

    def update(self, context: tuple[str, int], losses: Mapping[str, float]) -> dict[str, float]:
        """Hedge update from per-agent losses (already scaled, e.g. loss / baseline loss)."""
        w = self.get(context)
        n = len(self.agents)
        for agent in self.agents:
            if agent in losses and math.isfinite(losses[agent]):
                w[agent] *= math.exp(-self.rule.eta * losses[agent])
        total = sum(w.values()) or 1.0
        w = {a: (1 - self.rule.decay) * v / total + self.rule.decay / n for a, v in w.items()}
        # Mix with the floor so every weight is >= w_min and the order is kept.
        floor = min(self.rule.w_min, 1.0 / n)
        self.table[context] = {a: floor + (1 - n * floor) * v for a, v in w.items()}
        return self.table[context]

    def to_json(self) -> dict:
        return {f"{c}|{h}": w for (c, h), w in self.table.items()}


def aggregate(positions: Mapping[str, tuple[float, float, float]], weights: Mapping[str, float],
              widen: float = 0.5) -> tuple[float, float, float]:
    """Weighted vote per quantile, then widen the range by how much the medians disagree.

    The widening adds ``widen x`` the weighted spread of the agents' medians
    (relative to the voted median) to each side of the range.
    """
    agents = [a for a in positions if weights.get(a, 0) > 0]
    if not agents:
        raise ValueError("no positions to aggregate")
    total = sum(weights[a] for a in agents)
    vote = [sum(weights[a] * positions[a][i] for a in agents) / total for i in range(3)]
    q10, q50, q90 = sorted(vote)
    medians = [positions[a][1] for a in agents]
    if len(medians) > 1 and q50 > 0:
        spread = sum(weights[a] * abs(positions[a][1] - q50) for a in agents) / total
        q10 = max(q10 - widen * spread, 1e-4)
        q90 = q90 + widen * spread
    return (q10, q50, q90)


def disagreement(positions: Mapping[str, tuple[float, float, float]]) -> float:
    """Spread of the medians relative to their median (0 = full agreement)."""
    meds = [p[1] for p in positions.values()]
    if len(meds) < 2:
        return 0.0
    m = median(meds)
    return (max(meds) - min(meds)) / m if m > 0 else 0.0
