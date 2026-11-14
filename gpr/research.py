"""The research loop's lab notebook: hypotheses about what moves GPU prices,
tested on the training window only.

The brief asks for a system that "proposes a hypothesis about what moves GPU
prices, designs a test, runs it, keeps what survives, records what it learned
and repeats". Here:

- **Propose:** a research agent (LLM) picks the next hypothesis from a library
  of testable families, choosing its parameters (instrument class, horizon,
  look-back) and stating what it expects and why. It sees every earlier
  result, so it can follow up on a survivor or drop a dead end.
- **Test:** each family is a statistical test written in code, run on data
  dated on or before ``TRAIN_TRUTH_CAP`` only, with an effect size and a
  permutation p-value. The LLM never computes a statistic.
- **Keep:** a hypothesis survives if ``p < alpha`` and the effect has the
  predicted sign. Survivors become findings (``F1``, ``F2``...) on every
  shared board of the matching instrument class, which agents can cite.
- **Record:** every hypothesis, test and verdict, including the failures, goes
  to ``notebook.jsonl``. That file is what "the system learned across
  iterations".

Whether a survivor actually improves forecasts is a separate question,
answered by the outer loop's validation and by the ``use_findings`` ablation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .agents import extract_json
from .llm import LLM
from .rewards import TRAIN_TRUTH_CAP

CLASSES = ("neocloud_rate_card", "neocloud_spot", "marketplace_ask", "hyperscaler_spot", "hyperscaler_list",
           "any")


@dataclass
class Hypothesis:
    family: str
    params: dict[str, Any]
    statement: str
    expected_sign: int            # +1 or -1: the sign of the effect the hypothesis predicts
    rationale: str = ""


@dataclass
class Result:
    hid: str
    hypothesis: Hypothesis
    effect: float
    p_value: float
    n: int
    survived: bool
    summary: str
    iteration: int = 0
    tested_until: str = ""

    def finding_text(self) -> str:
        return (f"{self.hypothesis.statement} Tested on data up to {self.tested_until}: "
                f"{self.summary} (n={self.n}, p={self.p_value:.3f}).")

    def to_json(self) -> dict:
        row = asdict(self)
        return row


# ---- data helpers -----------------------------------------------------------------
def _series_frame(pack, cap: date, instrument_class: str) -> dict[str, pd.DataFrame]:
    df = pack.series[pack.series["date"] <= cap]
    if instrument_class != "any":
        df = df[df["instrument_class"] == instrument_class]
    else:  # every posted rental series; AWS spot is a different instrument and is left out
        df = df[~df["series_id"].str.startswith("aws_spot")]
    out = {}
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("date")
        if len(g) >= 30:
            out[sid] = g.set_index("date")
    return out


def _log_change(values: np.ndarray, lag: int) -> np.ndarray:
    a, b = values[:-lag], values[lag:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(b / a)
    return out


def _perm_p(x: np.ndarray, y: np.ndarray, stat: Callable[[np.ndarray, np.ndarray], float], sign: int,
            n: int = 500, seed: int = 0) -> tuple[float, float]:
    """One-sided permutation p-value for ``sign * stat(x, y)`` being this large."""
    rng = np.random.default_rng(seed)
    observed = stat(x, y)
    if not math.isfinite(observed):
        return observed, 1.0
    count = 0
    for _ in range(n):
        if sign * stat(x, rng.permutation(y)) >= sign * observed:
            count += 1
    return observed, (count + 1) / (n + 1)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---- the test families -------------------------------------------------------------
def test_stickiness(pack, cap, p) -> tuple[float, float, int, str]:
    """Share of h-day windows with no change at all, against a threshold (default 50%)."""
    h, threshold = int(p.get("horizon", 7)), float(p.get("threshold", 0.5))
    shares = []
    for g in _series_frame(pack, cap, p.get("instrument_class", "any")).values():
        v = g["value"].to_numpy(dtype=float)
        if len(v) > h:
            shares.append(float(np.mean(v[h:] == v[:-h])))
    if len(shares) < 5:
        return float("nan"), 1.0, len(shares), "too few series"
    arr = np.array(shares)
    effect = float(np.mean(arr) - threshold)
    # bootstrap over series for a one-sided p-value that the mean share exceeds the threshold
    rng = np.random.default_rng(0)
    boots = np.array([np.mean(rng.choice(arr, len(arr))) for _ in range(1000)])
    p_value = float(np.mean(boots <= threshold)) if effect > 0 else float(np.mean(boots >= threshold))
    return effect, max(p_value, 1 / 1001), len(arr), f"mean share of unchanged {h}-day windows {np.mean(arr):.0%}"


def _pairs_by_series(pack, cap, p, build: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]]):
    xs, ys = [], []
    for g in _series_frame(pack, cap, p.get("instrument_class", "any")).values():
        x, y = build(g)
        ok = np.isfinite(x) & np.isfinite(y)
        xs.append(x[ok])
        ys.append(y[ok])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def test_momentum(pack, cap, p) -> tuple[float, float, int, str]:
    """Correlation of the past ``lookback``-day log change with the next ``horizon``-day log change."""
    h, lb = int(p.get("horizon", 7)), int(p.get("lookback", 7))

    def build(g):
        v = g["value"].to_numpy(dtype=float)
        past = np.full(len(v), np.nan)
        fut = np.full(len(v), np.nan)
        past[lb:] = _log_change(v, lb)
        fut[:-h] = _log_change(v, h)
        moved = (past != 0) & np.isfinite(past)        # only after an actual move
        return past[moved][::h], fut[moved][::h]        # thin to roughly independent windows

    x, y = _pairs_by_series(pack, cap, p, build)
    effect, p_value = _perm_p(x, y, _corr, +1)
    return effect, p_value, len(x), f"correlation of past {lb}-day move with next {h}-day move {effect:+.2f}"


def test_reversion(pack, cap, p) -> tuple[float, float, int, str]:
    """Correlation of the deviation from the ``window``-day median with the next ``horizon``-day change."""
    h, w = int(p.get("horizon", 7)), int(p.get("window", 30))

    def build(g):
        v = g["value"].to_numpy(dtype=float)
        med = pd.Series(v).rolling(w, min_periods=w).median().to_numpy()
        dev = np.log(v / med)
        fut = np.full(len(v), np.nan)
        fut[:-h] = _log_change(v, h)
        keep = np.isfinite(dev) & (dev != 0)
        return dev[keep][::h], fut[keep][::h]

    x, y = _pairs_by_series(pack, cap, p, build)
    effect, p_value = _perm_p(x, y, _corr, -1)
    return effect, p_value, len(x), f"correlation of deviation from {w}-day median with next {h}-day move {effect:+.2f}"


def test_providers(pack, cap, p) -> tuple[float, float, int, str]:
    """Correlation of the change in provider count over ``lookback`` days with the next ``horizon``-day price change."""
    h, lb = int(p.get("horizon", 7)), int(p.get("lookback", 7))

    def build(g):
        n = g["n_providers"].to_numpy(dtype=float)
        v = g["value"].to_numpy(dtype=float)
        dn = np.full(len(v), np.nan)
        fut = np.full(len(v), np.nan)
        dn[lb:] = n[lb:] - n[:-lb]
        fut[:-h] = _log_change(v, h)
        keep = np.isfinite(dn) & (dn != 0)
        return dn[keep][::h], fut[keep][::h]

    x, y = _pairs_by_series(pack, cap, p, build)
    sign = int(p.get("sign", -1))
    effect, p_value = _perm_p(x, y, _corr, sign)
    return effect, p_value, len(x), f"correlation of {lb}-day change in provider count with next {h}-day move {effect:+.2f}"


def test_volatility_clustering(pack, cap, p) -> tuple[float, float, int, str]:
    """Correlation of the absolute past ``lookback``-day move with the absolute next ``horizon``-day move."""
    h, lb = int(p.get("horizon", 7)), int(p.get("lookback", 14))

    def build(g):
        v = g["value"].to_numpy(dtype=float)
        past = np.full(len(v), np.nan)
        fut = np.full(len(v), np.nan)
        past[lb:] = np.abs(_log_change(v, lb))
        fut[:-h] = np.abs(_log_change(v, h))
        return past[::h], fut[::h]

    x, y = _pairs_by_series(pack, cap, p, build)
    effect, p_value = _perm_p(x, y, _corr, +1)
    return effect, p_value, len(x), f"correlation of |past {lb}-day move| with |next {h}-day move| {effect:+.2f}"


def test_cross_source(pack, cap, p) -> tuple[float, float, int, str]:
    """Same GPU and tier, different collectors: does one collector's past move predict another's next move?"""
    h, lb = int(p.get("horizon", 7)), int(p.get("lookback", 7))
    df = pack.series[(pack.series["date"] <= cap) & ~pack.series["series_id"].str.startswith("aws_spot")]
    key = df["series_id"].str.split(".").str[2:4].str.join(".")   # gpu_slug.pricing_tier
    df = df.assign(key=key, source=df["series_id"].str.split(".").str[0])
    xs, ys = [], []
    for _, g in df.groupby("key"):
        wide = g.pivot_table(index="date", columns="source", values="value", aggfunc="first").sort_index()
        if wide.shape[1] < 2:
            continue
        cols = list(wide.columns)
        for a in cols:
            for b in cols:
                if a == b:
                    continue
                va, vb = wide[a].to_numpy(dtype=float), wide[b].to_numpy(dtype=float)
                past = np.full(len(va), np.nan)
                fut = np.full(len(vb), np.nan)
                past[lb:] = _log_change(va, lb)
                fut[:-h] = _log_change(vb, h)
                keep = np.isfinite(past) & np.isfinite(fut) & (past != 0)
                xs.append(past[keep][::h])
                ys.append(fut[keep][::h])
    if not xs:
        return float("nan"), 1.0, 0, "no overlapping collectors"
    x, y = np.concatenate(xs), np.concatenate(ys)
    effect, p_value = _perm_p(x, y, _corr, +1)
    return effect, p_value, len(x), f"correlation of another collector's past {lb}-day move with this one's next {h}-day move {effect:+.2f}"


FAMILIES: dict[str, tuple[Callable, str]] = {
    "stickiness": (test_stickiness, "params: instrument_class, horizon (1/3/7), threshold (0-1). Predicts +1 if "
                   "most h-day windows show no change."),
    "momentum": (test_momentum, "params: instrument_class, horizon, lookback. +1 = moves continue."),
    "reversion": (test_reversion, "params: instrument_class, horizon, window (days). -1 = deviations revert."),
    "providers": (test_providers, "params: instrument_class, horizon, lookback, sign (+1/-1). Does a change in the "
                  "number of providers predict the next price move?"),
    "volatility_clustering": (test_volatility_clustering, "params: instrument_class, horizon, lookback. +1 = big "
                              "moves follow big moves (wider ranges after moves)."),
    "cross_source": (test_cross_source, "params: horizon, lookback. +1 = collectors of the same GPU and tier "
                     "follow each other with a lag."),
}


def run_test(pack, hypothesis: Hypothesis, cap: date = TRAIN_TRUTH_CAP, alpha: float = 0.05,
             iteration: int = 0, hid: str = "H?") -> Result:
    fn, _ = FAMILIES[hypothesis.family]
    effect, p_value, n, summary = fn(pack, cap, hypothesis.params)
    sign_ok = math.isfinite(effect) and (effect * hypothesis.expected_sign > 0)
    survived = bool(sign_ok and p_value < alpha and n >= 20)
    return Result(hid, hypothesis, float(effect), float(p_value), int(n), survived, summary, iteration, cap.isoformat())


# ---- the research agent ------------------------------------------------------------------
SEED_AGENDA = (
    Hypothesis("stickiness", {"instrument_class": "neocloud_rate_card", "horizon": 7, "threshold": 0.5},
               "Neocloud rate cards usually do not move at all within 7 days.", +1,
               "Rate cards are list prices that change in steps."),
    Hypothesis("volatility_clustering", {"instrument_class": "any", "horizon": 7, "lookback": 14},
               "Large recent moves are followed by large moves.", +1, "Repricing comes in bursts."),
    Hypothesis("reversion", {"instrument_class": "neocloud_spot", "horizon": 7, "window": 30},
               "Spot prices that deviate from their 30-day median move back towards it.", -1,
               "Spot prices clear excess capacity and then recover."),
)


def research_chat(results: Sequence[Result]) -> list[dict[str, str]]:
    done = "\n".join(f"- {r.hid} [{'SURVIVED' if r.survived else 'rejected'}] {r.hypothesis.family} "
                     f"{json.dumps(r.hypothesis.params)}: {r.hypothesis.statement} -> {r.summary}, p={r.p_value:.3f}, n={r.n}"
                     for r in results) or "(none yet)"
    families = "\n".join(f"- {name}: {desc}" for name, (_, desc) in FAMILIES.items())
    user = (
        "You are the research agent of a team forecasting posted GPU rental prices 1, 3 and 7 days ahead. "
        "Propose the next hypothesis about what moves these prices, to be tested on historical data. Choose one "
        "test family and its parameters. Prefer hypotheses that would change how the forecasters set their "
        "median or the width of their range, and follow up on earlier results (refine a survivor, or test "
        "whether it holds in another instrument class or horizon). Do not repeat a test already run.\n\n"
        f"Test families:\n{families}\nInstrument classes: {', '.join(CLASSES)}\n\nResults so far:\n{done}\n\n"
        'Answer with JSON only: {"family": "...", "params": {...}, "statement": "one sentence", '
        '"expected_sign": 1 or -1, "rationale": "one sentence"}'
    )
    return [{"role": "user", "content": user}]


class Notebook:
    """The persistent lab notebook (memory across iterations)."""

    def __init__(self, path: Path):
        self.path = path
        self.results: list[Result] = []
        if path.is_file():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                row["hypothesis"] = Hypothesis(**row["hypothesis"])
                self.results.append(Result(**row))

    def _write(self, result: Result) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(result.to_json(), default=str) + "\n")

    def seen(self, h: Hypothesis) -> bool:
        return any(r.hypothesis.family == h.family and r.hypothesis.params == h.params for r in self.results)

    def test(self, pack, h: Hypothesis, iteration: int) -> Result | None:
        if h.family not in FAMILIES or self.seen(h):
            return None
        result = run_test(pack, h, iteration=iteration, hid=f"H{len(self.results) + 1}")
        self.results.append(result)
        self._write(result)
        return result

    def iterate(self, pack, llm: LLM | None, iteration: int, n: int = 2) -> list[Result]:
        """Run the seed agenda first, then ``n`` agent-proposed hypotheses per iteration."""
        out = []
        for h in SEED_AGENDA:
            if not self.seen(h):
                r = self.test(pack, h, iteration)
                if r:
                    out.append(r)
        if llm is None:
            return out
        for _ in range(n):
            text = llm.generate([research_chat(self.results)], n=1, temperature=0.8, max_tokens=500)[0][0].text
            data = extract_json(text)
            if not isinstance(data, dict):
                continue
            try:
                h = Hypothesis(str(data["family"]), dict(data.get("params") or {}), str(data.get("statement", ""))[:300],
                               1 if float(data.get("expected_sign", 1)) >= 0 else -1, str(data.get("rationale", ""))[:300])
            except (KeyError, TypeError, ValueError):
                continue
            r = self.test(pack, h, iteration)
            if r:
                out.append(r)
        return out

    def findings(self) -> list[Result]:
        return [r for r in self.results if r.survived]


def findings_for(results: Sequence[Result], instrument_class: str) -> list[tuple[str, str]]:
    """(ID, text) of surviving findings that apply to this instrument class, for the shared board."""
    out = []
    for r in results:
        cls = r.hypothesis.params.get("instrument_class", "any")
        if r.survived and cls in ("any", instrument_class):
            out.append((f"F{r.hid[1:]}", r.finding_text()))
    return out
