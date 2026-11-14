"""Offline tests for rewards, truth caps, the Improver's guard rails and the
acceptance test."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from gpr.board import Position
from gpr.improve import apply_edits
from gpr.evaluate import paired_bootstrap
from gpr.pack import HORIZONS, Splits
from gpr.rewards import TRAIN_TRUTH_CAP, VALIDATION_TRUTH_CAP, cell_skill, forecast_reward, pinball
from gpr.team import TeamConfig


def test_truth_caps_keep_training_and_validation_apart():
    s = Splits()
    assert TRAIN_TRUTH_CAP < min(s.validation)
    assert max(s.validation) < VALIDATION_TRUTH_CAP == min(s.test)
    assert len(s.test) == 3


def test_pinball_matches_the_scorer_definition():
    # y above the forecast: losses 0.1*1, 0.5*1, 0.9*1 -> mean 0.5
    assert pinball(3.0, (2.0, 2.0, 2.0)) == pytest.approx(0.5)
    assert pinball(2.0, (2.0, 2.0, 2.0)) == 0.0


def test_cell_skill_is_bounded_and_rewards_tight_flat_forecasts():
    assert cell_skill((2.0, 2.0, 2.0), 2.0, 2.0) == 1.0          # flat series, flat forecast
    assert cell_skill((1.0, 2.0, 3.0), 2.0, 2.0) < 0              # flat series, wide range is penalised
    assert cell_skill((0.1, 0.1, 0.1), 2.0, 2.0) == -3.0          # clipped
    assert cell_skill((2.9, 3.0, 3.1), 3.0, 2.0) > cell_skill((2.0, 2.0, 2.0), 3.0, 2.0)


def test_forecast_reward_penalises_repairs_and_unparseable():
    q = {h: (2.0, 2.0, 2.0) for h in HORIZONS}
    clean = Position("a", q, evidence=["E1"])
    repaired = Position("a", q, evidence=["E1"], problems=["crossed_h1", "no_citation"])
    broken = Position("a", q, problems=["unparseable"], fallback=True)
    truths = {1: 2.0, 3: 2.0}
    assert forecast_reward(clean, truths, 2.0) == 1.0
    assert forecast_reward(repaired, truths, 2.0) == pytest.approx(1.0 - 0.15)
    assert forecast_reward(broken, truths, 2.0) < -3


def test_improver_edits_are_whitelisted_and_bounded():
    cfg = TeamConfig()
    new, log = apply_edits(cfg, [{"path": "rule.widen", "value": 9}, {"path": "debate_rounds", "value": 1.6},
                                 {"path": "base_model", "value": "evil"}, {"path": "lenses.momentum.view", "value": "x" * 900},
                                 {"path": "lenses.nobody.view", "value": "y"}])
    assert new.rule.widen == 2.0 and new.debate_rounds == 2
    assert new.base_model == cfg.base_model
    assert len(next(l for l in new.lenses if l.name == "momentum").view) == 600
    assert cfg.rule.widen == 0.5  # the original is untouched
    assert log == ["rule.widen=2", "debate_rounds=2", "lenses.momentum.view rewritten"]


def test_paired_bootstrap_detects_a_clear_improvement():
    rows = []
    for s in range(30):
        for h in (1, 3, 7):
            rows.append({"series_id": f"s{s}", "origin_date": "2026-10-01", "horizon_days": h,
                         "loss": 1.0, "baseline_loss": 1.0})
    old = pd.DataFrame(rows)
    better = old.assign(loss=old["loss"] * 0.8)
    same = old.copy()
    assert paired_bootstrap(better, old)["p_better"] > 0.99
    assert paired_bootstrap(better, old)["delta"] == pytest.approx(0.2)
    assert paired_bootstrap(same, old)["p_better"] < 0.5


def test_research_notebook_tests_on_the_training_window_and_remembers(tmp_path):
    from datetime import date, timedelta

    from gpr.pack import Pack, PackFiles
    from gpr.research import Hypothesis, Notebook, findings_for

    rows = []
    for k in range(25):  # 25 rate-card series that change once every 20 days
        sid = f"src{k}.neocloud_rate_card.H100.on_demand.median"
        for i in range(260):  # runs past the 30 Sep training cap
            rows.append({"series_id": sid, "date": (date(2026, 3, 1) + timedelta(days=i)).isoformat(),
                         "value": 2.0 + 0.1 * (i // 20) + 0.01 * k, "n_providers": 4, "is_filled": 0,
                         "instrument_class": "neocloud_rate_card"})
    files = PackFiles(date(2026, 11, 6), tmp_path)
    pd.DataFrame(rows).to_csv(files.series, index=False)
    pack = Pack(files)
    nb = Notebook(tmp_path / "notebook.jsonl")
    h = Hypothesis("stickiness", {"instrument_class": "neocloud_rate_card", "horizon": 7, "threshold": 0.5},
                   "Rate cards rarely move within 7 days.", +1)
    r = nb.test(pack, h, iteration=0)
    assert r.survived and r.tested_until == "2026-09-30"
    assert nb.test(pack, h, iteration=1) is None          # never re-runs a test
    assert Notebook(tmp_path / "notebook.jsonl").results[0].hid == r.hid   # persisted
    assert findings_for(nb.findings(), "neocloud_rate_card")[0][0] == "F1"
    assert findings_for(nb.findings(), "neocloud_spot") == []
    # the test never reads data after the training truth cap
    assert pack.series[pack.series["date"] > date(2026, 9, 30)].shape[0] > 0
