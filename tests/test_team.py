"""Offline tests for parsing, the Aggregator, the board's no-look-ahead rule and
one full team run with the fake LLM on a synthetic pack."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from gpr.agents import extract_json, parse_challenges, parse_forecast
from gpr.board import build_board
from gpr.fake import fake_llm
from gpr.pack import HORIZONS, Pack, PackFiles, Splits
from gpr.team import Team, TeamConfig
from gpr.weights import VotingWeights, WeightRule, aggregate

FALLBACK = {h: (1.0, 1.0, 1.0) for h in HORIZONS}


def test_extract_json_handles_fences_thinking_and_prose():
    assert extract_json('<think>hmm {"x": 0}</think> Sure: {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    assert extract_json("```json\n{\"a\": 2}\n```") == {"a": 2}
    assert extract_json("nothing here") is None
    # stopped before the closing brace, with braces inside a string
    assert extract_json('{"forecasts": {"1": [1, 2, 3]}, "reasoning": "a {b} c"') == {
        "forecasts": {"1": [1, 2, 3]}, "reasoning": "a {b} c"}


def test_parse_forecast_repairs_and_records_problems():
    text = json.dumps({"forecasts": {"1": [2.0, 1.9, 2.1], "3": [-1, 2, 3], "7": [1, 2, 3]},
                       "evidence": ["E1", "E99"], "reasoning": "r"})
    pos = parse_forecast(text, "momentum", {"E1", "E2"}, FALLBACK)
    assert pos.quantiles[1] == (1.9, 2.0, 2.1)
    assert pos.quantiles[3][0] > 0
    assert pos.quantiles[30] == (1.0, 1.0, 1.0)
    assert {"crossed_h1", "nonpositive_h3", "missing_h30"} <= set(pos.problems)
    assert any(p.startswith("bad_citation") for p in pos.problems)
    assert pos.evidence == ["E1"] and not pos.fallback
    bad = parse_forecast("no json", "momentum", set(), FALLBACK)
    assert bad.fallback and bad.problems == ["unparseable"]


def test_parse_challenges_limits_and_validates():
    text = json.dumps({"challenges": [{"target": "momentum", "claim": "a", "evidence": ["E1"]}] * 3
                      + [{"target": "nobody", "claim": "b"}]})
    found, problems = parse_challenges(text, "critic", ["momentum", "reversion"], {"E1"})
    assert len(found) == 2 and all(c.target == "momentum" for c in found)
    assert any(p.startswith("unknown_target") for p in problems)


def test_aggregate_votes_and_widens_with_disagreement():
    agree = aggregate({"a": (1, 2, 3), "b": (1, 2, 3)}, {"a": 0.5, "b": 0.5})
    assert agree == (1, 2, 3)
    split = aggregate({"a": (1, 2, 3), "b": (2, 3, 4)}, {"a": 0.5, "b": 0.5}, widen=1.0)
    assert split[1] == 2.5 and split[0] < 1.5 and split[2] > 3.5
    assert aggregate({"a": (1, 2, 3), "b": (9, 9, 9)}, {"a": 1.0, "b": 0.0}) == (1, 2, 3)


def test_hedge_update_moves_weight_to_lower_loss_and_keeps_floor():
    w = VotingWeights(["a", "b", "c"], WeightRule(eta=5, decay=0.0, w_min=0.05))
    for _ in range(20):
        new = w.update(("spot", 1), {"a": 0.1, "b": 1.0, "c": 2.0})
    assert new["a"] > new["b"] > new["c"] >= 0.05 - 1e-9
    assert sum(new.values()) == pytest.approx(1.0)


def test_splits_are_ordered():
    Splits().check()


# ---- a synthetic pack -------------------------------------------------------------
@pytest.fixture
def tiny_pack(tmp_path) -> Pack:
    start = date(2026, 6, 1)
    rows = []
    for sid, base in (("src.neocloud_rate_card.H100.on_demand.median", 2.5), ("src.neocloud_spot.B200.spot.median", 4.0)):
        for i in range(120):
            value = base * (0.99 if i > 90 else 1.0)
            rows.append({"series_id": sid, "date": (start + timedelta(days=i)).isoformat(), "value": value,
                         "n_providers": 5, "is_filled": 0, "instrument_class": sid.split(".")[1]})
    files = PackFiles(date(2026, 9, 28), tmp_path)
    pd.DataFrame(rows).to_csv(files.series, index=False)
    cat = pd.DataFrame([{"series_id": s, "tier": "headline", "instrument_class": s.split(".")[1], "gpu_model": "H100",
                         "pricing_tier": "on_demand", "source_id": "src", "instance_type": None,
                         "benchmark_contract": "test contract", "truth_availability": "daily"}
                        for s in sorted({r["series_id"] for r in rows})])
    cat.to_csv(files.catalog, index=False)
    return Pack(files)


def baselines_for(pack: Pack, origin: date) -> dict:
    out = {}
    for sid in pack.series_ids():
        last = float(pack.history(sid, origin)["value"].iloc[-1])
        out[(sid, origin)] = {m: {h: (last * 0.95, last, last * 1.05) for h in HORIZONS}
                              for m in ("last_value_empirical", "drift")}
    return out


def test_board_never_shows_values_after_the_origin(tiny_pack: Pack):
    origin = date(2026, 8, 15)
    sid = tiny_pack.series_ids()[0]
    board = build_board(tiny_pack, sid, origin, baselines_for(tiny_pack, origin)[(sid, origin)])
    text = board.render_evidence()
    assert "2026-08-16" not in text and "08-20" not in text
    assert tiny_pack.history(sid, origin)["date"].max() == origin
    assert board.last_value == 2.5  # the drop to 0.99x happens on day 91 (2026-08-31)


def test_team_runs_one_origin_with_the_fake_llm(tiny_pack: Pack, monkeypatch):
    origin = date(2026, 9, 1)
    team = Team(TeamConfig(debate_rounds=1), fake_llm(), tiny_pack)
    monkeypatch.setattr(team, "baselines_for", lambda o: baselines_for(tiny_pack, o))
    result = team.run(origin, tiny_pack.series_ids())
    assert len(result.rows) == 2 * len(HORIZONS)
    assert (result.rows["q10"] <= result.rows["q50"]).all() and (result.rows["q50"] <= result.rows["q90"]).all()
    assert (result.rows["origin_date"] == origin.isoformat()).all()
    # 3 forecasters + 1 critic + 3 revisions per series, no events so no Market Analyst call
    assert result.calls == 2 * 7
    board = result.boards[0]
    assert set(board.positions) == set(board.revised) == {"momentum", "reversion", "supply_demand"}
    assert board.challenges and board.challenges[0].target == "momentum"
