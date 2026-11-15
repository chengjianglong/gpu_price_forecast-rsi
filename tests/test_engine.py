"""Offer tables, the list-replay engine, the shuffled-data control, list evidence and the range
stretch. The last test needs Pacific's pack and is skipped without it."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpr import engine
from gpr import pack as packlib
from gpr.board import build_board
from gpr.fake import fake_llm
from gpr.improve import apply_edits
from gpr.offers import Offers, OfferTable, build_table, list_evidence, make_placebo, median_moves, verify
from gpr.pack import HORIZONS, Pack, PackFiles
from gpr.team import Team, TeamConfig
from gpr.weights import WeightRule, stretch

START = date(2026, 6, 1)
SID = "src.neocloud_rate_card.H100.on_demand.median"
FIXTURE = Path(__file__).parent / "fixtures" / "engine_2026-11-06.csv"


def offer_rows(n_days: int = 100) -> pd.DataFrame:
    """Five providers. ``mover`` reprices every day, ``stepper`` once, ``late`` joins on day 60
    and ``gone`` leaves after day 80; ``flat`` and ``cheap`` never change."""
    rows = []
    for i in range(n_days):
        day = (START + timedelta(days=i)).toordinal()
        prices = {"cheap": 1.0, "flat": 2.0, "stepper": 2.2 if i < 50 else 2.6, "mover": 3.0 + 0.01 * (i % 7)}
        if i >= 60:
            prices["late"] = 1.5
        if i <= 80:
            prices["gone"] = 2.4
        for provider, price in prices.items():
            rows.append({"day": day, "snapshot": 0, "provider": provider, "sku": "h100", "region": "", "price": price})
    return pd.DataFrame(rows)


def make_table(n_days: int = 100, rows: pd.DataFrame | None = None) -> OfferTable:
    rows = offer_rows(n_days) if rows is None else rows
    series = rows.groupby("day")["price"].median().round(4).reset_index()
    series["date"] = [date.fromordinal(d) for d in series["day"]]
    return build_table(SID, "neocloud_rate_card", series.rename(columns={"price": "value"}), rows)


def test_median_of_the_offers_reproduces_the_series():
    table = make_table()
    check = verify([table]).iloc[0]
    assert check["match_share"] == 1.0 and check["days_with_offers"] == 1.0
    assert table.prices.shape == (100, 6)
    assert any(s.startswith("late|") for s in table.slots)


def test_cut_holds_nothing_after_the_origin():
    table = make_table()
    origin = START + timedelta(days=55)
    cut = table.cut(origin)
    assert cut.last_day == origin and len(cut.days) == 56
    assert not any(s.startswith("late|") for s in cut.slots)      # it joins on day 60
    assert table.cut(START - timedelta(days=1)) is None


def test_engine_cannot_see_past_the_origin():
    """Changing everything after the origin must not change the forecast made at the origin."""
    table = make_table()
    origin = START + timedelta(days=70)
    changed = make_table()
    changed.prices[71:] *= 3.0
    changed.value[71:] *= 3.0
    assert engine.forecast(table.cut(origin)) == engine.forecast(changed.cut(origin))


def test_engine_replays_repricing_and_the_neighbour_caps_the_middle():
    """The middle offer rises 1% a day to 2.9 today, under a flat offer at 3.0. Replayed, it is at
    2.9 x 1.01 after a day; after seven it would pass 3.0, so the 3.0 offer is the middle."""
    rows = pd.DataFrame([{"day": (START + timedelta(days=i)).toordinal(), "snapshot": 0, "provider": p, "sku": "",
                          "region": "", "price": v}
                         for i in range(40) for p, v in (("a", 1.0), ("b", 2.9 * 1.01 ** (i - 39)), ("c", 3.0))])
    out = engine.forecast(make_table(rows=rows))
    assert out[1] == pytest.approx((2.929,) * 3) and out[7] == pytest.approx((3.0,) * 3)


def test_engine_gives_no_forecast_with_too_little_history():
    assert engine.forecast(make_table(8)) == {}
    assert set(engine.forecast(make_table(15))) == {1, 3}      # 14 one-day pairs, 12 three-day pairs, 8 seven-day pairs


def test_forecast_rows_fall_back_to_drift_and_label_it():
    table = make_table()
    origin = START + timedelta(days=99)
    stale = START + timedelta(days=105)
    drift = pd.DataFrame([{"series_id": s, "origin_date": o.isoformat(), "horizon_days": h, "q10": 1.0, "q50": 2.0,
                           "q90": 3.0} for s in (SID, "other") for o in (origin, stale) for h in (1, 3, 7)])

    def table_for(sid, o):
        return table.cut(o) if sid == SID else None

    rows = engine.forecast_rows(table_for, origin, [SID, "other", "unknown"], drift)
    assert list(rows.columns[:6]) == ["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"]
    assert set(rows[rows["series_id"] == SID]["method_note"]) == {"engine"}
    other = rows[rows["series_id"] == "other"]
    assert set(other["method_note"]) == {"drift_fallback"} and (other["q50"] == 2.0).all()
    assert "unknown" not in set(rows["series_id"])                 # no engine forecast and no baseline row
    assert (rows["q10"] <= rows["q50"]).all() and (rows["q50"] <= rows["q90"]).all()
    late = engine.forecast_rows(table_for, stale, [SID], drift)    # no list on the origin day
    assert set(late["method_note"]) == {"drift_fallback"}


def test_placebo_keeps_the_changes_and_shuffles_their_order():
    table = make_table()
    fake = make_placebo(table, seed=3)
    assert fake.prices.shape == table.prices.shape
    assert np.array_equal(np.isnan(fake.prices), np.isnan(table.prices))         # same listing pattern
    assert verify([fake]).iloc[0]["match_share"] == 1.0                          # value is the median again
    j = next(i for i, s in enumerate(table.slots) if s.startswith("mover|"))
    real, shuffled = np.diff(np.log(table.prices[:, j])), np.diff(np.log(fake.prices[:, j]))
    assert np.allclose(np.sort(real), np.sort(shuffled))                         # the same set of changes
    assert not np.allclose(real, shuffled)                                       # in a different order
    assert np.array_equal(make_placebo(table, seed=3).prices, fake.prices, equal_nan=True)   # repeatable


def test_median_moves_splits_by_cause():
    out = median_moves(make_table())
    assert out["offers"] == 6 and out["frequent_offers"] == 1
    assert out["list"] > 0                                    # ``late`` joining and ``gone`` leaving move the median
    assert out["rare"] > 0 and out["frequent"] == 0           # ``stepper`` is near the middle, ``mover`` is not
    swapped = pd.DataFrame([{"day": START.toordinal() + i, "snapshot": 0, "provider": p, "sku": "", "region": "",
                             "price": v} for i, (p, v) in enumerate((("a", 1.0), ("b", 2.0)))])
    assert median_moves(make_table(rows=swapped))["list"] == pytest.approx(np.log(2))   # the whole list replaced


def test_list_evidence_marks_the_middle_and_names_who_joined():
    text = list_evidence(make_table().cut(START + timedelta(days=65)))
    assert "6 offers" in text[0] and text[0].count(" << (") == 2   # an even list has two middle offers
    assert "stepper|h100 2.6" in text[0] and "last change 07-21: 2.2 -> 2.6" in text[0]
    assert "joined: late|h100" in text[1] and "left: none" in text[1]
    after = list_evidence(make_table().cut(START + timedelta(days=90)))
    assert "left: gone|h100" in after[1]


def test_list_evidence_takes_the_middle_over_every_snapshot():
    rows = offer_rows()
    again = rows[rows["day"] == rows["day"].max()].assign(snapshot=1)      # a second snapshot today ...
    again.loc[again["provider"] == "cheap", "price"] = 5.0                # ... in which one offer has repriced
    table = make_table(rows=pd.concat([rows, again]))
    text, changes = list_evidence(table)
    assert "10 observations of 5 offers" in text and text.count(" x2 ") == 4
    marked = [float(line.split()[1]) for line in text.split("\n") if " << (" in line]
    assert marked == [2.0, 2.6] and table.value[-1] == 2.3         # the official median is between them
    assert "joined: none; left: none" in changes                   # a new snapshot is not a new offer


def test_stretch_widens_around_the_median_and_one_is_a_no_op():
    assert stretch((1.9, 2.0, 2.2), 1.0) == (1.9, 2.0, 2.2)
    q10, q50, q90 = stretch((1.9, 2.0, 2.2), 2.0)
    assert q50 == 2.0 and q10 == pytest.approx(2.0 * 0.95 ** 2) and q90 == pytest.approx(2.0 * 1.1 ** 2)
    assert stretch((2.0, 2.0, 2.0), 3.0) == (2.0, 2.0, 2.0)        # a flat forecast stays flat
    rule = WeightRule(stretch_7=1.65)
    assert rule.stretch_for(7) == 1.65 and rule.stretch_for(3) == 1.0 and rule.stretch_for(30) == 1.0


def test_a_saved_team_config_restores_in_full():
    cfg = TeamConfig.load(Path(__file__).parents[1] / "configs" / "team_list.yaml")
    assert len(cfg.lenses) == 5 and TeamConfig.from_json(json.loads(json.dumps(cfg.to_json()))) == cfg


def test_improver_may_set_the_stretch_within_bounds():
    new, log = apply_edits(TeamConfig(), [{"path": "rule.stretch_7", "value": 9}, {"path": "rule.stretch_3", "value": 1.3}])
    assert new.rule.stretch_7 == 3.0 and new.rule.stretch_3 == 1.3 and log == ["rule.stretch_7=3", "rule.stretch_3=1.3"]


# ---- the team with the list on the board ----------------------------------------------------
@pytest.fixture
def list_pack(tmp_path) -> Pack:
    table = make_table()
    files = PackFiles(START + timedelta(days=99), tmp_path)
    pd.DataFrame({"series_id": SID, "date": [date.fromordinal(int(d)).isoformat() for d in table.days],
                  "value": table.value, "n_providers": 5, "is_filled": 0,
                  "instrument_class": "neocloud_rate_card"}).to_csv(files.series, index=False)
    pd.DataFrame([{"series_id": SID, "tier": "headline", "table": "daily_statistics",
                   "instrument_class": "neocloud_rate_card", "gpu_model": "H100", "pricing_tier": "on_demand",
                   "source_id": "src", "instance_type": None, "benchmark_contract": "test contract",
                   "truth_availability": "daily"}]).to_csv(files.catalog, index=False)
    pack = Pack(files)
    pack._offers = Offers(pack)
    pack._offers._tables[SID] = table       # in place of reading rental_observations
    return pack


def test_board_shows_the_list_only_up_to_the_origin(list_pack: Pack):
    origin = START + timedelta(days=55)
    flat = {m: {h: (2.0, 2.2, 2.4) for h in HORIZONS} for m in ("last_value_empirical",)}
    plain = build_board(list_pack, SID, origin, flat)
    board = build_board(list_pack, SID, origin, flat, offers=list_pack._offers.table(SID, origin))
    text = board.render_evidence()
    assert len(board.evidence) == len(plain.evidence) + 2
    assert "The list behind this median on 2026-07-26" in text
    uncut = build_board(list_pack, SID, origin, flat, offers=list_pack._offers.full_table(SID))
    assert len(uncut.evidence) == len(plain.evidence)               # a list dated after the origin is not shown


def test_team_posts_the_engine_and_stretches_the_range(list_pack: Pack, monkeypatch):
    origin = START + timedelta(days=55)
    last = float(list_pack.history(SID, origin)["value"].iloc[-1])
    rows = pd.DataFrame([{"series_id": SID, "origin_date": origin.isoformat(), "horizon_days": h, "q10": last * 0.9,
                          "q50": last, "q90": last * 1.1, "method": m}
                         for m in ("last_value_empirical", "drift") for h in HORIZONS])
    monkeypatch.setattr("gpr.team.baseline_rows", lambda pack, method, o, cache: rows[rows["method"] == method])
    cfg = TeamConfig(debate_rounds=0, baseline_methods=("last_value_empirical", "drift", "engine"), list_evidence=True,
                     rule=WeightRule(widen=0.0, stretch_7=2.0))
    result = Team(cfg, fake_llm(), list_pack, cache_dir=list_pack.files.directory / "baselines").run(origin, [SID])
    text = result.boards[0].render_evidence()
    assert "Baseline engine (q10, q50, q90): h=1:" in text and "The list behind this median" in text
    assert "late|h100" not in text                                  # joins four days after the origin
    seen = engine.forecast(list_pack._offers.table(SID, origin))
    assert result.boards[0].baselines["engine"][7] == pytest.approx(seen[7], abs=1e-6)
    assert result.calls == 3                                        # three Forecasters, no debate
    by_h = {int(r.horizon_days): r for r in result.rows.itertuples()}
    assert by_h[1].q10 == pytest.approx(last * 0.9) and by_h[1].q90 == pytest.approx(last * 1.1)
    assert by_h[7].q10 == pytest.approx(last * 0.9 ** 2, rel=1e-5) and by_h[7].q90 == pytest.approx(last * 1.1 ** 2, rel=1e-5)
    plain = Team(TeamConfig(debate_rounds=0), fake_llm(), list_pack,
                 cache_dir=list_pack.files.directory / "baselines").run(origin, [SID])
    assert "engine" not in plain.boards[0].render_evidence() and "The list behind" not in plain.boards[0].render_evidence()


# ---- the port against the sister project's recorded forecast (needs the pack) ---------------
@pytest.mark.skipif(not (packlib.PACK_DIR / "rental_observations").is_dir(), reason="needs Pacific's pack")
def test_port_reproduces_the_recorded_engine_forecast():
    """``gpu-price`` recorded its engine's 6 Nov forecast for the 37 headline series
    (``submission/alt_engine_2026-11-06.csv`` there). The port must give the same numbers."""
    ref = pd.read_csv(FIXTURE)
    pack = Pack(packlib.build(packlib.DATA_CUTOFF))
    offers = Offers(pack)
    origin = packlib.DATA_CUTOFF
    assert len(ref) == 111 and ref["method_note"].str.startswith("engine").all()
    for sid, group in ref.groupby("series_id"):
        table = offers.table(sid, origin)
        assert verify([table]).iloc[0]["match_share"] == 1.0
        ours = engine.forecast(table)
        for row in group.itertuples():
            got = ours[int(row.horizon_days)]
            assert (round(got[0], 6), round(got[1], 6), round(got[2], 6)) == (row.q10, row.q50, row.q90)
