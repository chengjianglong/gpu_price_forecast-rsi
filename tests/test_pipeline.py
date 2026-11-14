"""CPU tests: action parsing, calibration against known motion, and a full
closed-loop rollout on the mock world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hcwm.actions import (
    END, dataset_segments, is_insight_wait_only, parse_agent_reply, reference_plan, reference_plan_length,
)
from hcwm.agent import ScriptedAgent
from hcwm.backends.mock import MockWorld
from hcwm.calibrate import camera_motion, probe_actions, suggest_scales
from hcwm.runner import RolloutConfig, run_task
from hcwm.tasks import load_tasks

GC_TASK = {
    "task_id": "GC002",
    "category": "GC",
    "prompt": "Move along a rectangular path around the sculpture, then return.",
    "image_path": "GC/001.jpg",
    "perspective": "first person",
    "action": "d*3 -> w*12 -> a*6 -> s*12 -> d*3",
    "action_sequence_steps": ["hold(D,1350ms)", "hold(W,5400ms)", "hold(A,2700ms)", "hold(S,5400ms)", "hold(D,1350ms)"],
}


def textured_image(h: int = 360, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "gc").mkdir()
    (tmp_path / "insight").mkdir()
    Image.fromarray(textured_image()).save(tmp_path / "gc" / "001.jpg")
    Image.fromarray(textured_image(seed=1)).save(tmp_path / "insight" / "001.jpg")
    (tmp_path / "gc" / "data.json").write_text(json.dumps([{**GC_TASK, "image_path": "gc/001.jpg"}]))
    insight = {"task_id": "OE001", "category": "OE", "prompt": "Watch the candle burn down.",
               "image_path": "insight/001.jpg", "action": "wait*30"}
    (tmp_path / "insight" / "data.json").write_text(json.dumps([insight]))
    return tmp_path


def test_dataset_segments_and_plan():
    assert dataset_segments(GC_TASK) == [("D", 3), ("W", 12), ("A", 6), ("S", 12), ("D", 3)]
    plan = reference_plan(GC_TASK, num_steps=40, tokens_per_step=3)
    assert plan[:6] == ["D", "W", "W", "W", "W", "A"]
    assert len(plan) == 40
    # falls back to the hold() steps when the compact string is missing
    no_compact = {**GC_TASK, "action": ""}
    assert dataset_segments(no_compact) == [("D", 3), ("W", 12), ("A", 6), ("S", 12), ("D", 3)]


def test_parse_agent_reply():
    allowed = ("W", "A", "S", "D", "LEFT", "RIGHT", "UP", "DOWN", "WAIT", END)
    assert parse_agent_reply("LEFT", allowed) == "LEFT"
    assert parse_agent_reply("wait.", allowed) == "WAIT"
    assert parse_agent_reply("END", allowed) == END
    assert parse_agent_reply("no idea", allowed) is None


def test_insight_detection(data_root: Path):
    tasks = {t["task_id"]: t for t in load_tasks(data_root)}
    assert is_insight_wait_only(tasks["OE001"])
    assert not is_insight_wait_only(tasks["GC002"])
    assert tasks["OE001"]["_group"] == "OE" and tasks["GC002"]["_group"] == "GC"
    assert Path(tasks["GC002"]["image_path"]).is_file()


def test_camera_motion_recovers_known_yaw():
    world = MockWorld(fps=12, step_seconds=1.0, turn_deg_per_s=20)
    world.reset(textured_image(), "")
    frames = world.step("RIGHT") + world.step("RIGHT")
    yaw = camera_motion(frames, hfov_deg=90)["yaw_deg"]
    # 2 steps x 12 frames = 23 frame-to-frame moves of 20/12 deg each
    assert yaw == pytest.approx(40 * 23 / 24, rel=0.03)
    left = camera_motion(world.step("LEFT"), hfov_deg=90)["yaw_deg"]
    assert left < 0


def test_calibration_suggests_gain_for_under_turning_model():
    world = MockWorld(fps=12, step_seconds=1.0, turn_deg_per_s=30, yaw_gain=0.5)
    probes = probe_actions(world, textured_image(), "", actions=("LEFT", "RIGHT", "WAIT"), steps=2)
    by = {p.action: p for p in probes}
    assert by["RIGHT"].yaw_deg_per_step == pytest.approx(15 * 23 / 24, rel=0.03)
    assert by["LEFT"].yaw_deg_per_step == pytest.approx(-15 * 23 / 24, rel=0.03)
    assert by["WAIT"].mean_abs_change == pytest.approx(0, abs=1e-6)  # the mock freezes on WAIT
    assert suggest_scales(probes, target_deg_per_step=30)["yaw_scale"] == pytest.approx(2.0 * 24 / 23, rel=0.03)


def test_closed_loop_rollout_on_mock(data_root: Path, tmp_path: Path):
    tasks = {t["task_id"]: t for t in load_tasks(data_root)}
    world = MockWorld(fps=12, step_seconds=1.0)
    out = tmp_path / "out"

    # the reference plan d*3 -> w*12 -> a*6 -> s*12 -> d*3 is 1+4+2+4+1 = 12 steps
    assert reference_plan_length(tasks["GC002"], 3) == 12
    gc = run_task(tasks["GC002"], world, ScriptedAgent(), out, RolloutConfig(max_steps=40))
    assert gc["status"] == "completed", gc["error"]
    assert [a["action"] for a in gc["actions"]][:3] == ["D", "W", "W"]
    assert gc["actions"][-1]["action"] == END and len(gc["actions"]) == 13
    assert gc["total_frames"] == 1 + 12 * 12
    for name in ("GC002.mp4", "before_actions.jpg", "after_actions.jpg", "result.json"):
        assert (out / "GC002" / name).is_file()

    insight = run_task(tasks["OE001"], world, ScriptedAgent(), out, RolloutConfig(insight_seconds=10))
    assert insight["stationary"] and insight["status"] == "completed"
    assert {a["action"] for a in insight["actions"]} == {"WAIT"}
    assert insight["seconds_of_video"] >= 10
