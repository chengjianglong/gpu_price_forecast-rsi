"""Closed-loop PlayWorld rollouts: agent decides, world model generates, repeat.

Outputs follow PlayWorld's layout, one folder per task:
``<out>/<task_id>/{<task_id>.mp4, before_actions.jpg, after_actions.jpg, result.json}``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import END, is_insight_wait_only, reference_plan, reference_plan_length
from .agent import Agent, AgentContext
from .backends.base import WorldModel
from .video import load_rgb, save_jpg, write_mp4


@dataclass
class RolloutConfig:
    max_steps: int = 40            # PlayWorld's interaction budget
    tokens_per_step: int = 3       # reference-plan tokens per agent step
    insight_seconds: float = 60.0  # insight cases: hold still and observe this long
    insight_max_steps: int = 200
    prompt_source: str = "caption"  # text given to the world model: caption | objective | both


class Hook:
    """Extension point for memory, tracking and re-anchoring modules.

    A hook sees every rollout as it happens. Later components (3D memory,
    object tracker, still-scene guard) subclass this.
    """

    def on_reset(self, task: dict[str, Any], frames: list[np.ndarray]) -> None: ...

    def on_step(self, step: int, action: str, frames: list[np.ndarray]) -> None: ...

    def on_finish(self, result: dict[str, Any]) -> None: ...


def world_prompt(task: dict[str, Any], source: str) -> str:
    caption = str(task.get("image_caption") or "").strip()
    objective = str(task.get("prompt") or "").strip()
    if source == "objective":
        return objective
    if source == "both":
        return f"{caption} {objective}".strip()
    return caption or objective


def run_task(
    task: dict[str, Any],
    world: WorldModel,
    agent: Agent,
    out_root: Path,
    cfg: RolloutConfig = RolloutConfig(),
    hooks: list[Hook] = (),
) -> dict[str, Any]:
    task_id = task["task_id"]
    task_dir = Path(out_root) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    first = load_rgb(task["image_path"])
    all_frames = list(world.reset(first, world_prompt(task, cfg.prompt_source)))
    for hook in hooks:
        hook.on_reset(task, all_frames)
    save_jpg(all_frames[0], task_dir / "before_actions.jpg")

    stationary = is_insight_wait_only(task)
    max_steps = cfg.insight_max_steps if stationary else cfg.max_steps
    plan = reference_plan(task, max_steps, cfg.tokens_per_step)
    plan_steps = reference_plan_length(task, cfg.tokens_per_step)
    step_frames: list[list[np.ndarray]] = []
    records: list[dict[str, Any]] = []
    status, error = "completed", None
    try:
        for step in range(max_steps):
            if stationary:
                if len(all_frames) / world.fps >= cfg.insight_seconds:
                    break
                action, source = "WAIT", "fixed"
            else:
                ctx = AgentContext(
                    task=task, step=step, max_steps=max_steps, history=[r["action"] for r in records],
                    recommended=plan[step], plan_steps=plan_steps, initial_frame=first, step_frames=step_frames,
                )
                action, source = agent.decide(ctx), agent.name
            if action == END:
                records.append({"step": step, "action": END, "source": source, "recommended": plan[step]})
                break
            t0 = time.time()
            frames = world.step(action)
            step_frames.append(frames)
            all_frames.extend(frames)
            for hook in hooks:
                hook.on_step(step, action, frames)
            records.append({
                "step": step, "action": action, "source": source, "recommended": plan[step],
                "frames": len(frames), "seconds": round(time.time() - t0, 2),
            })
            print(f"  [{task_id}] step {step + 1}: {action} ({source}) -> {len(frames)} frames")
    except Exception as exc:  # keep what was generated; the failure is in result.json
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        print(f"  [{task_id}] FAILED: {error}")
    finally:
        world.finish()

    video = write_mp4(all_frames, task_dir / f"{task_id}.mp4", world.fps)
    save_jpg(all_frames[-1], task_dir / "after_actions.jpg")
    result = {
        "task_id": task_id,
        "model": world.name,
        "agent": agent.name,
        "group": task.get("_group"),
        "split": task.get("_split"),
        "view": task.get("perspective"),
        "prompt": task.get("prompt"),
        "image_path": task["image_path"],
        "dataset": task.get("_dataset"),
        "stationary": stationary,
        "actions": records,
        "total_frames": len(all_frames),
        "seconds_of_video": round(len(all_frames) / world.fps, 2),
        "elapsed_s": round(time.time() - started, 1),
        "video_path": str(video),
        "status": status,
        "error": error,
    }
    (task_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for hook in hooks:
        hook.on_finish(result)
    return result
