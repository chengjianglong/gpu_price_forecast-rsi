#!/usr/bin/env python3
"""Run PlayWorld cases in closed loop against one world model.

Examples:
  # plumbing check on CPU, no API key
  python scripts/run_playworld.py --config configs/mock.yaml --agent scripted \
      --data-root third_party/PlayWorld/example --dataset third_party/PlayWorld/example/GC/001.json

  # a real baseline (GPU + ANTHROPIC_API_KEY)
  python scripts/run_playworld.py --config configs/abot.yaml --data-root data/playworld --run abot_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcwm.agent import make_agent  # noqa: E402
from hcwm.config import PROJECT_ROOT, build_rollout, build_world, load_config  # noqa: E402
from hcwm.runner import run_task  # noqa: E402
from hcwm.tasks import SPLITS, load_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path, help="PlayWorld data folder")
    parser.add_argument("--dataset", type=Path, help="one task JSON file instead of every split")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--tasks", nargs="+", default=[], help="task IDs, e.g. GC001 OE014")
    parser.add_argument("--limit", type=int, help="run only the first N tasks")
    parser.add_argument("--agent", choices=["claude", "scripted"], default="claude")
    parser.add_argument("--run", help="run name; outputs go to outputs/<run>/ (default: backend name)")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--resume", action="store_true", help="skip tasks that already completed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rollout = build_rollout(cfg)
    tasks = load_tasks(args.data_root, args.splits, args.tasks, args.dataset)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        parser.error(f"no tasks found under {args.data_root}")

    agent = make_agent(args.agent)
    world = build_world(cfg)
    out = args.out_root / (args.run or world.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"config": cfg, "agent": args.agent}, indent=2) + "\n")

    done = failed = 0
    try:
        for i, task in enumerate(tasks, 1):
            result_path = out / task["task_id"] / "result.json"
            if args.resume and result_path.is_file() and json.loads(result_path.read_text()).get("status") == "completed":
                continue
            print(f"[{i}/{len(tasks)}] {task['task_id']} ({task['_split']}): {task.get('prompt', '')[:80]}")
            result = run_task(task, world, agent, out, rollout)
            done += result["status"] == "completed"
            failed += result["status"] != "completed"
    finally:
        world.close()
    print(f"Finished: {done} completed, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
