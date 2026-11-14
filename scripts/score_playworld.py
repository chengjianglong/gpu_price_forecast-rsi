#!/usr/bin/env python3
"""Score a run with PlayWorld's own Rubric Verifier, several passes, and aggregate.

For every task in outputs/<run>/ this calls PlayWorld's scorer
(``python -m metrics.vqa.score``, Gemini 3.1 Pro) in score mode, plus gate
mode where PlayWorld gates (GC, third-person IF, out-of-sight OE). It then
runs PlayWorld's ``aggregate_fail_one_oe_split`` on each pass and reports the
mean and spread across passes. The paper measured a per-model variance of
about 0.011 between two passes, so report at least 3.

Needs GEMINI_API_KEY. --dry-run checks the plumbing without calling Gemini.

  python scripts/score_playworld.py --run abot_baseline --passes 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("gc", "if", "insight", "out_of_sight")


def view_of(perspective: str | None) -> str:
    return "third-person" if "third" in str(perspective or "").lower() else "first-person"


def needs_gate(item: dict) -> bool:
    return item["group"] == "GC" or (item["group"] == "IF" and item["view"] == "third-person") or (
        item["group"] == "OE" and item["split"] == "outsight"
    )


def scorer(playworld: Path, mode: str, item: dict, pass_dir: Path, run: str, dry_run: bool) -> int:
    key = f"{run}::{item['path']}"
    cmd = [
        sys.executable, "-m", "metrics.vqa.score", "--mode", mode,
        "--dataset", item["dataset"], "--task-id", item["task"], "--video", item["video"],
        "--world-model", run, "--cache-key", key,
        "--output", str(pass_dir / f"{item['task']}_{mode}.json"),
    ]
    if mode == "score":
        cmd += ["--scores-cache", str(pass_dir / "scores.json"), "--reference-image", item["image"]]
    else:
        cmd += ["--gc-completion-cache", str(pass_dir / "gc_completion.json")]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=playworld, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="folder name under outputs/")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--playworld", type=Path, default=ROOT / "third_party" / "PlayWorld")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true", help="only re-aggregate existing passes")
    args = parser.parse_args()

    if not (args.playworld / "metrics" / "vqa" / "score.py").is_file():
        parser.error(f"PlayWorld checkout not found at {args.playworld}; run scripts/setup_third_party.sh")
    if not args.dry_run and not args.skip_scoring and not os.environ.get("GEMINI_API_KEY"):
        parser.error("set GEMINI_API_KEY (or pass --dry-run)")

    run_dir = args.outputs / args.run
    videos = []
    for result_path in sorted(run_dir.glob("*/result.json")):
        r = json.loads(result_path.read_text())
        video = Path(r["video_path"])
        videos.append({
            "model": args.run, "path": f"{args.run}/{r['task_id']}/{video.name}", "task": r["task_id"],
            "group": r.get("group"), "split": r.get("split"), "view": view_of(r.get("view")),
            "dataset": r.get("dataset"), "video": str(video.resolve()), "image": r.get("image_path"),
        })
    if not videos:
        parser.error(f"no result.json files under {run_dir}")

    eval_dir = args.eval_root / args.run
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "videos_manifest.json").write_text(json.dumps(videos, indent=2) + "\n")

    if not args.skip_scoring:
        for k in range(1, args.passes + 1):
            pass_dir = eval_dir / f"pass{k}"
            pass_dir.mkdir(parents=True, exist_ok=True)
            for item in videos:
                print(f"[pass {k}] {item['task']}")
                scorer(args.playworld, "score", item, pass_dir, args.run, args.dry_run)
                if needs_gate(item):
                    scorer(args.playworld, "gate", item, pass_dir, args.run, args.dry_run)

    sys.path.insert(0, str(args.playworld))
    from metrics.aggregate import aggregate_fail_one_oe_split  # PlayWorld's own aggregation

    rows = []
    for pass_dir in sorted(eval_dir.glob("pass*")):
        scores_path, gate_path = pass_dir / "scores.json", pass_dir / "gc_completion.json"
        scores = json.loads(scores_path.read_text()) if scores_path.is_file() else {}
        gates = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
        row = aggregate_fail_one_oe_split(videos, scores, gates, models=((args.run, args.run),))[0]
        rows.append({"pass": pass_dir.name, **{d: row[d]["average"] for d in DIMENSIONS}, "overall": row["overall"]})

    summary = {}
    for key in (*DIMENSIONS, "overall"):
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = {
            "mean": round(statistics.mean(values), 3) if values else None,
            "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else None,
            "passes": len(values),
        }
    (eval_dir / "summary.json").write_text(json.dumps({"passes": rows, "summary": summary}, indent=2) + "\n")
    print(f"\n| {args.run} | GC | IF | IE | OE | Overall |\n|---|---|---|---|---|---|")
    cells = [f"{summary[k]['mean']}" + (f" ± {summary[k]['stdev']}" if summary[k]["stdev"] is not None else "")
             for k in (*DIMENSIONS, "overall")]
    print(f"| mean over {len(rows)} passes | " + " | ".join(cells) + " |")
    print("Note: tasks missing from this run count as 1 (PlayWorld's Fail = 1 policy).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
