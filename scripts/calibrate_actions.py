#!/usr/bin/env python3
"""Measure how far each action turns the generated camera, and suggest gains.

Runs every action for a few steps from a fresh world per probe image, then
prints degrees per step, how much WAIT changes the picture, and the yaw/pitch
scale that would make one step turn --target-deg. Write the scales into the
backend config (Matrix-Game: yaw_scale / pitch_scale).

The right --target-deg comes from what PlayWorld's trajectory check expects
for one agent step; it is not published, so measure it from the reference
cases before trusting the suggested gain.

  python scripts/calibrate_actions.py --config configs/matrixgame3.yaml \
      --images third_party/PlayWorld/example/GC/001.jpg --target-deg 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcwm.calibrate import probe_actions, probes_to_rows, suggest_scales  # noqa: E402
from hcwm.config import build_world, load_config  # noqa: E402
from hcwm.video import load_rgb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--actions", nargs="+", default=["LEFT", "RIGHT", "UP", "DOWN", "W", "WAIT"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--hfov", type=float, default=90.0, help="assumed horizontal field of view, degrees")
    parser.add_argument("--target-deg", type=float, default=30.0)
    parser.add_argument("--out", type=Path, help="write the measurements as JSON")
    args = parser.parse_args()

    world = build_world(load_config(args.config))
    per_image = []
    try:
        for path in args.images:
            probes = probe_actions(world, load_rgb(path), args.prompt, args.actions, args.steps, args.hfov)
            per_image.append({"image": str(path), "probes": probes_to_rows(probes),
                              "suggested": suggest_scales(probes, args.target_deg)})
    finally:
        world.close()

    print(f"{'action':<7} {'yaw/step':>9} {'pitch/step':>11} {'mean |dI|':>10}")
    for item in per_image:
        print(f"-- {item['image']}")
        for row in item["probes"]:
            print(f"{row['action']:<7} {row['yaw_deg_per_step']:>9.2f} {row['pitch_deg_per_step']:>11.2f} "
                  f"{row['mean_abs_change']:>10.2f}")
        print(f"suggested for {args.target_deg} deg/step: {item['suggested']}")
    yaw = [i["suggested"]["yaw_scale"] for i in per_image if i["suggested"]["yaw_scale"]]
    pitch = [i["suggested"]["pitch_scale"] for i in per_image if i["suggested"]["pitch_scale"]]
    summary = {
        "yaw_scale": round(float(np.median(yaw)), 3) if yaw else None,
        "pitch_scale": round(float(np.median(pitch)), 3) if pitch else None,
    }
    print(f"median over images: {summary}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"images": per_image, "summary": summary, "target_deg": args.target_deg},
                                       indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
