"""Choose between base models (Qwen vs DeepSeek) on validation only.

    python scripts/compare_models.py outputs/rsi/overnight outputs/rsi/overnight_deepseek

Both runs use the same config, seeds, training pool, validation origins and
stopping rules; only ``rl.base_model`` differs. The decision rule is fixed
before looking at results, and never uses the locked test:

1. Each model's representative is its best member by validation pooled skill
   over the whole run (its accepted incumbent if it has one).
2. A paired bootstrap on the matched validation cells compares the two.
3. The model whose representative is better with P(better) >= 0.9 wins.
   Otherwise it is a tie, and the winner is the one whose run accepted an
   incumbent over Pacific's baseline; if both or neither did, the higher
   validation skill, then the faster run.

The chosen model then goes to scripts/final.py for the one locked-test run.
Writes results/model_comparison.json and .md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr.evaluate import paired_bootstrap  # noqa: E402


def load_run(run: Path) -> dict:
    rows = [json.loads(l) for l in (run / "archive.jsonl").read_text().splitlines() if l.strip()]
    members = [r for r in rows if r.get("kind") == "member"]
    gens = [r for r in rows if r.get("kind") == "generation"]
    base = next((r for r in rows if r.get("kind") == "baseline"), {})
    best_json = json.loads((run / "best.json").read_text()) if (run / "best.json").is_file() else None
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").is_file() else {}
    model = (members[0].get("rl") or {}).get("base_model") if members else None
    top = max(members, key=lambda r: r["val"]) if members else None
    cells = None
    if top is not None:
        path = run / f"gen{top['gen']:02d}" / top["mid"] / "cells.csv"
        if path.is_file():
            cells = pd.read_csv(path)
            cells = cells[cells["status"] == "scored"]
    rl_steps = sum(len(r.get("steps") or []) for r in members)
    return {"run": str(run), "model": model, "baseline_val": base.get("val"), "generations": len(gens),
            "members": len(members), "best_member": top and {k: top[k] for k in ("gen", "mid", "note", "val")},
            "accepted": bool(best_json and best_json.get("checkpoint")), "incumbent": best_json,
            "stop_reason": summary.get("stop_reason", "running"), "cells": cells,
            "val_by_generation": [max((m["val"] for m in members if m["gen"] == g["gen"]), default=None) for g in gens],
            "rl_steps": rl_steps}


def main() -> None:
    runs = [load_run(Path(p)) for p in sys.argv[1:3]]
    a, b = runs
    if a["cells"] is None or b["cells"] is None:
        raise SystemExit("both runs need at least one evaluated member")
    test = paired_bootstrap(b["cells"], a["cells"])          # P(b better than a)
    if test["p_better"] >= 0.9:
        winner, reason = b, f"better on validation with P = {test['p_better']:.2f}"
    elif test["p_better"] <= 0.1:
        winner, reason = a, f"better on validation with P = {1 - test['p_better']:.2f}"
    else:
        tie = f"tie on validation (P(B better) = {test['p_better']:.2f})"
        if a["accepted"] != b["accepted"]:
            winner = a if a["accepted"] else b
            reason = f"{tie}; only this run accepted an incumbent over Pacific's baseline"
        else:
            winner = a if a["best_member"]["val"] >= b["best_member"]["val"] else b
            reason = f"{tie}; higher validation pooled skill"
    out = {"runs": [{k: v for k, v in r.items() if k != "cells"} for r in runs], "bootstrap_b_vs_a": test,
           "winner": winner["model"], "reason": reason}
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "model_comparison.json").write_text(json.dumps(out, indent=1, default=str))
    lines = ["# Base model comparison (validation only)", "",
             "| Model | Generations | Best validation pooled | Accepted an incumbent | Stop | Baseline on the same cells |",
             "|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| `{r['model']}` | {r['generations']} | {r['best_member']['val']:+.4f} ({r['best_member']['mid']}, gen "
                     f"{r['best_member']['gen']}) | {'yes' if r['accepted'] else 'no'} | {r['stop_reason']} | "
                     f"{r['baseline_val']:+.4f} |")
    lines += ["", f"Paired bootstrap on matched validation cells: delta {test['delta']:+.4f} for `{b['model']}` over "
              f"`{a['model']}`, P(better) {test['p_better']:.2f}, {test['cells']} cells.", "",
              f"**Chosen: `{winner['model']}`**: {reason}."]
    (res / "model_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
