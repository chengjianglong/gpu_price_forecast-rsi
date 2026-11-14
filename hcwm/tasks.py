"""Load PlayWorld cases.

The full dataset lays out ``<root>/{gc,if,insight,outsight}/data.json``; the
bundled examples use ``<root>/{GC,IF}/NNN.json``. Both are JSON arrays of
task records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("gc", "if", "insight", "outsight")
GROUP_OF_SPLIT = {"gc": "GC", "if": "IF", "insight": "OE", "outsight": "OE"}


def _resolve_image(record: dict[str, Any], json_path: Path, data_root: Path) -> Path:
    raw = Path(str(record["image_path"]))
    if raw.is_absolute():
        return raw
    for candidate in (data_root / raw, json_path.parent / raw, json_path.parent / raw.name):
        if candidate.is_file():
            return candidate
    return data_root / raw


def load_json_tasks(json_path: Path, data_root: Path, split: str | None = None) -> list[dict[str, Any]]:
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    tasks = []
    for record in records:
        task = dict(record)
        task_split = (split or str(task.get("category", "")).lower() or "gc").lower()
        if task_split == "oe":
            task_split = "outsight"
        task["_split"] = task_split
        task["_group"] = GROUP_OF_SPLIT.get(task_split, task_split.upper())
        task["image_path"] = str(_resolve_image(task, Path(json_path), Path(data_root)))
        task["_dataset"] = str(Path(json_path).resolve())
        tasks.append(task)
    return tasks


def load_tasks(
    data_root: Path,
    splits: Iterable[str] = SPLITS,
    task_ids: Iterable[str] = (),
    dataset: Path | None = None,
) -> list[dict[str, Any]]:
    """Load tasks from one ``dataset`` file, or every split under ``data_root``."""
    data_root = Path(data_root)
    tasks: list[dict[str, Any]] = []
    if dataset is not None:
        tasks = load_json_tasks(Path(dataset), data_root)
    else:
        for split in splits:
            path = data_root / split / "data.json"
            if path.is_file():
                tasks += load_json_tasks(path, data_root, split)
    wanted = list(task_ids)
    if wanted:
        by_id = {t["task_id"]: t for t in tasks}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            raise KeyError(f"Task IDs not found: {', '.join(missing)}")
        tasks = [by_id[t] for t in wanted]
    return tasks
