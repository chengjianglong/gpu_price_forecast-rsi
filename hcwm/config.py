"""YAML run configs: which backend, its arguments, and rollout settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .backends import make_backend
from .backends.base import WorldModel
from .runner import RolloutConfig


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg.setdefault("backend", {})
    cfg.setdefault("rollout", {})
    for key, value in (overrides or {}).items():
        cfg["backend"].setdefault("args", {})[key] = value
    return cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATH_ARGS = ("repo_dir", "ckpt_dir", "ref_dir")


def build_world(cfg: dict[str, Any]) -> WorldModel:
    backend = cfg["backend"]
    args = dict(backend.get("args") or {})
    for key in _PATH_ARGS:  # config paths are relative to the project root
        if args.get(key) and not Path(args[key]).is_absolute():
            args[key] = str(PROJECT_ROOT / args[key])
    return make_backend(backend["kind"], **args)


def build_rollout(cfg: dict[str, Any]) -> RolloutConfig:
    return RolloutConfig(**(cfg.get("rollout") or {}))
