"""World-model backends. Heavy backends import torch and their repos lazily."""

from __future__ import annotations

from typing import Any

from .base import WorldModel


def make_backend(kind: str, **kwargs: Any) -> WorldModel:
    if kind == "mock":
        from .mock import MockWorld

        return MockWorld(**kwargs)
    if kind == "abot":
        from .abot import ABotWorld

        return ABotWorld(**kwargs)
    if kind == "matrixgame3":
        from .matrixgame3 import MatrixGame3World

        return MatrixGame3World(**kwargs)
    raise ValueError(f"Unknown backend: {kind}")


__all__ = ["WorldModel", "make_backend"]
