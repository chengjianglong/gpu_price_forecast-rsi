"""The world-model interface every backend implements.

One ``step`` is one agent decision: the backend holds the action for its
configured duration and returns every frame it generated in that time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WorldModel(ABC):
    name = "world_model"
    fps: float = 16.0

    @abstractmethod
    def reset(self, image: np.ndarray, prompt: str) -> list[np.ndarray]:
        """Start a new world from ``image``; return the frames shown so far
        (at least the first frame)."""

    @abstractmethod
    def step(self, action: str) -> list[np.ndarray]:
        """Apply one PlayWorld action (``W``, ``LEFT``, ``WAIT``, ...) and return
        the new frames, in order."""

    def finish(self) -> None:
        """Called once a rollout ends (agent END or step budget)."""

    def close(self) -> None:
        """Release models and GPU memory."""
