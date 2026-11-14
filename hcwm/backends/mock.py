"""A CPU stand-in for a world model, with known ground truth.

It moves a virtual camera over the input image: LEFT/RIGHT pan, UP/DOWN tilt,
W/S zoom in/out, A/D slide sideways, WAIT holds still. Because the true yaw is
known, it is what the calibration tests check against. It is not a world
model: nothing new is ever generated, so it says nothing about PlayWorld
scores.
"""

from __future__ import annotations

import math

import numpy as np

from .base import WorldModel


class MockWorld(WorldModel):
    name = "mock"

    def __init__(
        self,
        fps: float = 12.0,
        step_seconds: float = 1.0,
        out_size: tuple[int, int] = (360, 640),
        hfov_deg: float = 90.0,
        turn_deg_per_s: float = 30.0,
        tilt_deg_per_s: float = 15.0,
        zoom_per_s: float = 0.15,
        slide_frac_per_s: float = 0.10,
        yaw_gain: float = 1.0,
    ):
        self.fps = fps
        self.frames_per_step = max(1, round(fps * step_seconds))
        self.out_h, self.out_w = out_size
        self.hfov_deg = hfov_deg
        # yaw_gain < 1 simulates a model that under-turns, for calibration tests.
        self.rates = {
            "LEFT": ("yaw", -turn_deg_per_s * yaw_gain),
            "RIGHT": ("yaw", turn_deg_per_s * yaw_gain),
            "UP": ("pitch", tilt_deg_per_s),
            "DOWN": ("pitch", -tilt_deg_per_s),
            "W": ("zoom", zoom_per_s),
            "S": ("zoom", -zoom_per_s),
            "A": ("slide", -slide_frac_per_s),
            "D": ("slide", slide_frac_per_s),
        }
        self.canvas: np.ndarray | None = None
        self.state = {"yaw": 0.0, "pitch": 0.0, "zoom": 0.0, "slide": 0.0}

    @property
    def focal_px(self) -> float:
        return (self.out_w / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    def reset(self, image: np.ndarray, prompt: str) -> list[np.ndarray]:
        image = _resize(image, self.out_h, self.out_w)
        # Pad so the camera can look around before it runs out of picture.
        pad_h, pad_w = self.out_h, self.out_w * 2
        self.canvas = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
        self.state = {key: 0.0 for key in self.state}
        return [self._render()]

    def step(self, action: str) -> list[np.ndarray]:
        frames = []
        kind, rate = self.rates.get(action, (None, 0.0))
        for _ in range(self.frames_per_step):
            if kind:
                self.state[kind] += rate / self.fps
            frames.append(self._render())
        return frames

    def _render(self) -> np.ndarray:
        assert self.canvas is not None, "call reset() first"
        canvas_h, canvas_w = self.canvas.shape[:2]
        f = self.focal_px
        scale = math.exp(-self.state["zoom"])  # zooming in shows a smaller window
        win_h, win_w = self.out_h * scale, self.out_w * scale
        # A rotating camera shifts the view by about f pixels per radian near the
        # centre; keep that rate constant so motion matches a rotation, not a slide.
        cx = canvas_w / 2 + f * math.radians(self.state["yaw"]) + self.state["slide"] * self.out_w
        cy = canvas_h / 2 - f * math.radians(self.state["pitch"])
        ys = np.clip((cy - win_h / 2 + (np.arange(self.out_h) + 0.5) * scale).astype(int), 0, canvas_h - 1)
        xs = np.clip((cx - win_w / 2 + (np.arange(self.out_w) + 0.5) * scale).astype(int), 0, canvas_w - 1)
        return self.canvas[ys][:, xs]


def _resize(image: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, image.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, image.shape[1] - 1, w).astype(int)
    return image[ys][:, xs]
