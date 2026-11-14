"""Measure what each action actually does to the camera.

Matrix-Game 3.0's low PlayWorld score comes largely from failing the
trajectory gate (rotation pass rate 18.1%), so the first job is to measure,
per action, how far the generated camera really turns and whether WAIT keeps
time moving. Camera motion is estimated from consecutive frames with phase
correlation: a global image shift ``dx`` maps to a yaw of ``atan(dx / f)``.
That is accurate for pure rotations and small steps, which is what the probe
runs; it is not a full visual-odometry pipeline.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .backends.base import WorldModel


def _gray(frame: np.ndarray, max_side: int = 256) -> np.ndarray:
    step = max(1, int(math.ceil(max(frame.shape[:2]) / max_side)))
    small = frame[::step, ::step].astype(np.float32)
    return small.mean(axis=2) if small.ndim == 3 else small


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return (dy, dx, scale) where content in ``a`` moved by (dy, dx) pixels
    in ``b``, in full-resolution pixels, and ``scale`` converts to them."""
    ga, gb = _gray(a), _gray(b)
    scale = a.shape[1] / ga.shape[1]
    window = np.outer(np.hanning(ga.shape[0]), np.hanning(ga.shape[1]))
    fa, fb = np.fft.fft2(ga * window), np.fft.fft2(gb * window)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape

    def subpixel(left: float, mid: float, right: float) -> float:
        # Foroosh et al. (2002): the phase-correlation peak is a sampled sinc,
        # so the offset comes from the ratio of the larger neighbour to the peak.
        side, neighbour = (1.0, right) if right >= left else (-1.0, left)
        if mid <= 0 or neighbour <= 0:
            return 0.0
        return side * neighbour / (neighbour + mid)

    fy = py + subpixel(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    fx = px + subpixel(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    dy = fy - h if fy > h / 2 else fy
    dx = fx - w if fx > w / 2 else fx
    return float(dy * scale), float(dx * scale), scale


def camera_motion(frames: Sequence[np.ndarray], hfov_deg: float = 90.0) -> dict[str, float]:
    """Total yaw/pitch (degrees) and mean pixel change across ``frames``."""
    if len(frames) < 2:
        return {"yaw_deg": 0.0, "pitch_deg": 0.0, "mean_abs_change": 0.0}
    width = frames[0].shape[1]
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    yaw = pitch = 0.0
    changes = []
    for a, b in zip(frames[:-1], frames[1:]):
        dy, dx, _ = phase_shift(a, b)
        yaw += -math.degrees(math.atan(dx / focal))   # turning right moves content left
        pitch += math.degrees(math.atan(dy / focal))  # tilting up moves content down
        changes.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()))
    return {"yaw_deg": yaw, "pitch_deg": pitch, "mean_abs_change": float(np.mean(changes))}


@dataclass
class ActionProbe:
    action: str
    steps: int
    frames: int
    yaw_deg_per_step: float
    pitch_deg_per_step: float
    mean_abs_change: float


def probe_actions(
    world: WorldModel,
    image: np.ndarray,
    prompt: str,
    actions: Sequence[str] = ("LEFT", "RIGHT", "UP", "DOWN", "W", "WAIT"),
    steps: int = 3,
    hfov_deg: float = 90.0,
) -> list[ActionProbe]:
    """Run each action from a fresh world and measure the camera motion."""
    probes = []
    for action in actions:
        world.reset(image, prompt)
        frames: list[np.ndarray] = []
        for _ in range(steps):
            frames += world.step(action)
        world.finish()
        motion = camera_motion(frames, hfov_deg)
        probes.append(ActionProbe(
            action=action, steps=steps, frames=len(frames),
            yaw_deg_per_step=motion["yaw_deg"] / steps,
            pitch_deg_per_step=motion["pitch_deg"] / steps,
            mean_abs_change=motion["mean_abs_change"],
        ))
    return probes


def suggest_scales(probes: Sequence[ActionProbe], target_deg_per_step: float) -> dict[str, float | None]:
    """Gain to apply to each camera axis so one step turns ``target_deg_per_step``."""
    by_action = {p.action: p for p in probes}

    def gain(names: tuple[str, str], field: str) -> float | None:
        measured = [abs(getattr(by_action[n], field)) for n in names if n in by_action]
        mean = float(np.mean(measured)) if measured else 0.0
        return round(target_deg_per_step / mean, 3) if mean > 1e-3 else None

    return {
        "yaw_scale": gain(("LEFT", "RIGHT"), "yaw_deg_per_step"),
        "pitch_scale": gain(("UP", "DOWN"), "pitch_deg_per_step"),
    }


def probes_to_rows(probes: Sequence[ActionProbe]) -> list[dict]:
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(p).items()} for p in probes]
