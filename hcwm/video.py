"""Frame and video helpers shared by the runner, agent and calibration."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_jpg(frame: np.ndarray, path: str | Path, quality: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(path, quality=quality)
    return path


def jpg_b64(frame: np.ndarray, max_side: int = 512, quality: int = 75) -> str:
    image = Image.fromarray(frame.astype(np.uint8))
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_mp4(frames: Sequence[np.ndarray], path: str | Path, fps: float) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"]
    ) as writer:
        for frame in frames:
            writer.append_data(_even(frame))
    return path


def _even(frame: np.ndarray) -> np.ndarray:
    """libx264 with yuv420p needs even width and height."""
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def sample_evenly(frames: Sequence[np.ndarray], count: int) -> list[np.ndarray]:
    if not frames or count <= 0:
        return []
    if len(frames) <= count:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[i] for i in idx]
