"""Matrix-Game 3.0 (SkyworkAI/Matrix-Game, ``Matrix-Game-3/``) driven step by step.

Its interactive pipeline runs one long ``generate`` loop that asks for the
next action on stdin (``get_current_action``) before every segment and writes
each decoded segment with ``process_video``, which draws key and mouse icons
on the frames. We run that loop in a worker thread and replace both functions
in the module: actions come from the agent, and frames come back clean,
without the overlay, so neither the agent nor the judge sees it.

The first segment is 57 frames, later ones 40, at 17 FPS. Camera control is a
per-frame [pitch, yaw] value; the repo uses 0.1 per frame, which is the knob
``scripts/calibrate_actions.py`` tunes.

Single GPU only for now: with several GPUs, actions are read on rank 0 and
broadcast, so this thread-based adapter would need a torchrun entry point.
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import WorldModel

KEYBOARD = {
    "W": [1, 0, 0, 0, 0, 0], "S": [0, 1, 0, 0, 0, 0], "A": [0, 0, 1, 0, 0, 0], "D": [0, 0, 0, 1, 0, 0],
}
NO_KEY = [0, 0, 0, 0, 0, 0]


class _StopRollout(Exception):
    pass


class MatrixGame3World(WorldModel):
    name = "matrixgame3"
    fps = 17.0

    def __init__(
        self,
        repo_dir: str,
        ckpt_dir: str,
        max_steps: int = 40,
        cam_value: float = 0.1,
        pitch_scale: float = 1.0,
        yaw_scale: float = 1.0,
        size: str = "704*1280",
        num_inference_steps: int = 3,
        seed: int = 42,
        extra_args: list[str] | None = None,
        step_timeout_s: float = 900.0,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        if not (self.repo_dir / "pipeline" / "inference_interactive_pipeline.py").is_file():
            raise FileNotFoundError(
                f"Matrix-Game-3 checkout not found at {self.repo_dir}; run scripts/setup_third_party.sh"
            )
        self.ckpt_dir = str(Path(ckpt_dir).resolve())
        self.max_steps = max_steps
        self.pitch = cam_value * pitch_scale
        self.yaw = cam_value * yaw_scale
        self.size = size
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.extra_args = list(extra_args or [])
        self.step_timeout_s = step_timeout_s
        self._tmp = tempfile.TemporaryDirectory(prefix="mg3_")
        self._prev_cwd = os.getcwd()
        self._thread: threading.Thread | None = None
        self._load()

    # ---- loading -----------------------------------------------------------
    def _parse_args(self) -> Any:
        placeholder = Path(self._tmp.name) / "placeholder.png"
        Image.new("RGB", (64, 64)).save(placeholder)
        argv = [
            "generate.py", "--size", self.size, "--ckpt_dir", self.ckpt_dir,
            "--image", str(placeholder), "--prompt", "placeholder", "--interactive",
            "--num_iterations", str(self.max_steps), "--num_inference_steps", str(self.num_inference_steps),
            "--fa_version", "3", "--use_int8", "--vae_type", "mg_lightvae", "--lightvae_pruning_rate", "0.5",
            "--output_dir", self._tmp.name, "--save_name", "rollout", "--seed", str(self.seed),
            *self.extra_args,
        ]
        saved = sys.argv
        try:
            sys.argv = argv
            return self.mg._parse_args()
        finally:
            sys.argv = saved

    def _load(self) -> None:
        os.chdir(self.repo_dir)  # its configs and assets use repo-relative paths
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        import torch
        import generate as mg
        import pipeline.inference_interactive_pipeline as mgi

        self.torch, self.mg, self.mgi = torch, mg, mgi
        self.args = self._parse_args()
        if getattr(self.args, "use_async_vae", False):
            raise ValueError("--use_async_vae is not supported: frames must come back after every segment")
        a = self.args
        self.pipe = mg.MatrixGame3InteractivePipeline(
            config=mg.WAN_CONFIGS["matrix_game3"], checkpoint_dir=a.ckpt_dir, device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=a.t5_cpu,
            convert_model_dtype=a.convert_model_dtype, args=a, fa_version=a.fa_version,
            use_base_model=a.use_base_model,
        )
        mgi.get_current_action = self._next_action
        mgi.process_video = self._capture_segment

    # ---- hooks called from inside Matrix-Game's loop -------------------------
    def _next_action(self) -> dict[str, Any]:
        action = self._actions.get()
        if action is None:
            raise _StopRollout()
        keyboard = KEYBOARD.get(action, NO_KEY)
        mouse = {
            "UP": [self.pitch, 0.0], "DOWN": [-self.pitch, 0.0],
            "LEFT": [0.0, -self.yaw], "RIGHT": [0.0, self.yaw],
        }.get(action, [0.0, 0.0])
        return {"mouse": self.torch.tensor(mouse), "keyboard": self.torch.tensor(keyboard)}

    def _capture_segment(self, input_video, output_video, *args: Any, **kwargs: Any) -> None:
        # Called once per segment ("..._current_iteration_k.mp4") and once at the
        # end for the whole video; only the per-segment calls are needed.
        if "_current_iteration_" in str(output_video):
            self._frames.put([np.asarray(f, dtype=np.uint8) for f in input_video])

    def _worker(self, image: Image.Image, prompt: str) -> None:
        a = self.args
        try:
            self.pipe.generate(
                prompt, image, max_area=self.mg.MAX_AREA_CONFIGS[a.size], shift=a.sample_shift,
                num_inference_steps=a.num_inference_steps, guide_scale=a.sample_guide_scale,
                seed=a.seed, use_base_model=a.use_base_model, args=a,
            )
        except _StopRollout:
            pass
        except BaseException as error:  # surfaced to the caller by step()
            self._error = error
        finally:
            self._frames.put(None)

    # ---- WorldModel ------------------------------------------------------------
    def reset(self, image: np.ndarray, prompt: str) -> list[np.ndarray]:
        self.finish()
        self._actions: queue.Queue = queue.Queue()
        self._frames: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        pil = Image.fromarray(image).convert("RGB")
        self._thread = threading.Thread(target=self._worker, args=(pil, prompt or "A first-person game scene."), daemon=True)
        self._thread.start()
        h, w = (int(v) for v in self.size.split("*"))
        return [np.asarray(pil.resize((w, h)))]

    def step(self, action: str) -> list[np.ndarray]:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Matrix-Game rollout is not running (step budget used up, or it failed)")
        self._actions.put(action)
        frames = self._frames.get(timeout=self.step_timeout_s)
        if frames is None:
            raise RuntimeError(f"Matrix-Game rollout ended early: {self._error!r}")
        return frames

    def finish(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._actions.put(None)
            self._thread.join(timeout=self.step_timeout_s)
        self._thread = None

    def close(self) -> None:
        self.finish()
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()
