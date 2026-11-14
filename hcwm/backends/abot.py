"""ABot-World-0 (amap-cvlab/ABot-World) driven block by block.

Uses the same calls as the repo's Gradio worker (``web_client/inference.py``):
``set_prompts`` -> ``set_ref_latent_mask_from_exists_paths`` -> ``reset_stream``
-> ``set_first_frame_latent``, then per block ``set_act`` + ``generate_next_block``
and ``decode_block_to_frames``. One block decodes to 12 frames at 12 FPS
(the first block to 9).

Its key state is eight booleans in the order W A S D I J K L. W/A/S/D move;
I/J/K/L look (I up, J left, K down, L right), the same convention as
Matrix-Game 3.0. That arrow mapping is an assumption until
``scripts/calibrate_actions.py`` confirms it on real rollouts.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .base import WorldModel

KEY_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")
DEFAULT_KEY_MAP = {
    "W": ["W"], "A": ["A"], "S": ["S"], "D": ["D"],
    "UP": ["I"], "LEFT": ["J"], "DOWN": ["K"], "RIGHT": ["L"],
    "WAIT": [],
}


class ABotWorld(WorldModel):
    name = "abot_world0"

    def __init__(
        self,
        repo_dir: str,
        blocks_per_step: int = 2,
        key_map: dict[str, list[str]] | None = None,
        vae_type: str | None = None,
        use_fp8_gemm: bool = True,
        ref_dir: str | None = None,
        fps: float = 12.0,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        if not (self.repo_dir / "web_client" / "pipeline_loader.py").is_file():
            raise FileNotFoundError(f"ABot-World checkout not found at {self.repo_dir}; run scripts/setup_third_party.sh")
        self.blocks_per_step = blocks_per_step
        self.key_map = {**DEFAULT_KEY_MAP, **(key_map or {})}
        self.vae_type = vae_type
        self.use_fp8_gemm = use_fp8_gemm
        self.ref_dir = ref_dir
        self.fps = fps
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory(prefix="abot_")
        self._load()

    def _load(self) -> None:
        # ABot's configs use paths relative to the repo root.
        os.chdir(self.repo_dir)
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        import torch
        from web_client import config as abot_config
        from web_client.pipeline_loader import decode_block_to_frames, get_pipeline

        self.torch = torch
        self.cfg = abot_config
        self._decode = decode_block_to_frames
        self.pipeline, self.config, self.device = get_pipeline(
            vae_type=self.vae_type, use_fp8_gemm=self.use_fp8_gemm
        )
        self.pipeline.torch_dtype = torch.bfloat16
        num_fpb = int(getattr(self.config, "num_frame_per_block", 1))
        vae = self.pipeline.encoder if self.pipeline.encoder is not None else self.pipeline.vae
        up = getattr(vae, "upsampling_factor", 16)
        self.num_fpb = num_fpb
        self.noise_shape = (1, num_fpb, vae.z_dim, abot_config.STREAM_HEIGHT // up, abot_config.STREAM_WIDTH // up)

    def reset(self, image: np.ndarray, prompt: str) -> list[np.ndarray]:
        torch, cfg = self.torch, self.cfg
        first = Path(self._tmp.name) / "first_frame.png"
        Image.fromarray(image).save(first)
        self.pipeline.set_prompts([prompt], device=self.device)
        # Without the five pre-generated reference views (head/left/right/front/back)
        # ABot falls back to zero reference latents, which it supports.
        ref_dir = self.ref_dir or str(Path(self._tmp.name) / "no_refs")
        self.pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=ref_dir, device=self.device)
        self.pipeline.reset_stream(batch_size=1, dtype=torch.bfloat16, device=self.device, initial_latent=None)
        self.pipeline.set_first_frame_latent(
            str(first), height=cfg.STREAM_HEIGHT, width=cfg.STREAM_WIDTH, device=self.device
        )
        self.noise = torch.randn(self.noise_shape, device=self.device, dtype=torch.bfloat16)
        return [np.asarray(Image.open(first).convert("RGB").resize((cfg.STREAM_WIDTH, cfg.STREAM_HEIGHT)))]

    def _keys(self, action: str) -> dict[str, bool]:
        pressed = set(self.key_map.get(action, []))
        return {key: key in pressed for key in KEY_ORDER}

    def step(self, action: str) -> list[np.ndarray]:
        torch, cfg = self.torch, self.cfg
        keys = self._keys(action)
        frames: list[np.ndarray] = []
        for _ in range(self.blocks_per_step):
            latent = None
            while latent is None:  # the pipeline returns None while it warms its cache
                self.pipeline.set_act(
                    keys, height=cfg.STREAM_HEIGHT, width=cfg.STREAM_WIDTH, num_frames=self.num_fpb, device=self.device
                )
                latent = self.pipeline.generate_next_block(self.noise)
                self.noise = torch.randn_like(self.noise)
            frames += [np.asarray(f) for f in self._decode(self.pipeline, latent)]
        return frames

    def close(self) -> None:
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()
