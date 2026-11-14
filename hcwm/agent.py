"""The agent player: picks the next action from the frames generated so far.

``ClaudeAgent`` follows PlayWorld's local-model protocol (the SANA-WM adapter):
it sees the initial frame, a few frames from older steps and more from the
latest step, the objective, the action history and the reference plan's
recommendation, and answers with one action word. PlayWorld reports all runs
with Claude Haiku 4.5 as the player, so that is the default here; keep it
fixed when comparing against the leaderboard.

``ScriptedAgent`` replays the reference plan. It needs no API key and is the
"preset only" ablation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .actions import AGENT_ACTIONS, END, parse_agent_reply
from .video import jpg_b64, sample_evenly

DEFAULT_PLAYER_MODEL = "claude-haiku-4-5"


@dataclass
class AgentContext:
    task: dict[str, Any]
    step: int
    max_steps: int
    history: list[str]
    recommended: str
    plan_steps: int  # steps the reference plan covers; later steps have no recommendation
    initial_frame: np.ndarray
    step_frames: Sequence[Sequence[np.ndarray]] = field(default_factory=list)


class Agent:
    name = "agent"

    def decide(self, ctx: AgentContext) -> str:
        raise NotImplementedError


class ScriptedAgent(Agent):
    """Follow the reference plan and finish when it runs out."""

    name = "scripted"

    def __init__(self, plan_length: int | None = None):
        self.plan_length = plan_length

    def decide(self, ctx: AgentContext) -> str:
        limit = self.plan_length if self.plan_length is not None else ctx.plan_steps
        return END if ctx.step >= limit else ctx.recommended


def build_prompt(ctx: AgentContext, labels: list[str]) -> str:
    history = " -> ".join(ctx.history) if ctx.history else "none"
    if ctx.step < ctx.plan_steps:
        hint = (f"The reference plan suggests {ctx.recommended} for this step; follow it unless the "
                "frames show it is wrong.\n")
    else:
        hint = "The reference plan is complete; act only if the goal is not yet visibly achieved.\n"
    return (
        f"Our goal is: {ctx.task.get('prompt', '')}\n"
        f"We are now at step {ctx.step + 1} of at most {ctx.max_steps}. "
        f"The actions already taken are: {history}.\n"
        f"{hint}"
        f"The attached frames are chronological: {'; '.join(labels)}.\n"
        "Look at the actual visual progress in these frames and decide what action is needed now "
        "to accomplish the goal.\n"
        f"Choose one action from {', '.join(AGENT_ACTIONS)}. W/S move forward/back, A/D strafe "
        "left/right, LEFT/RIGHT turn the camera, UP/DOWN tilt it, WAIT keeps still while time passes. "
        "If the full goal is already visually complete, return END. "
        "Return exactly one action word and nothing else."
    )


class ClaudeAgent(Agent):
    name = "claude"

    def __init__(
        self,
        model: str | None = None,
        older_frames_per_step: int = 4,
        latest_frames: int = 8,
        max_images: int = 40,
        max_retries: int = 3,
    ):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("PLAYER_MODEL", DEFAULT_PLAYER_MODEL)
        self.older_frames_per_step = older_frames_per_step
        self.latest_frames = latest_frames
        self.max_images = max_images
        self.max_retries = max_retries

    def _images(self, ctx: AgentContext) -> tuple[list[str], list[str]]:
        images, labels = [jpg_b64(ctx.initial_frame)], ["initial frame"]
        steps = list(ctx.step_frames)
        older, latest = steps[:-1], steps[-1:] if steps else []
        # Keep the most recent older steps that fit under the image cap.
        budget = self.max_images - 1 - (self.latest_frames if latest else 0)
        keep = max(0, budget // max(1, self.older_frames_per_step))
        first_kept = len(older) - min(len(older), keep)
        for index in range(first_kept, len(older)):
            for frame in sample_evenly(older[index], self.older_frames_per_step):
                images.append(jpg_b64(frame))
                labels.append(f"step {index + 1}")
        for frame in sample_evenly(latest[0] if latest else [], self.latest_frames):
            images.append(jpg_b64(frame))
            labels.append(f"step {len(steps)} (latest)")
        return images, labels

    def decide(self, ctx: AgentContext) -> str:
        import anthropic

        images, labels = self._images(ctx)
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
            for b64 in images
        ]
        content.append({"type": "text", "text": build_prompt(ctx, labels)})
        allowed = AGENT_ACTIONS + (END,)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": content}],
                )
            except anthropic.RateLimitError as error:
                print(f"    [agent] rate limited (attempt {attempt}): {error}")
                continue
            except anthropic.APIStatusError as error:
                print(f"    [agent] API error {error.status_code} (attempt {attempt}): {error.message}")
                continue
            except anthropic.APIConnectionError as error:
                print(f"    [agent] connection error (attempt {attempt}): {error}")
                continue
            text = "".join(block.text for block in response.content if block.type == "text")
            action = parse_agent_reply(text, allowed)
            if action:
                return action
            print(f"    [agent] unparseable reply (attempt {attempt}): {text!r}")
        print(f"    [agent] falling back to the reference plan: {ctx.recommended}")
        return ctx.recommended


def make_agent(kind: str, **kwargs: Any) -> Agent:
    if kind == "claude":
        return ClaudeAgent(**kwargs)
    if kind == "scripted":
        return ScriptedAgent(**kwargs)
    raise ValueError(f"Unknown agent: {kind}")
