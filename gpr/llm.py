"""The language-model layer the agents call.

Every agent call is a chat (a list of messages) answered by ``n`` samples.
Each sample keeps its prompt and completion tokens and the sampler's
log-probabilities, which the RL step needs later.

- ``FakeLLM`` answers with a Python function: offline tests and dry runs.
- ``TinkerLLM`` samples from a Tinker base model or a saved LoRA checkpoint.
  All requests in one ``generate`` call are submitted together and collected
  afterwards, so a batch costs about one round trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

Messages = list[dict[str, str]]

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Sample:
    text: str
    prompt_tokens: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str = ""


class LLM:
    name = "llm"

    def generate(self, chats: Sequence[Messages], n: int = 1, temperature: float = 0.7,
                 max_tokens: int = 512) -> list[list[Sample]]:
        raise NotImplementedError


class FakeLLM(LLM):
    """Answers each chat with ``respond(messages, sample_index)``."""

    name = "fake"

    def __init__(self, respond: Callable[[Messages, int], str]):
        self.respond = respond
        self.calls: list[Messages] = []

    def generate(self, chats, n=1, temperature=0.7, max_tokens=512):
        out = []
        for chat in chats:
            self.calls.append(chat)
            out.append([Sample(text=self.respond(chat, i)) for i in range(n)])
        return out


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=value lines from .env without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


class TinkerLLM(LLM):
    name = "tinker"

    def __init__(self, base_model: str, model_path: str | None = None, service_client: Any | None = None,
                 chat_template_kwargs: dict[str, Any] | None = None):
        import tinker

        load_env()
        self.tinker = tinker
        self.base_model = base_model
        self.model_path = model_path
        self.service = service_client or tinker.ServiceClient()
        future = self.service.create_sampling_client(model_path=model_path,
                                                     base_model=None if model_path else base_model)
        self.client = future.result() if hasattr(future, "result") else future
        self.tokenizer = self.client.get_tokenizer()
        # Qwen3-style templates accept enable_thinking; short JSON answers don't need thinking.
        self.chat_template_kwargs = {"enable_thinking": False, **(chat_template_kwargs or {})}

    def encode_chat(self, chat: Messages) -> list[int]:
        try:
            text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                      **self.chat_template_kwargs)
        except TypeError:  # template without the extra keyword
            text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, chats, n=1, temperature=0.7, max_tokens=512):
        types = self.tinker.types
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        prompts = [self.encode_chat(chat) for chat in chats]
        futures = [self.client.sample(prompt=types.ModelInput.from_ints(p), num_samples=n, sampling_params=params)
                   for p in prompts]
        out = []
        for prompt, future in zip(prompts, futures):
            response = future.result()
            samples = []
            for seq in response.sequences:
                tokens = list(seq.tokens)
                samples.append(Sample(
                    text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    prompt_tokens=prompt,
                    tokens=tokens,
                    logprobs=list(seq.logprobs or []),
                    stop_reason=str(seq.stop_reason),
                ))
            out.append(samples)
        return out
