"""Explicit prompt rendering for plain-text and chat-template backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from epistemic_geometry.reproducibility import stable_digest
from epistemic_geometry.types import BenchmarkItem


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    mode: str
    hash: str


def render_prompt(
    item: BenchmarkItem,
    mode: str = "plain",
    tokenizer: Any | None = None,
) -> RenderedPrompt:
    """Render task content without silently guessing a chat format."""

    if mode not in {"plain", "chat"}:
        raise ValueError("prompt mode must be plain or chat")
    if mode == "plain":
        text = item.prompt
    else:
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "prompt_mode=chat requires a tokenizer with apply_chat_template; "
                "choose prompt_mode=plain for a base model"
            )
        messages = [{"role": "user", "content": item.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(text, str) or not text:
            raise ValueError("Tokenizer chat template returned no rendered prompt")
    return RenderedPrompt(text=text, mode=mode, hash=stable_digest("rendered_prompt", text))

