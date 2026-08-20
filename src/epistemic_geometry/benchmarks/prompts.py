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
    enable_thinking: bool | None = None,
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
        system_prompt = item.metadata.get("system_prompt")
        messages = []
        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise ValueError("metadata.system_prompt must be a non-empty string")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": item.prompt})
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError as exc:
            if enable_thinking is not None:
                raise ValueError(
                    "The configured tokenizer does not accept the explicit "
                    "enable_thinking chat-template argument"
                ) from exc
            raise
        if not isinstance(text, str) or not text:
            raise ValueError("Tokenizer chat template returned no rendered prompt")
    return RenderedPrompt(text=text, mode=mode, hash=stable_digest("rendered_prompt", text))
