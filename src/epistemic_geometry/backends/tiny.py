"""Network-free randomly initialized Transformer fixture.

This module exists to exercise real Torch/Transformers mechanics. It has no
language capability and its outputs are not scientific evidence.
"""

from __future__ import annotations

from typing import Any

from epistemic_geometry.backends.huggingface import HuggingFaceBackend
from epistemic_geometry.config import BackendConfig
from epistemic_geometry.reproducibility import stable_seed


class TinyWhitespaceTokenizer:
    """Small deterministic tokenizer sufficient for CPU integration tests."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.name_or_path = "tiny-whitespace-tokenizer"
        self.padding_side = "left"

    def _encode_token(self, token: str) -> int:
        return 3 + stable_seed("tiny-token", token) % (self.vocab_size - 3)

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
        padding: bool = False,
    ) -> dict[str, Any]:
        if return_tensors != "pt":
            raise ValueError("TinyWhitespaceTokenizer only supports return_tensors='pt'")
        import torch

        texts = [text] if isinstance(text, str) else text
        token_rows = []
        for value in texts:
            tokens = [self.bos_token_id] if add_special_tokens else []
            tokens.extend(self._encode_token(token) for token in value.split())
            if add_special_tokens:
                tokens.append(self.eos_token_id)
            token_rows.append(tokens)
        if len(token_rows) > 1 and not padding:
            raise ValueError("TinyWhitespaceTokenizer requires padding=True for batches")
        width = max(len(tokens) for tokens in token_rows)
        padded_rows = []
        masks = []
        for tokens in token_rows:
            pad_count = width - len(tokens)
            if self.padding_side == "left":
                padded_rows.append([self.pad_token_id] * pad_count + tokens)
                masks.append([0] * pad_count + [1] * len(tokens))
            else:
                padded_rows.append(tokens + [self.pad_token_id] * pad_count)
                masks.append([1] * len(tokens) + [0] * pad_count)
        return {
            "input_ids": torch.tensor(padded_rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        labels = {3: "A", 4: "B", 5: "C", 6: "D"}
        decoded: list[str] = []
        for value in values:
            value = int(value)
            if skip_special_tokens and value in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }:
                continue
            decoded.append(labels.get(value, "X"))
        return " ".join(decoded)


class TinyRandomTransformerBackend(HuggingFaceBackend):
    """Two-layer randomly initialized GPT-2-style decoder for software tests."""

    def __init__(self, config: BackendConfig, seed: int = 0) -> None:
        try:
            import torch
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as exc:
            raise RuntimeError(
                "tiny_transformer mode requires torch and transformers; "
                "install the optional HF stack"
            ) from exc
        torch.manual_seed(seed)
        model_config = GPT2Config(
            vocab_size=64,
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=False,
        )
        model = GPT2LMHeadModel(model_config)
        tokenizer = TinyWhitespaceTokenizer(vocab_size=model_config.vocab_size)
        injected_config = config
        if injected_config.hidden_size != model_config.n_embd:
            raise ValueError(
                f"tiny_transformer hidden_size must be {model_config.n_embd}, "
                f"got {injected_config.hidden_size}"
            )
        super().__init__(
            injected_config,
            model=model,
            tokenizer=tokenizer,
            model_identifier="tiny-random-gpt2-config",
            tokenizer_identifier="tiny-whitespace-tokenizer",
            model_revision="local-config",
        )
