"""Optional decoder-only Transformers backend with temporary forward hooks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from epistemic_geometry.backends.base import (
    ModelBackend,
    OptionalDependencyError,
    validate_vector_dimension,
)
from epistemic_geometry.config import BackendConfig
from epistemic_geometry.types import BackendOutput, BenchmarkItem, Intervention


class HuggingFaceBackend(ModelBackend):
    """Inference-only backend for common decoder-only Transformers models.

    Imports and model loading are intentionally delayed until this class is
    constructed. The generic package therefore remains usable without Torch or
    Transformers installed, and no model is downloaded by ``ceg doctor``.
    """

    def __init__(self, config: BackendConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "HuggingFace mode requires torch and transformers. Install with "
                "pip install -e '.[hf]' after confirming the appropriate Torch build."
            ) from exc

        model_name = config.model_path or config.model_id
        if not model_name:
            raise ValueError(
                "backend.model_id or backend.model_path is required for huggingface mode"
            )
        self.config = config
        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"trust_remote_code": False}
        dtype = self._resolve_dtype(config.dtype)
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if config.device_map is not None:
            load_kwargs["device_map"] = config.device_map
        elif config.device == "auto" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory while loading the model. Try a smaller model, "
                    "an explicit bf16/fp16 dtype, or a device map with more available memory."
                ) from exc
            raise
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if "device_map" not in load_kwargs:
            target_device = self._resolve_device(config.device)
            self.model.to(target_device)
        self.device = next(self.model.parameters()).device
        self._layer_stack = self._locate_layer_stack(config.layer_path)
        config_hidden_size = getattr(self.model.config, "hidden_size", None)
        if config_hidden_size is None:
            config_hidden_size = getattr(self.model.config, "n_embd", None)
        if config_hidden_size is None:
            raise RuntimeError("Could not determine model hidden size from Transformers config")
        self._hidden_size = int(config_hidden_size)

    def _resolve_dtype(self, dtype_name: str) -> Any:
        torch = self.torch
        if dtype_name == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        choices = {"bf16": torch.bfloat16, "fp16": torch.float16, "float32": torch.float32}
        if dtype_name not in choices:
            raise ValueError("backend.dtype must be auto, bf16, fp16, or float32")
        if dtype_name in {"bf16", "fp16"} and not torch.cuda.is_available():
            raise ValueError(f"backend.dtype={dtype_name} requires a CUDA device in this setup")
        return choices[dtype_name]

    def _resolve_device(self, device_name: str) -> Any:
        if device_name == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return self.torch.device(device_name)

    def _resolve_path(self, root: Any, path: str) -> Any:
        current = root
        for part in path.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def _locate_layer_stack(self, explicit_path: str | None) -> Any:
        candidates = [
            explicit_path,
            "model.model.layers",
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "base_model.model.layers",
        ]
        failures: list[str] = []
        for path in candidates:
            if not path:
                continue
            try:
                stack = self._resolve_path(self, path)
            except (AttributeError, IndexError, TypeError) as exc:
                failures.append(f"{path}: {exc}")
                continue
            if hasattr(stack, "__len__") and len(stack) > 0:
                return stack
            failures.append(f"{path}: object is not a non-empty layer stack")
        detail = "; ".join(failures)
        raise RuntimeError(
            "Could not locate a transformer layer stack. Set backend.layer_path explicitly "
            f"(tried {detail})."
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def _tokenize(self, prompt: str) -> dict[str, Any]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def predict(self, item: BenchmarkItem) -> BackendOutput:
        encoded = self._tokenize(item.prompt)
        input_length = int(encoded["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**encoded, **generation_kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory during generation. Reduce max_new_tokens, use a "
                    "smaller batch/model, or choose an explicit supported dtype."
                ) from exc
            raise
        new_tokens = generated[0, input_length:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return BackendOutput(raw_output=raw_output, metadata={"model": self.model_name})

    def extract_activation(self, item: BenchmarkItem) -> np.ndarray:
        """Extract the last non-padding token at the configured layer."""

        encoded = self._tokenize(item.prompt)
        captured: list[Any] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, self.torch.Tensor):
                raise TypeError("Transformer layer output did not contain a Tensor")
            last_index = int(encoded["attention_mask"][0].sum().item()) - 1
            captured.append(hidden[:, last_index, :].detach().float().cpu())
            return output

        handle = self._layer_stack[self.config.layer].register_forward_hook(capture)
        try:
            with self.torch.inference_mode():
                self.model(**encoded, use_cache=False)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Activation hook did not capture a layer output")
        return captured[0][0].numpy().copy()

    def _hook_for(self, intervention: Intervention):
        validate_vector_dimension(intervention.vector, self)
        if intervention.layer >= len(self._layer_stack):
            raise ValueError(
                f"Intervention layer {intervention.layer} is outside layer stack of size "
                f"{len(self._layer_stack)}"
            )
        vector = self.torch.as_tensor(
            intervention.vector.values,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        ).view(1, 1, -1)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, self.torch.Tensor):
                hidden = output
                rest: tuple[Any, ...] | None = None
            elif (
                isinstance(output, (tuple, list))
                and output
                and isinstance(output[0], self.torch.Tensor)
            ):
                hidden = output[0]
                rest = tuple(output[1:])
            else:
                raise TypeError(
                    "Unsupported transformer layer output; expected Tensor or tuple[Tensor, ...]"
                )
            updated = hidden.clone()
            delta = intervention.alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
            if intervention.token_scope == "all_tokens":
                updated = updated + delta
            else:
                updated[:, -1:, :] = updated[:, -1:, :] + delta
            if rest is None:
                return updated
            if isinstance(output, tuple):
                return (updated, *rest)
            return [updated, *rest]

        return self._layer_stack[intervention.layer].register_forward_hook(hook)

    @contextmanager
    def steer(self, intervention: Intervention) -> Iterator[None]:
        """Install exactly one hook and remove it even if generation fails."""

        handle = self._hook_for(intervention)
        try:
            yield
        finally:
            handle.remove()
