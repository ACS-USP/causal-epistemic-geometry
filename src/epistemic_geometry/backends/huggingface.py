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
from epistemic_geometry.benchmarks.prompts import render_prompt
from epistemic_geometry.config import BackendConfig
from epistemic_geometry.reproducibility import stable_digest
from epistemic_geometry.types import BackendOutput, BenchmarkItem, Intervention


class HuggingFaceBackend(ModelBackend):
    """Inference-only backend for common decoder-only Transformers models.

    Imports and model loading are intentionally delayed until this class is
    constructed. The generic package therefore remains usable without Torch or
    Transformers installed, and no model is downloaded by ``ceg doctor``.
    """

    def __init__(
        self,
        config: BackendConfig,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_identifier: str | None = None,
        tokenizer_identifier: str | None = None,
        model_revision: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError(
                "HuggingFace mode requires torch. Install the approved Torch build first."
            ) from exc

        if (model is None) != (tokenizer is None):
            raise ValueError("Injected HuggingFace tests must provide both model and tokenizer")

        model_name = config.model_path or config.model_id
        if model is None and not model_name:
            raise ValueError(
                "backend.model_id or backend.model_path is required for huggingface mode"
            )
        self.config = config
        self.torch = torch
        self.model_name = model_identifier or model_name or "injected-model"
        self.tokenizer_name = tokenizer_identifier or config.tokenizer_id or self.model_name
        self.model_revision = model_revision or config.model_revision
        self._injected_model = model is not None
        self._choice_prompt_index: int | None = None
        if model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise OptionalDependencyError(
                    "HuggingFace mode requires transformers. Install with "
                    "pip install -e '.[hf]' after confirming the appropriate Torch build."
                ) from exc
            tokenizer_source = config.tokenizer_id or model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                revision=config.tokenizer_revision or config.model_revision,
            )
        else:
            self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = 0

        load_kwargs: dict[str, Any] = {"trust_remote_code": False}
        if model is None:
            dtype = self._resolve_dtype(config.dtype)
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if config.device_map is not None:
                load_kwargs["device_map"] = config.device_map
            elif config.device == "auto" and torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    revision=config.model_revision,
                    **load_kwargs,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "CUDA out of memory while loading the model. Try a smaller model, "
                        "an explicit bf16/fp16 dtype, or a device map with more available memory."
                    ) from exc
                raise
        else:
            self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if model is None and "device_map" not in load_kwargs:
            target_device = self._resolve_device(config.device)
            self.model.to(target_device)
        elif model is not None and config.device not in {"auto", "injected"}:
            self.model.to(self._resolve_device(config.device))
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
        candidates = [explicit_path] if explicit_path else [
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
            for root_name, root in (("backend", self), ("model", self.model)):
                try:
                    stack = self._resolve_path(root, path)
                except (AttributeError, IndexError, TypeError) as exc:
                    failures.append(f"{root_name}.{path}: {exc}")
                    continue
                if hasattr(stack, "__len__") and len(stack) > 0:
                    self._resolved_layer_path = (
                        path if root_name == "backend" else f"model.{path}"
                    )
                    return stack
                failures.append(f"{root_name}.{path}: object is not a non-empty layer stack")
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

    def _encode_item(self, item: BenchmarkItem) -> tuple[dict[str, Any], str, str]:
        rendered = render_prompt(
            item,
            mode=self.config.prompt_mode,
            tokenizer=self.tokenizer,
            enable_thinking=self.config.enable_thinking,
        )
        encoded = self.tokenizer(rendered.text, return_tensors="pt")
        return (
            {key: value.to(self.device) for key, value in encoded.items()},
            rendered.text,
            rendered.hash,
        )

    def predict(self, item: BenchmarkItem) -> BackendOutput:
        if self.config.inference_mode == "choice_loglikelihood":
            return self._predict_choice_loglikelihood(item)
        encoded, _rendered_prompt, prompt_hash = self._encode_item(item)
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
        return BackendOutput(
            raw_output=raw_output,
            metadata={
                "model": self.model_name,
                "prompt_mode": self.config.prompt_mode,
                "rendered_prompt_hash": prompt_hash,
                "input_token_count": input_length,
                "generation": {
                    "do_sample": self.config.do_sample,
                    "temperature": self.config.temperature,
                    "max_new_tokens": self.config.max_new_tokens,
                },
            },
        )

    def _candidate_labels(self, item: BenchmarkItem) -> list[str]:
        labels = item.metadata.get("candidate_labels", self.config.candidate_labels)
        if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(f"Item {item.id} does not provide valid candidate labels")
        return labels

    def _text_token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        values = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        token_ids = [int(value) for value in values]
        if not token_ids:
            raise ValueError(f"Text produced no tokens: {text!r}")
        return token_ids

    def _predict_choice_loglikelihood(self, item: BenchmarkItem) -> BackendOutput:
        """Score complete candidate continuations without generation or sampling."""

        _encoded, rendered_prompt, prompt_hash = self._encode_item(item)
        prompt_ids = self._text_token_ids(rendered_prompt)
        labels = self._candidate_labels(item)
        scores: dict[str, float] = {}
        token_ids_by_label: dict[str, list[int]] = {}
        self._choice_prompt_index = len(prompt_ids) - 1
        try:
            with self.torch.inference_mode():
                prompt_tensor = self.torch.tensor(
                    [prompt_ids], dtype=self.torch.long, device=self.device
                )
                for label in labels:
                    candidate_ids = self._text_token_ids(label)
                    token_ids_by_label[label] = candidate_ids
                    full_ids = self.torch.cat(
                        [prompt_tensor, self.torch.tensor([candidate_ids], device=self.device)],
                        dim=1,
                    )
                    attention_mask = self.torch.ones_like(full_ids)
                    output = self.model(
                        input_ids=full_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    candidate_start = len(prompt_ids) - 1
                    candidate_logits = output.logits[0, candidate_start:-1, :]
                    candidate_targets = full_ids[0, len(prompt_ids) :]
                    log_probs = self.torch.log_softmax(candidate_logits, dim=-1)
                    selected = log_probs.gather(1, candidate_targets.unsqueeze(1)).squeeze(1)
                    score = float(selected.sum().item())
                    if not self.torch.isfinite(selected).all():
                        raise RuntimeError(f"Non-finite candidate score for {item.id}/{label}")
                    scores[label] = score
        finally:
            self._choice_prompt_index = None
        if not scores or not all(np.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"No finite candidate scores were produced for {item.id}")
        prediction = max(scores, key=scores.get)
        return BackendOutput(
            raw_output=prediction,
            metadata={
                "model": self.model_name,
                "prompt_mode": self.config.prompt_mode,
                "enable_thinking": self.config.enable_thinking,
                "inference_mode": "choice_loglikelihood",
                "rendered_prompt_hash": prompt_hash,
                "prompt_token_count": len(prompt_ids),
                "candidate_scores": scores,
                "candidate_token_ids": token_ids_by_label,
                "candidate_token_counts": {
                    label: len(token_ids) for label, token_ids in token_ids_by_label.items()
                },
            },
        )

    def extract_activation(self, item: BenchmarkItem) -> np.ndarray:
        """Extract the last non-padding token at the configured layer."""

        encoded, _rendered_prompt, _prompt_hash = self._encode_item(item)
        captured: list[Any] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, self.torch.Tensor):
                raise TypeError("Transformer layer output did not contain a Tensor")
            last_index = int(encoded["attention_mask"][0].sum().item()) - 1
            captured.append(hidden[:, last_index, :].detach().float().cpu())
            return output

        layer = self.layer_module(self.config.layer)
        handle = layer.register_forward_hook(capture)
        try:
            with self.torch.inference_mode():
                self.model(**encoded, use_cache=False)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Activation hook did not capture a layer output")
        return captured[0][0].numpy().copy()

    def layer_module(self, layer: int) -> Any:
        """Return a validated transformer block for integration diagnostics."""

        if layer < 0 or layer >= len(self._layer_stack):
            raise ValueError(
                f"Layer {layer} is outside the discovered layer stack of size "
                f"{len(self._layer_stack)}"
            )
        return self._layer_stack[layer]

    def _hook_for(self, intervention: Intervention):
        validate_vector_dimension(intervention.vector, self)
        layer = self.layer_module(intervention.layer)
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
                token_index = intervention.token_index
                if token_index is None:
                    token_index = self._choice_prompt_index
                if token_index is None:
                    token_index = -1
                if token_index < 0:
                    token_index += hidden.shape[1]
                if token_index >= hidden.shape[1]:
                    raise ValueError(
                        f"Intervention token index {token_index} is outside sequence length "
                        f"{hidden.shape[1]}"
                    )
                updated[:, token_index : token_index + 1, :] = (
                    updated[:, token_index : token_index + 1, :] + delta
                )
            if rest is None:
                return updated
            if isinstance(output, tuple):
                return (updated, *rest)
            return [updated, *rest]

        return layer.register_forward_hook(hook)

    @contextmanager
    def steer(self, intervention: Intervention) -> Iterator[None]:
        """Install exactly one hook and remove it even if generation fails."""

        self._choice_prompt_index = None
        handle = self._hook_for(intervention)
        try:
            yield
        finally:
            handle.remove()
            self._choice_prompt_index = None

    def provenance(self) -> dict[str, Any]:
        """Return model/tokenizer identity without exposing credentials."""

        parameters = list(self.model.parameters())
        devices = sorted({str(parameter.device) for parameter in parameters})
        config = getattr(self.model, "config", None)
        architectures = getattr(config, "architectures", None) if config is not None else None
        revision = self.model_revision or getattr(config, "_commit_hash", None)
        resolved_path = getattr(config, "_name_or_path", None) if config is not None else None
        return {
            "model_identifier": self.model_name,
            "resolved_model_path": resolved_path
            or ("INJECTED_TEST_MODEL" if self._injected_model else "UNKNOWN"),
            "model_revision": revision or "UNKNOWN",
            "tokenizer_identifier": self.tokenizer_name,
            "tokenizer_revision": self.config.tokenizer_revision or "UNKNOWN",
            "transformers_model_class": self.model.__class__.__name__,
            "architectures": architectures or [self.model.__class__.__name__],
            "num_layers": len(self._layer_stack),
            "hidden_size": self.hidden_size,
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "dtype": str(parameters[0].dtype) if parameters else "UNKNOWN",
            "devices": devices,
            "quantization": self.config.quantization,
            "layer_path": self._resolved_layer_path,
            "layer_index_for_activation": self.config.layer,
            "prompt_mode": self.config.prompt_mode,
            "inference_mode": self.config.inference_mode,
            "enable_thinking": self.config.enable_thinking,
            "cpu_offload_detected": any(device == "cpu" for device in devices),
            "injected_test_model": self._injected_model,
            "model_fingerprint": stable_digest(
                self.model.__class__.__name__, self.hidden_size, len(self._layer_stack),
                sum(parameter.numel() for parameter in parameters),
            ),
        }
