"""Optional decoder-only Transformers backend with temporary forward hooks."""

from __future__ import annotations

import copy
import inspect
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
from epistemic_geometry.benchmarks.reasoning.engines import deterministic_length_batches
from epistemic_geometry.config import BackendConfig
from epistemic_geometry.inference.planner import group_conditions_by_layer, plan_prepared_items
from epistemic_geometry.reproducibility import require_remote_hf_execution, stable_digest
from epistemic_geometry.types import (
    BackendOutput,
    BenchmarkItem,
    Intervention,
    PreparedChoiceItem,
    SteeringVector,
)


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
        self._execution_stats: dict[str, int] = {
            "forward_calls": 0,
            "serial_candidate_forwards": 0,
            "prefill_forwards": 0,
            "decode_forwards": 0,
            "tokens_processed": 0,
        }
        if model is None:
            require_remote_hf_execution("HuggingFace model/tokenizer loading")
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
                load_kwargs["dtype"] = dtype
            if config.device_map is not None:
                load_kwargs["device_map"] = config.device_map
            elif config.device == "auto" and torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
            if config.attention_implementation != "auto":
                load_kwargs["attn_implementation"] = config.attention_implementation
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
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
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

    def generate_reasoning(
        self,
        item: BenchmarkItem,
        *,
        sampling_seed: int,
        max_new_tokens: int | None = None,
    ) -> BackendOutput:
        """Generate one seeded reasoning trajectory without any intervention.

        This is the baseline-only execution primitive for Q1 V3 calibration.
        One-shot steering is intentionally not accepted here until the new
        reasoning instrument qualifies.
        """

        if self.config.enable_thinking is not True:
            raise ValueError("Q1 V3 reasoning generation requires enable_thinking=true")
        if not self.config.do_sample:
            raise ValueError("Q1 V3 canonical reasoning generation requires do_sample=true")
        encoded, _rendered_prompt, prompt_hash = self._encode_item(item)
        input_length = int(encoded["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self.config.max_new_tokens),
            "do_sample": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        try:
            rng_devices = []
            if self.device.type == "cuda":
                rng_devices = [self.device.index or self.torch.cuda.current_device()]
            with self.torch.random.fork_rng(devices=rng_devices, enabled=True):
                self.torch.manual_seed(int(sampling_seed))
                if self.device.type == "cuda":
                    self.torch.cuda.manual_seed_all(int(sampling_seed))
                with self.torch.inference_mode():
                    generated = self.model.generate(**encoded, **generation_kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory during reasoning generation. Reduce max_new_tokens, "
                    "use a smaller batch/model, or choose an explicit supported dtype."
                ) from exc
            raise
        new_tokens = generated[0, input_length:]
        # Preserve the complete decoded trajectory; the parser is responsible
        # for whitespace normalization around the exact FINAL field.
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return BackendOutput(
            raw_output=raw_output,
            metadata={
                "model": self.model_name,
                "model_revision": self.model_revision or "UNKNOWN",
                "prompt_mode": self.config.prompt_mode,
                "enable_thinking": True,
                "rendered_prompt_hash": prompt_hash,
                "input_token_count": input_length,
                "generated_token_count": int(new_tokens.numel()),
                "generated_token_ids": [int(token) for token in new_tokens.tolist()],
                "generation_seed": int(sampling_seed),
                "generation": {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                    "max_new_tokens": int(max_new_tokens or self.config.max_new_tokens),
                },
                "intervention": "none",
            },
        )

    def generate_reasoning_view(
        self,
        view: Any,
        *,
        sampling_seed: int,
        max_new_tokens: int | None = None,
    ) -> BackendOutput:
        """Generate a reasoning view while retaining latent-view provenance."""

        item = BenchmarkItem(
            id=view.view_id,
            prompt=view.prompt,
            target=str(view.answer),
            metadata={
                "latent_id": view.latent_id,
                "view_id": view.view_id,
                "response_channel": "reasoning_exact_final",
                "source_prompt_hash": view.prompt_hash,
                "source_template_hash": view.template_hash,
            },
        )
        output = self.generate_reasoning(
            item,
            sampling_seed=sampling_seed,
            max_new_tokens=max_new_tokens,
        )
        metadata = dict(output.metadata)
        metadata.update(
            {
                "latent_id": view.latent_id,
                "view_id": view.view_id,
                "source_prompt_hash": view.prompt_hash,
                "source_template_hash": view.template_hash,
            }
        )
        return BackendOutput(raw_output=output.raw_output, metadata=metadata)

    def generate_reasoning_batch(
        self,
        rows: list[tuple[Any, int]],
        *,
        max_new_tokens: int,
        batch_size: int,
        max_prefill_tokens: int = 8192,
    ) -> list[BackendOutput]:
        """Generate independent reasoning rows with length-aware KV batching.

        Each row owns a CUDA/CPU ``torch.Generator`` seeded with the frozen
        rollout seed.  Sampling is therefore independent of batch order.  The
        model's own logits warpers implement the configured temperature/top-k/
        top-p/min-p semantics; this method only replaces the outer serial
        ``generate`` loop with explicit cached decoding.
        """

        if not rows:
            return []
        if not self.config.enable_thinking or not self.config.do_sample:
            raise ValueError("batched reasoning requires thinking and sampling enabled")
        if batch_size <= 0 or max_prefill_tokens <= 0 or max_new_tokens <= 0:
            raise ValueError("batch_size, max_prefill_tokens, and max_new_tokens must be positive")

        try:
            from transformers import GenerationConfig
            from transformers.generation.logits_process import (
                LogitsProcessorList,
                MinPLogitsWarper,
                TemperatureLogitsWarper,
                TopKLogitsWarper,
                TopPLogitsWarper,
            )
        except ImportError as exc:  # pragma: no cover - guarded by backend construction
            raise OptionalDependencyError("batched reasoning requires transformers") from exc

        prepared: list[dict[str, Any]] = []
        for view, seed in rows:
            item = BenchmarkItem(
                id=view.view_id,
                prompt=view.prompt,
                target=str(view.answer),
                metadata={
                    "latent_id": view.latent_id,
                    "view_id": view.view_id,
                    "response_channel": "reasoning_exact_final",
                    "source_prompt_hash": view.prompt_hash,
                    "source_template_hash": view.template_hash,
                },
            )
            rendered = render_prompt(
                item,
                mode=self.config.prompt_mode,
                tokenizer=self.tokenizer,
                enable_thinking=self.config.enable_thinking,
            )
            prepared.append({"view": view, "seed": int(seed), "rendered": rendered})

        previous_padding_side = getattr(self.tokenizer, "padding_side", None)
        if previous_padding_side is not None:
            self.tokenizer.padding_side = self.config.padding_side
        try:
            encoded = self.tokenizer(
                [row["rendered"].text for row in prepared],
                padding=True,
                return_tensors="pt",
            )
        finally:
            if previous_padding_side is not None:
                self.tokenizer.padding_side = previous_padding_side
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        attention_mask = encoded["attention_mask"]
        prompt_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
        # Deterministic length buckets are formed from the already-tokenized
        # prompts.  Results are restored to the caller's original row order.
        planned_batches = deterministic_length_batches(
            [(str(index), length) for index, length in enumerate(prompt_lengths)],
            batch_size=batch_size,
            max_padded_tokens=max_prefill_tokens,
        )
        batches = [[int(index) for index in batch] for batch in planned_batches]
        output_rows: list[BackendOutput | None] = [None] * len(rows)
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        # ``_get_logits_warper`` existed in some Transformers releases and was
        # folded into ``_get_logits_processor`` in others.  Construct the
        # small, explicitly frozen sampler here so the optimized path does not
        # depend on a private model method or on a particular model class.
        warpers = LogitsProcessorList()
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p))
        if generation_config.min_p is not None and generation_config.min_p > 0.0:
            warpers.append(MinPLogitsWarper(min_p=generation_config.min_p))
        eos_ids = generation_config.eos_token_id
        if eos_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = {int(value) for value in eos_ids}

        model_parameters = inspect.signature(self.model.forward).parameters
        model_has_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in model_parameters.values()
        )
        supports_cache_position = (
            "cache_position" in model_parameters or model_has_var_kwargs
        )

        for batch_index, batch_indices in enumerate(batches):
            # A deterministic max-token budget is used here.  The planner
            # never changes scientific row identity; it only chooses grouping.
            batch_inputs = encoded["input_ids"][batch_indices]
            batch_masks = encoded["attention_mask"][batch_indices]
            # Explicit position IDs make left- and right-padded batches
            # equivalent to the corresponding unbatched calls.  This is
            # particularly important for decoder-only models whose automatic
            # position inference differs across Transformers versions.
            batch_position_ids = batch_masks.long().cumsum(dim=-1) - 1
            batch_position_ids = batch_position_ids.clamp_min(0)
            sequences = batch_inputs.clone()
            masks = batch_masks.clone()
            generators = []
            for index in batch_indices:
                generator = self.torch.Generator(device=self.device)
                generator.manual_seed(int(prepared[index]["seed"]))
                generators.append(generator)
            generated: list[list[int]] = [[] for _ in batch_indices]
            finished = [False] * len(batch_indices)
            stop_reasons: list[str | None] = [None] * len(batch_indices)
            with self.torch.inference_mode():
                prefill_kwargs: dict[str, Any] = {
                    "input_ids": batch_inputs,
                    "attention_mask": batch_masks,
                    "position_ids": batch_position_ids,
                    "use_cache": True,
                }
                if supports_cache_position:
                    prefill_kwargs["cache_position"] = self.torch.arange(
                        batch_inputs.shape[1], dtype=self.torch.long, device=self.device
                    )
                model_outputs = self.model(**prefill_kwargs)
                past_key_values = model_outputs.past_key_values
                for step in range(max_new_tokens):
                    if step == 0:
                        last_positions = batch_masks.sum(dim=1).long() - 1
                        logits = model_outputs.logits[
                            self.torch.arange(len(batch_indices), device=self.device),
                            last_positions,
                        ]
                    else:
                        logits = model_outputs.logits[:, -1, :]
                    next_tokens: list[int] = []
                    for row_index in range(len(batch_indices)):
                        if finished[row_index]:
                            next_tokens.append(int(self.tokenizer.pad_token_id))
                            continue
                        row_logits = logits[row_index : row_index + 1]
                        row_logits = warpers(sequences[row_index : row_index + 1], row_logits)
                        probabilities = self.torch.softmax(row_logits, dim=-1)
                        sampled = self.torch.multinomial(
                            probabilities,
                            num_samples=1,
                            generator=generators[row_index],
                        )
                        token = int(sampled.item())
                        next_tokens.append(token)
                        generated[row_index].append(token)
                        if token in eos_set:
                            finished[row_index] = True
                            stop_reasons[row_index] = "eos_token"
                    next_tensor = self.torch.tensor(
                        next_tokens, dtype=batch_inputs.dtype, device=self.device
                    ).unsqueeze(1)
                    sequences = self.torch.cat((sequences, next_tensor), dim=1)
                    masks = self.torch.cat(
                        (
                            masks,
                            self.torch.ones(
                                (len(batch_indices), 1), dtype=masks.dtype, device=self.device
                            ),
                        ),
                        dim=1,
                    )
                    if all(finished):
                        break
                    decode_kwargs: dict[str, Any] = {
                        "input_ids": next_tensor,
                        "attention_mask": masks,
                        "position_ids": (masks.sum(dim=1).long() - 1)
                        .unsqueeze(1)
                        .clamp_min(0),
                        "past_key_values": past_key_values,
                        "use_cache": True,
                    }
                    if supports_cache_position:
                        decode_kwargs["cache_position"] = self.torch.full(
                            (len(batch_indices),),
                            sequences.shape[1] - 1,
                            dtype=self.torch.long,
                            device=self.device,
                        )
                    model_outputs = self.model(**decode_kwargs)
                    past_key_values = model_outputs.past_key_values

            for row_index, original_index in enumerate(batch_indices):
                tokens = tuple(generated[row_index])
                if stop_reasons[row_index] is None:
                    stop_reasons[row_index] = "max_new_tokens"
                view = prepared[original_index]["view"]
                output_rows[original_index] = BackendOutput(
                    raw_output=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    metadata={
                        "model": self.model_name,
                        "model_revision": self.model_revision or "UNKNOWN",
                        "prompt_mode": self.config.prompt_mode,
                        "enable_thinking": True,
                        "rendered_prompt_hash": prepared[original_index]["rendered"].hash,
                        "input_token_count": prompt_lengths[original_index],
                        "generated_token_count": len(tokens),
                        "generated_token_ids": list(tokens),
                        "generation_seed": prepared[original_index]["seed"],
                        "stop_reason": stop_reasons[row_index],
                        "generation": {
                            "do_sample": True,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "top_k": self.config.top_k,
                            "min_p": self.config.min_p,
                            "max_new_tokens": max_new_tokens,
                        },
                        "batch_index": batch_index,
                        "batch_size": len(batch_indices),
                        "batch_prompt_lengths": [prompt_lengths[index] for index in batch_indices],
                        "batch_padded_prefill_tokens": len(batch_indices)
                        * max(prompt_lengths[index] for index in batch_indices),
                        "batch_planner": "q1-v3-reasoning-length-bucket-v1",
                        "intervention": "none",
                        "view_id": view.view_id,
                        "source_prompt_hash": view.prompt_hash,
                        "source_template_hash": view.template_hash,
                    },
                )
        if any(output is None for output in output_rows):
            raise RuntimeError("batched reasoning did not produce one output per row")
        return [output for output in output_rows if output is not None]

    def derive_reasoning_prefix_output(
        self,
        output: BackendOutput,
        *,
        prefix_length: int,
        source_max_budget: int,
        physical_generation_id: str,
        derived_from_prefix: bool,
        natural_completion_length: int,
        reasoning_budget: int,
    ) -> BackendOutput:
        """Decode an exact generated-token prefix without rerunning the model."""

        token_ids = tuple(int(token) for token in output.metadata.get("generated_token_ids", ()))
        if prefix_length < 0 or prefix_length > len(token_ids):
            raise ValueError("prefix_length is outside generated token range")
        prefix_ids = token_ids[:prefix_length]
        metadata = dict(output.metadata)
        metadata.update(
            {
                "generated_token_ids": list(prefix_ids),
                "generated_token_count": len(prefix_ids),
                "physical_generation_id": physical_generation_id,
                "source_max_budget": source_max_budget,
                "prefix_length": prefix_length,
                "derived_from_prefix": derived_from_prefix,
                "natural_completion_length": natural_completion_length,
                "reasoning_budget": reasoning_budget,
            }
        )
        return BackendOutput(
            raw_output=self.tokenizer.decode(prefix_ids, skip_special_tokens=True),
            metadata=metadata,
        )

    def _candidate_labels(self, item: BenchmarkItem) -> list[str]:
        labels = item.metadata.get("candidate_labels", self.config.candidate_labels)
        if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(f"Item {item.id} does not provide valid candidate labels")
        return labels

    def prepare_choice_item(self, item: BenchmarkItem) -> PreparedChoiceItem:
        """Render and tokenize one choice item once for all conditions."""

        if self.config.inference_mode != "choice_loglikelihood":
            raise ValueError("PreparedChoiceItem requires choice_loglikelihood inference")
        encoded, rendered_prompt, prompt_hash = self._encode_item(item)
        mask = encoded["attention_mask"][0].bool()
        prompt_ids = tuple(int(value) for value in encoded["input_ids"][0][mask].tolist())
        labels = tuple(self._candidate_labels(item))
        token_ids: dict[str, tuple[int, ...]] = {}
        context_ids: dict[str, tuple[int, ...]] = {}
        for label in labels:
            standalone = tuple(self._text_token_ids(label))
            token_ids[label] = standalone
            joined = self.tokenizer(rendered_prompt + label, add_special_tokens=False)
            joined_values = joined["input_ids"] if "input_ids" in joined else joined
            if hasattr(joined_values, "tolist"):
                joined_values = joined_values.tolist()
            if joined_values and isinstance(joined_values[0], list):
                joined_values = joined_values[0]
            joined_ids = tuple(int(value) for value in joined_values)
            context_ids[label] = (
                joined_ids[len(prompt_ids) :]
                if joined_ids[: len(prompt_ids)] == prompt_ids
                else tuple()
            )
        semantic_ids = item.metadata.get("semantic_option_ids")
        if not isinstance(semantic_ids, list):
            semantic_ids = list(range(len(labels)))
        return PreparedChoiceItem(
            item_id=item.id,
            target=item.target,
            metadata=dict(item.metadata),
            rendered_prompt=rendered_prompt,
            rendered_prompt_hash=prompt_hash,
            prompt_ids=prompt_ids,
            candidate_labels=labels,
            candidate_token_ids=token_ids,
            context_compatible_candidate_ids=context_ids,
            semantic_option_ids=tuple(int(value) for value in semantic_ids),
            permutation_id=item.metadata.get("permutation_id"),
        )

    def prepare_choice_items(self, items: list[BenchmarkItem]) -> list[PreparedChoiceItem]:
        """Prepare a stable list before entering the GPU inference loop."""

        return [self.prepare_choice_item(item) for item in items]

    def candidate_token_audit(self, prepared: PreparedChoiceItem) -> dict[str, Any]:
        """Report boundary-aware candidate tokenization for one rendered prompt."""

        return {
            "item_id": prepared.item_id,
            "rendered_prompt_hash": prepared.rendered_prompt_hash,
            "candidates": {
                label: {
                    "standalone_token_ids": list(prepared.candidate_token_ids[label]),
                    "context_compatible_token_ids": list(
                        prepared.context_compatible_candidate_ids[label]
                    ),
                    "standalone_token_count": len(prepared.candidate_token_ids[label]),
                    "context_compatible": bool(
                        prepared.context_compatible_candidate_ids[label]
                    ),
                }
                for label in prepared.candidate_labels
            },
            "all_standalone_single_token": prepared.all_candidates_single_token,
        }

    def _text_token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        values = encoded["input_ids"] if "input_ids" in encoded else encoded
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

        encoded, _rendered_prompt, prompt_hash = self._encode_item(item)
        prompt_length = int(encoded["attention_mask"][0].sum().item())
        prompt_ids = encoded["input_ids"][0, :prompt_length].tolist()
        labels = self._candidate_labels(item)
        scores: dict[str, float] = {}
        token_ids_by_label: dict[str, list[int]] = {}
        self._choice_prompt_index = len(prompt_ids) - 1
        try:
            with self.torch.inference_mode():
                candidate_rows: list[list[int]] = []
                for label in labels:
                    candidate_ids = self._text_token_ids(label)
                    token_ids_by_label[label] = candidate_ids
                    candidate_rows.append(candidate_ids)
                for label, candidate_ids in zip(labels, candidate_rows, strict=True):
                    full_ids = self.torch.tensor(
                        [prompt_ids + candidate_ids], dtype=self.torch.long, device=self.device
                    )
                    attention_mask = self.torch.ones_like(full_ids)
                    try:
                        output = self._forward(
                            self.model,
                            {
                                "input_ids": full_ids,
                                "attention_mask": attention_mask,
                                "use_cache": False,
                            },
                            "serial_candidate",
                        )
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise RuntimeError(
                                "CUDA out of memory during choice scoring. Reduce the candidate "
                                "label count or use a smaller model/device configuration."
                            ) from exc
                        raise
                    candidate_start = len(prompt_ids) - 1
                    candidate_logits = output.logits[
                        0, candidate_start : candidate_start + len(candidate_ids), :
                    ]
                    candidate_targets = full_ids[0, len(prompt_ids) :]
                    log_probs = self._choice_log_softmax(candidate_logits)
                    selected = log_probs.gather(1, candidate_targets.unsqueeze(1)).squeeze(1)
                    if not self.torch.isfinite(selected).all():
                        raise RuntimeError(f"Non-finite candidate score for {item.id}/{label}")
                    scores[label] = float(selected.sum().item())
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

    def _choice_log_softmax(self, candidate_logits: Any) -> Any:
        """Compute candidate probabilities in FP32 without changing model weights.

        Q1 V1 used the model output dtype directly. V1.1 freezes the numerical
        audit by promoting only this probability calculation, leaving the
        BF16 model, activations, and candidate-likelihood semantics unchanged.
        """

        return self.torch.log_softmax(candidate_logits.float(), dim=-1)

    def reset_execution_stats(self) -> None:
        """Reset inference counters used by engineering profiles."""

        for key in self._execution_stats:
            self._execution_stats[key] = 0

    def execution_stats(self) -> dict[str, int]:
        """Return a snapshot of counted expensive forward operations."""

        return dict(self._execution_stats)

    def _forward(self, module: Any, kwargs: dict[str, Any], phase: str) -> Any:
        """Call a model/core forward while recording non-scientific counters."""

        self._execution_stats["forward_calls"] += 1
        if phase == "serial_candidate":
            self._execution_stats["serial_candidate_forwards"] += 1
        elif phase == "prefill":
            self._execution_stats["prefill_forwards"] += 1
        elif phase == "decode":
            self._execution_stats["decode_forwards"] += 1
        input_ids = kwargs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "numel"):
            self._execution_stats["tokens_processed"] += int(input_ids.numel())
        return module(**kwargs)

    def predict_choice_batch(
        self,
        prepared_items: list[PreparedChoiceItem],
        conditions: list[tuple[dict[str, Any], SteeringVector | None]],
        mode: str | None = None,
    ) -> list[tuple[PreparedChoiceItem, dict[str, Any], BackendOutput]]:
        """Evaluate many item/condition pairs with explicit cache provenance.

        The returned order is item-major, then condition-major.  This method is
        deliberately separate from ``predict``: ``serial_reference`` remains
        the untouched correctness oracle, while optimized callers opt into
        prepared prompts, row-wise deltas, and prefix-cache reuse explicitly.
        """

        if not prepared_items or not conditions:
            return []
        execution_mode = mode or self.config.execution_mode
        if execution_mode == "serial_reference":
            raise ValueError("predict_choice_batch requires a non-serial execution mode")
        if execution_mode == "cached_suffix_replay":
            from epistemic_geometry.backends.qwen3_replay import Qwen3CachedSuffixReplayEngine

            Qwen3CachedSuffixReplayEngine(self).require_supported()
        condition_layers = {
            int(spec.get("layer", self.config.layer)) for spec, _vector in conditions
        }
        if len(condition_layers) > 1:
            grouped_outputs: list[tuple[PreparedChoiceItem, dict[str, Any], BackendOutput]] = []
            for grouped_conditions in group_conditions_by_layer(
                conditions, self.config.layer
            ).values():
                grouped_outputs.extend(
                    self.predict_choice_batch(prepared_items, grouped_conditions, execution_mode)
                )
            output_by_key = {
                (item.item_id, str(spec["condition"])): (item, spec, output)
                for item, spec, output in grouped_outputs
            }
            return [
                output_by_key[(item.item_id, str(spec["condition"]))]
                for item in prepared_items
                for spec, _vector in conditions
            ]
        if any(len(item.prompt_ids) < 2 for item in prepared_items):
            raise ValueError("Cached choice scoring requires at least two prompt tokens")
        if any(not item.all_candidates_single_token for item in prepared_items):
            return self._predict_choice_batch_cached_multitoken(prepared_items, conditions)
        if execution_mode == "full_prompt_batched":
            return self._predict_choice_batch_full_prompt(prepared_items, conditions)
        if execution_mode == "cached_decode":
            return self._predict_choice_batch_cached(prepared_items, conditions)
        if execution_mode == "cached_suffix_replay":
            from epistemic_geometry.backends.qwen3_replay import Qwen3CachedSuffixReplayEngine

            return Qwen3CachedSuffixReplayEngine(self).predict_choice_batch(
                prepared_items, conditions
            )
        raise ValueError(f"Unsupported choice execution mode: {execution_mode}")

    def _pad_token_sequences(
        self, sequences: list[tuple[int, ...]]
    ) -> tuple[Any, Any, Any, list[int]]:
        torch = self.torch
        if not sequences:
            raise ValueError("Cannot pad an empty sequence list")
        max_length = max(len(sequence) for sequence in sequences)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        input_ids = torch.full(
            (len(sequences), max_length), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(sequences), max_length), dtype=torch.long, device=self.device
        )
        lengths: list[int] = []
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            lengths.append(length)
            if self.config.padding_side == "left":
                start = max_length - length
            else:
                start = 0
            input_ids[row, start : start + length] = torch.tensor(
                sequence, dtype=torch.long, device=self.device
            )
            attention_mask[row, start : start + length] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return input_ids, attention_mask, position_ids, lengths

    def _expand_conditions(
        self,
        prepared_items: list[PreparedChoiceItem],
        condition_chunk: list[tuple[dict[str, Any], SteeringVector | None]],
    ) -> Any:
        """Create one B*C x hidden row-delta matrix in item-major order."""

        torch = self.torch
        deltas: list[Any] = []
        layer_values: set[int] = set()
        for _item in prepared_items:
            for spec, vector in condition_chunk:
                layer = int(spec.get("layer", self.config.layer))
                layer_values.add(layer)
                if vector is None or float(spec.get("alpha", 0.0)) == 0.0:
                    deltas.append(torch.zeros(self.hidden_size, device=self.device))
                    continue
                if vector.dimension != self.hidden_size:
                    raise ValueError(
                        f"Steering vector dimension {vector.dimension} does not match "
                        f"hidden size {self.hidden_size}"
                    )
                values = torch.as_tensor(
                    vector.values, dtype=next(self.model.parameters()).dtype, device=self.device
                )
                deltas.append(values * float(spec["alpha"]))
        if len(layer_values) != 1:
            raise NotImplementedError(
                "One optimized decode batch must share a steering layer; group heterogeneous "
                "layers before execution"
            )
        return torch.stack(deltas, dim=0)

    def _rowwise_hook(self, layer: int, deltas: Any, target_positions: Any):
        torch = self.torch
        module = self.layer_module(layer)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, torch.Tensor):
                hidden = output
                rest: tuple[Any, ...] | None = None
            elif (
                isinstance(output, (tuple, list))
                and output
                and isinstance(output[0], torch.Tensor)
            ):
                hidden = output[0]
                rest = tuple(output[1:])
            else:
                raise TypeError("Optimized hook expected Tensor or tuple[Tensor, ...]")
            if hidden.shape[0] != deltas.shape[0]:
                raise RuntimeError(
                    f"Row-delta batch {deltas.shape[0]} does not match hidden batch "
                    f"{hidden.shape[0]}"
                )
            updated = hidden.clone()
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            updated[rows, target_positions, :] += deltas.to(
                device=hidden.device, dtype=hidden.dtype
            )
            if rest is None:
                return updated
            if isinstance(output, tuple):
                return (updated, *rest)
            return [updated, *rest]

        return module.register_forward_hook(hook)

    def _model_core(self) -> Any:
        if hasattr(self.model, "model") and callable(getattr(self.model.model, "forward", None)):
            return self.model.model
        if hasattr(self.model, "transformer"):
            return self.model.transformer
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else None
        if base is not None:
            return base
        raise RuntimeError("Could not locate the decoder core for optimized choice scoring")

    def _forward_kwargs(
        self,
        module: Any,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        past_key_values: Any | None = None,
        cache_position: Any | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "return_dict": True,
        }
        parameters = inspect.signature(module.forward).parameters
        has_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if past_key_values is not None and (
            "past_key_values" in parameters or has_var_kwargs
        ):
            kwargs["past_key_values"] = past_key_values
        if cache_position is not None and (
            "cache_position" in parameters or has_var_kwargs
        ):
            kwargs["cache_position"] = cache_position
        return kwargs

    def _clone_and_repeat_cache(self, cache: Any, repeats: int) -> Any:
        """Copy a Transformers cache before a decode mutates it."""

        if cache is None:
            raise RuntimeError("The decoder did not return a prefix cache")
        if hasattr(cache, "batch_repeat_interleave"):
            repeated = copy.deepcopy(cache)
            repeated.batch_repeat_interleave(repeats)
            return repeated
        if isinstance(cache, (tuple, list)):
            layers = []
            for layer_cache in cache:
                if isinstance(layer_cache, (tuple, list)):
                    layers.append(
                        type(layer_cache)(
                            tensor.repeat_interleave(repeats, dim=0)
                            for tensor in layer_cache
                        )
                    )
                else:
                    layers.append(layer_cache.repeat_interleave(repeats, dim=0))
            return type(cache)(layers)
        raise TypeError(f"Unsupported Transformers cache type: {type(cache)!r}")

    def _candidate_outputs(
        self,
        output: Any,
        rows: list[PreparedChoiceItem],
        row_conditions: list[dict[str, Any]],
        engine: str,
        logit_positions: Any | None = None,
    ) -> list[BackendOutput]:
        torch = self.torch
        candidate_counts = [len(row.candidate_labels) for row in rows]
        max_candidates = max(candidate_counts)
        candidate_ids = torch.zeros(
            (len(rows), max_candidates), dtype=torch.long, device=self.device
        )
        for index, row in enumerate(rows):
            values = [row.candidate_token_ids[label][0] for label in row.candidate_labels]
            candidate_ids[index, : len(values)] = torch.tensor(
                values, dtype=torch.long, device=self.device
            )
        semantics = self.config.candidate_head_mode
        if logit_positions is None:
            sequence_length = (
                output.last_hidden_state.shape[1]
                if semantics == "candidate_only"
                else output.logits.shape[1]
            )
            logit_positions = torch.full(
                (len(rows),), sequence_length - 1, dtype=torch.long, device=self.device
            )
        if semantics == "candidate_only":
            hidden = output.last_hidden_state[
                torch.arange(len(rows), device=self.device), logit_positions, :
            ]
            output_embeddings = self.model.get_output_embeddings()
            if output_embeddings is None or not hasattr(output_embeddings, "weight"):
                raise RuntimeError("Candidate-only head requires output embedding weights")
            weights = output_embeddings.weight[candidate_ids]
            scores_tensor = torch.einsum("rd,rcd->rc", hidden, weights)
            if getattr(output_embeddings, "bias", None) is not None:
                scores_tensor = scores_tensor + output_embeddings.bias[candidate_ids]
            score_semantics = "candidate_logits_no_vocab_normalization"
        else:
            logits = output.logits[
                torch.arange(len(rows), device=self.device), logit_positions, :
            ]
            scores_tensor = self._choice_log_softmax(logits).gather(1, candidate_ids)
            score_semantics = "full_vocab_log_probability"
        results: list[BackendOutput] = []
        for index, (row, spec) in enumerate(zip(rows, row_conditions, strict=True)):
            labels = row.candidate_labels
            scores = {
                label: float(scores_tensor[index, label_index].item())
                for label_index, label in enumerate(labels)
            }
            prediction = max(scores, key=scores.get)
            results.append(
                BackendOutput(
                    raw_output=prediction,
                    metadata={
                        "model": self.model_name,
                        "prompt_mode": self.config.prompt_mode,
                        "enable_thinking": self.config.enable_thinking,
                        "inference_mode": "choice_loglikelihood",
                        "rendered_prompt_hash": row.rendered_prompt_hash,
                        "prompt_token_count": row.prompt_length,
                        "candidate_scores": scores,
                        "candidate_score_semantics": score_semantics,
                        "candidate_token_ids": {
                            label: list(row.candidate_token_ids[label]) for label in labels
                        },
                        "candidate_token_counts": {
                            label: len(row.candidate_token_ids[label]) for label in labels
                        },
                        "execution_engine": engine,
                        "condition": spec.get("condition"),
                        "prefix_cache_enabled": engine == "cached_decode",
                    },
                )
            )
        return results

    def _planned_item_batches(
        self, prepared_items: list[PreparedChoiceItem]
    ) -> list[list[PreparedChoiceItem]]:
        plans, _payload = plan_prepared_items(
            prepared_items,
            max_items=self.config.item_batch_size,
            max_prefill_tokens=self.config.max_prefill_tokens,
        )
        by_id = {item.item_id: item for item in prepared_items}
        return [[by_id[item_id] for item_id in plan.item_ids] for plan in plans]

    def _predict_choice_batch_full_prompt(
        self,
        prepared_items: list[PreparedChoiceItem],
        conditions: list[tuple[dict[str, Any], SteeringVector | None]],
    ) -> list[tuple[PreparedChoiceItem, dict[str, Any], BackendOutput]]:
        torch = self.torch
        results: dict[tuple[str, str], BackendOutput] = {}
        chunk_size = self.config.condition_chunk_size
        for item_batch in self._planned_item_batches(prepared_items):
            sequences = [
                item.prompt_ids
                + (
                    (item.candidate_token_ids[item.candidate_labels[0]][0],)
                    if self.config.serial_shape_reference
                    else ()
                )
                for item in item_batch
            ]
            input_ids, attention_mask, position_ids, lengths = self._pad_token_sequences(
                sequences
            )
            target_offset = 2 if self.config.serial_shape_reference else 1
            if self.config.padding_side == "left":
                target_positions_by_item = [input_ids.shape[1] - target_offset] * len(lengths)
            else:
                target_positions_by_item = [length - target_offset for length in lengths]
            for condition_start in range(0, len(conditions), chunk_size):
                condition_chunk = conditions[condition_start : condition_start + chunk_size]
                row_items = [item for item in item_batch for _ in condition_chunk]
                row_specs = [spec for _item in item_batch for spec, _vector in condition_chunk]
                row_input_ids = input_ids.repeat_interleave(len(condition_chunk), dim=0)
                row_attention = attention_mask.repeat_interleave(len(condition_chunk), dim=0)
                row_positions = position_ids.repeat_interleave(len(condition_chunk), dim=0)
                row_deltas = self._expand_conditions(item_batch, condition_chunk)
                layer = int(condition_chunk[0][0].get("layer", self.config.layer))
                target_positions = torch.tensor(
                    [
                        target_position
                        for target_position in target_positions_by_item
                        for _condition in condition_chunk
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                handle = self._rowwise_hook(layer, row_deltas, target_positions)
                try:
                    with torch.inference_mode():
                        if self.config.candidate_head_mode == "candidate_only":
                            core = self._model_core()
                            output = self._forward(
                                core,
                                self._forward_kwargs(
                                    core, row_input_ids, row_attention, row_positions
                                ),
                                "decode",
                            )
                        else:
                            output = self._forward(
                                self.model,
                                self._forward_kwargs(
                                    self.model, row_input_ids, row_attention, row_positions
                                ),
                                "decode",
                            )
                finally:
                    handle.remove()
                outputs = self._candidate_outputs(
                    output,
                    row_items,
                    row_specs,
                    (
                        "full_prompt_batched_serial_shape"
                        if self.config.serial_shape_reference
                        else "full_prompt_batched"
                    ),
                    logit_positions=target_positions,
                )
                for item, spec, output_row in zip(row_items, row_specs, outputs, strict=True):
                    results[(item.item_id, str(spec["condition"]))] = output_row
        return [
            (item, spec, results[(item.item_id, str(spec["condition"]))])
            for item in prepared_items
            for spec, _vector in conditions
        ]

    def _predict_choice_batch_cached(
        self,
        prepared_items: list[PreparedChoiceItem],
        conditions: list[tuple[dict[str, Any], SteeringVector | None]],
    ) -> list[tuple[PreparedChoiceItem, dict[str, Any], BackendOutput]]:
        torch = self.torch
        results: dict[tuple[str, str], BackendOutput] = {}
        chunk_size = self.config.condition_chunk_size
        core = self._model_core()
        if self.config.padding_side != "left":
            raise ValueError(
                "cached_decode currently requires left padding so prefix cache positions "
                "remain contiguous; use left padding or full_prompt_batched"
            )
        for item_batch in self._planned_item_batches(prepared_items):
            prefix_ids, prefix_mask, _prefix_positions, prefix_lengths = self._pad_token_sequences(
                [item.prompt_ids[:-1] for item in item_batch]
            )
            query_ids = torch.tensor(
                [[item.prompt_ids[-1]] for item in item_batch],
                dtype=torch.long,
                device=self.device,
            )
            with torch.inference_mode():
                prefix_output = self._forward(
                    core,
                    self._forward_kwargs(
                        core,
                        prefix_ids,
                        prefix_mask,
                        _prefix_positions,
                        cache_position=torch.arange(
                            prefix_ids.shape[1], dtype=torch.long, device=self.device
                        ),
                    ),
                    "prefill",
                )
            prefix_cache = getattr(prefix_output, "past_key_values", None)
            if prefix_cache is None and isinstance(prefix_output, (tuple, list)):
                prefix_cache = prefix_output[1]
            if prefix_cache is None:
                raise RuntimeError("Cached decode requested but model returned no past_key_values")
            for condition_start in range(0, len(conditions), chunk_size):
                condition_chunk = conditions[condition_start : condition_start + chunk_size]
                condition_count = len(condition_chunk)
                row_items = [item for item in item_batch for _ in condition_chunk]
                row_specs = [spec for _item in item_batch for spec, _vector in condition_chunk]
                row_query_ids = query_ids.repeat_interleave(condition_count, dim=0)
                row_prefix_mask = prefix_mask.repeat_interleave(condition_count, dim=0)
                query_mask = torch.ones(
                    (row_query_ids.shape[0], 1), dtype=torch.long, device=self.device
                )
                row_attention = torch.cat([row_prefix_mask, query_mask], dim=1)
                query_position_ids = torch.tensor(
                    [[length] for length in prefix_lengths],
                    dtype=torch.long,
                    device=self.device,
                ).repeat_interleave(condition_count, dim=0)
                row_deltas = self._expand_conditions(item_batch, condition_chunk)
                layer = int(condition_chunk[0][0].get("layer", self.config.layer))
                target_positions = torch.zeros(
                    (len(row_items),), dtype=torch.long, device=self.device
                )
                handle = self._rowwise_hook(layer, row_deltas, target_positions)
                try:
                    repeated_cache = self._clone_and_repeat_cache(prefix_cache, condition_count)
                    with torch.inference_mode():
                        if self.config.candidate_head_mode == "candidate_only":
                            output = self._forward(
                                core,
                                self._forward_kwargs(
                                    core,
                                    row_query_ids,
                                    row_attention,
                                    query_position_ids,
                                    past_key_values=repeated_cache,
                                    cache_position=torch.tensor(
                                        [prefix_ids.shape[1]], dtype=torch.long, device=self.device
                                    ),
                                ),
                                "decode",
                            )
                        else:
                            output = self._forward(
                                self.model,
                                self._forward_kwargs(
                                    self.model,
                                    row_query_ids,
                                    row_attention,
                                    query_position_ids,
                                    past_key_values=repeated_cache,
                                    cache_position=torch.tensor(
                                        [prefix_ids.shape[1]], dtype=torch.long, device=self.device
                                    ),
                                ),
                                "decode",
                            )
                finally:
                    handle.remove()
                outputs = self._candidate_outputs(output, row_items, row_specs, "cached_decode")
                for item, spec, output_row in zip(row_items, row_specs, outputs, strict=True):
                    results[(item.item_id, str(spec["condition"]))] = output_row
        return [
            (item, spec, results[(item.item_id, str(spec["condition"]))])
            for item in prepared_items
            for spec, _vector in conditions
        ]

    def _predict_choice_batch_cached_multitoken(
        self,
        prepared_items: list[PreparedChoiceItem],
        conditions: list[tuple[dict[str, Any], SteeringVector | None]],
    ) -> list[tuple[PreparedChoiceItem, dict[str, Any], BackendOutput]]:
        """Score multi-token labels from one shared prefix cache per item.

        This deliberately favors a small, auditable fallback over candidate-wise
        full-prompt forwards.  It is not the common MMLU-Pro path, where the
        single-token fast path is expected to apply.
        """

        if self.config.candidate_head_mode == "candidate_only":
            raise ValueError(
                "candidate_only head is only defined for single-token candidates; "
                "use full_vocab_reference for the multi-token fallback"
            )
        torch = self.torch
        core = self._model_core()
        results: dict[tuple[str, str], BackendOutput] = {}
        for item in prepared_items:
            prefix_ids, prefix_mask, prefix_positions, prefix_lengths = self._pad_token_sequences(
                [item.prompt_ids[:-1]]
            )
            query_ids = torch.tensor(
                [[item.prompt_ids[-1]]], dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                prefix_output = self._forward(
                    core,
                    self._forward_kwargs(
                        core,
                        prefix_ids,
                        prefix_mask,
                        prefix_positions,
                        cache_position=torch.arange(
                            prefix_ids.shape[1], dtype=torch.long, device=self.device
                        ),
                    ),
                    "prefill",
                )
            prefix_cache = getattr(prefix_output, "past_key_values", None)
            if prefix_cache is None and isinstance(prefix_output, (tuple, list)):
                prefix_cache = prefix_output[1]
            if prefix_cache is None:
                raise RuntimeError("Cached decode requested but model returned no past_key_values")
            for spec, vector in conditions:
                delta = self._expand_conditions([item], [(spec, vector)])
                handle = self._rowwise_hook(
                    int(spec.get("layer", self.config.layer)), delta, torch.zeros(
                        (1,), dtype=torch.long, device=self.device
                    )
                )
                try:
                    query_cache = self._clone_and_repeat_cache(prefix_cache, 1)
                    query_attention = torch.cat(
                        [prefix_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)],
                        dim=1,
                    )
                    with torch.inference_mode():
                        query_output = self._forward(
                            self.model,
                            self._forward_kwargs(
                                self.model,
                                query_ids,
                                query_attention,
                                torch.tensor(
                                    [[prefix_lengths[0]]], dtype=torch.long, device=self.device
                                ),
                                past_key_values=query_cache,
                                cache_position=torch.tensor(
                                    [prefix_ids.shape[1]], dtype=torch.long, device=self.device
                                ),
                            ),
                            "decode",
                        )
                finally:
                    handle.remove()
                scores: dict[str, float] = {}
                first_log_probs = self._choice_log_softmax(query_output.logits[:, -1, :])
                query_cache = getattr(query_output, "past_key_values", None)
                if query_cache is None and isinstance(query_output, (tuple, list)):
                    query_cache = query_output[1]
                if query_cache is None:
                    raise RuntimeError("Multi-token fallback did not receive a continuation cache")
                for label in item.candidate_labels:
                    candidate_ids = item.candidate_token_ids[label]
                    total = first_log_probs[0, candidate_ids[0]]
                    continuation_cache = query_cache
                    for offset, token_id in enumerate(candidate_ids[1:], start=1):
                        continuation_cache = self._clone_and_repeat_cache(continuation_cache, 1)
                        input_token = torch.tensor(
                            [[token_id]], dtype=torch.long, device=self.device
                        )
                        attention = torch.ones(
                            (1, len(item.prompt_ids) + offset),
                            dtype=torch.long,
                            device=self.device,
                        )
                        with torch.inference_mode():
                            continuation_output = self._forward(
                                self.model,
                                self._forward_kwargs(
                                    self.model,
                                    input_token,
                                    attention,
                                    torch.tensor(
                                        [[len(item.prompt_ids) + offset]],
                                        dtype=torch.long,
                                        device=self.device,
                                    ),
                                    past_key_values=continuation_cache,
                                    cache_position=torch.tensor(
                                        [prefix_ids.shape[1] + offset],
                                        dtype=torch.long,
                                        device=self.device,
                                    ),
                                ),
                                "decode",
                            )
                        total = total + self._choice_log_softmax(
                            continuation_output.logits[:, -1, :]
                        )[0, candidate_ids[offset]]
                        continuation_cache = getattr(
                            continuation_output, "past_key_values", None
                        )
                        if continuation_cache is None:
                            raise RuntimeError(
                                "Multi-token fallback lost its continuation cache"
                            )
                    scores[label] = float(total.item())
                prediction = max(scores, key=scores.get)
                results[(item.item_id, str(spec["condition"]))] = BackendOutput(
                    raw_output=prediction,
                    metadata={
                        "model": self.model_name,
                        "prompt_mode": self.config.prompt_mode,
                        "enable_thinking": self.config.enable_thinking,
                        "inference_mode": "choice_loglikelihood",
                        "rendered_prompt_hash": item.rendered_prompt_hash,
                        "prompt_token_count": item.prompt_length,
                        "candidate_scores": scores,
                        "candidate_score_semantics": "full_vocab_log_probability",
                        "candidate_token_ids": {
                            label: list(item.candidate_token_ids[label])
                            for label in item.candidate_labels
                        },
                        "candidate_token_counts": {
                            label: len(item.candidate_token_ids[label])
                            for label in item.candidate_labels
                        },
                        "execution_engine": "cached_decode_multitoken",
                        "condition": spec.get("condition"),
                        "prefix_cache_enabled": True,
                    },
                )
        return [
            (item, spec, results[(item.item_id, str(spec["condition"]))])
            for item in prepared_items
            for spec, _vector in conditions
        ]

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

    def extract_activations_batch(
        self, items: list[BenchmarkItem], layers: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        """Capture selected last-token activations in one padded forward pass."""

        if not items:
            return {}
        selected_layers = layers or [self.config.layer]
        for layer in selected_layers:
            self.layer_module(layer)
        encoded_items = [self._encode_item(item) for item in items]
        sequences = []
        for encoded, _prompt, _prompt_hash in encoded_items:
            mask = encoded["attention_mask"][0].bool()
            sequences.append(tuple(int(value) for value in encoded["input_ids"][0][mask].tolist()))
        input_ids, attention_mask, position_ids, lengths = self._pad_token_sequences(sequences)
        if self.config.padding_side == "left":
            positions = [input_ids.shape[1] - 1] * len(lengths)
        else:
            positions = [length - 1 for length in lengths]
        target_positions = self.torch.tensor(positions, dtype=self.torch.long, device=self.device)
        captured: dict[int, np.ndarray] = {}
        handles = []

        def make_capture(layer: int):
            def capture(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(hidden, self.torch.Tensor):
                    raise TypeError("Transformer layer output did not contain a Tensor")
                rows = self.torch.arange(hidden.shape[0], device=hidden.device)
                captured[layer] = hidden[rows, target_positions, :].detach().float().cpu().numpy()
                return output

            return capture

        try:
            for layer in selected_layers:
                handles.append(self.layer_module(layer).register_forward_hook(make_capture(layer)))
            with self.torch.inference_mode():
                self._forward(
                    self.model,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "use_cache": False,
                    },
                    "decode",
                )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(selected_layers):
            raise RuntimeError(
                "Batched activation extraction did not capture every requested layer"
            )
        return captured

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
        attention_backend = (
            getattr(config, "_attn_implementation", None) if config is not None else None
        )
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
            "execution_engine": self.config.execution_mode,
            "candidate_head_mode": self.config.candidate_head_mode,
            "attention_implementation": attention_backend
            or self.config.attention_implementation,
            "torch_compile": self.config.torch_compile,
            "cuda_graphs": self.config.cuda_graphs,
            "item_batch_size": self.config.item_batch_size,
            "condition_chunk_size": self.config.condition_chunk_size,
            "max_prefill_tokens": self.config.max_prefill_tokens,
            "padding_side": self.config.padding_side,
            "serial_shape_reference": self.config.serial_shape_reference,
            "enable_thinking": self.config.enable_thinking,
            "cpu_offload_detected": any(device == "cpu" for device in devices),
            "injected_test_model": self._injected_model,
            "model_fingerprint": stable_digest(
                self.model.__class__.__name__, self.hidden_size, len(self._layer_stack),
                sum(parameter.numel() for parameter in parameters),
            ),
        }
