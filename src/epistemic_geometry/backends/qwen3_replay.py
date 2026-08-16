"""Strict Qwen3 suffix-replay boundary.

Suffix replay is intentionally isolated because it depends on the exact
Transformers Qwen3 decoder-layer and cache APIs.  This module validates the
model before any replay is attempted and fails closed when the installed
stack is not the audited one.  It must not silently approximate a Qwen3 pass.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


class SuffixReplayUnavailable(RuntimeError):
    """Raised when architecture/version guards do not permit exact replay."""


@dataclass(frozen=True)
class SuffixReplayStatus:
    supported: bool
    reason: str
    model_class: str
    transformers_version: str


class Qwen3CachedSuffixReplayEngine:
    """Architecture gate for a future exact Qwen3 suffix implementation.

    The production path remains ``cached_decode`` until a real Qwen3 model on
    the approved RunPod environment passes numerical and discrete-equivalence
    tests.  Keeping this boundary explicit prevents an attractive but
    scientifically unsafe hand-written attention approximation.
    """

    SUPPORTED_TRANSFORMERS_PREFIX = "4.57."

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        model = backend.model
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "UNKNOWN"))
        model_class = model.__class__.__name__
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = "UNKNOWN"
        reasons: list[str] = []
        if not model_type.startswith("qwen3") and "Qwen3" not in model_class:
            reasons.append(f"model is {model_class}/{model_type}, not Qwen3")
        if not transformers_version.startswith(self.SUPPORTED_TRANSFORMERS_PREFIX):
            reasons.append(
                f"Transformers {transformers_version} is not the audited "
                f"{self.SUPPORTED_TRANSFORMERS_PREFIX} series"
            )
        if getattr(backend, "_resolved_layer_path", "") != "model.model.layers":
            reasons.append(
                "resolved layer path is not model.model.layers; set and audit the exact path"
            )
        self.status = SuffixReplayStatus(
            supported=not reasons,
            reason="; ".join(reasons) if reasons else "guards passed; equivalence still required",
            model_class=model_class,
            transformers_version=transformers_version,
        )

    def require_supported(self) -> None:
        """Fail before inference when exact replay is not safe to attempt."""

        if not self.status.supported:
            raise SuffixReplayUnavailable(
                "Qwen3 cached suffix replay is unavailable: " + self.status.reason
            )

    def _attention_masks(
        self,
        core: Any,
        hidden_states: Any,
        attention_mask: Any,
        cache_position: Any,
        cache: Any,
        position_ids: Any,
    ) -> dict[str, Any]:
        """Build the same native masks used by ``Qwen3Model.forward``."""

        try:
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
        except ImportError as exc:  # pragma: no cover - guarded by real HF extra
            raise SuffixReplayUnavailable(
                "Transformers 4.57 masking utilities are unavailable"
            ) from exc
        kwargs = {
            "config": core.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        masks = {"full_attention": create_causal_mask(**kwargs)}
        if getattr(core, "has_sliding_layers", False):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
        return masks

    def predict_choice_batch(self, prepared_items: list[Any], conditions: list[Any]) -> list[Any]:
        """Replay Qwen3 suffixes after one baseline final-token trajectory.

        The prefix cache is never reused after mutation. Each replay receives a
        fresh clone of the prefix-only cache, and every suffix layer is invoked
        through the model's native decoder-layer API. This is intentionally
        narrow: heterogeneous layers are grouped by the caller and candidates
        must be single tokens.
        """

        self.require_supported()
        backend = self.backend
        torch = backend.torch
        if not prepared_items or not conditions:
            return []
        if any(not item.all_candidates_single_token for item in prepared_items):
            raise SuffixReplayUnavailable(
                "Qwen3 suffix replay currently requires single-token candidates"
            )
        layers = {int(spec.get("layer", backend.config.layer)) for spec, _ in conditions}
        if len(layers) != 1:
            raise SuffixReplayUnavailable("Suffix replay batches must share one steering layer")
        layer_index = next(iter(layers))
        backend.layer_module(layer_index)
        core = backend._model_core()
        decoder_layers = core.layers
        lm_head = backend.model.get_output_embeddings()
        if lm_head is None:
            raise SuffixReplayUnavailable("Qwen3 model has no output embedding head")

        results: dict[tuple[str, str], Any] = {}
        for item_batch in backend._planned_item_batches(prepared_items):
            (
                prefix_ids,
                prefix_mask,
                prefix_positions,
                prefix_lengths,
            ) = backend._pad_token_sequences([item.prompt_ids[:-1] for item in item_batch])
            query_ids = torch.tensor(
                [[item.prompt_ids[-1]] for item in item_batch],
                dtype=torch.long,
                device=backend.device,
            )
            query_mask = torch.ones(
                (len(item_batch), 1), dtype=torch.long, device=backend.device
            )
            attention = torch.cat([prefix_mask, query_mask], dim=1)
            query_positions = torch.tensor(
                [[length] for length in prefix_lengths],
                dtype=torch.long,
                device=backend.device,
            )
            cache_position = torch.tensor(
                [prefix_ids.shape[1]], dtype=torch.long, device=backend.device
            )
            with torch.inference_mode():
                prefix_output = backend._forward(
                    core,
                    backend._forward_kwargs(
                        core,
                        prefix_ids,
                        prefix_mask,
                        prefix_positions,
                        cache_position=torch.arange(
                            prefix_ids.shape[1], dtype=torch.long, device=backend.device
                        ),
                    ),
                    "prefill",
                )
            prefix_cache = getattr(prefix_output, "past_key_values", None)
            if prefix_cache is None:
                raise SuffixReplayUnavailable("Qwen3 prefix prefill returned no cache")

            baseline_cache = backend._clone_and_repeat_cache(prefix_cache, 1)
            captured: dict[int, Any] = {}
            handles = []

            def make_capture(index: int, captured=captured):
                def capture(_module: Any, _inputs: Any, layer_output: Any) -> Any:
                    hidden = (
                        layer_output[0]
                        if isinstance(layer_output, (tuple, list))
                        else layer_output
                    )
                    if not isinstance(hidden, torch.Tensor):
                        raise SuffixReplayUnavailable(
                            f"Qwen3 layer {index} did not return a hidden-state tensor"
                        )
                    captured[index] = hidden.detach().clone()
                    return layer_output

                return capture

            try:
                for index, decoder_layer in enumerate(decoder_layers):
                    handles.append(decoder_layer.register_forward_hook(make_capture(index)))
                with torch.inference_mode():
                    baseline_core = backend._forward(
                        core,
                        backend._forward_kwargs(
                            core,
                            query_ids,
                            attention,
                            query_positions,
                            past_key_values=baseline_cache,
                            cache_position=cache_position,
                        ),
                        "decode",
                    )
            finally:
                for handle in handles:
                    handle.remove()
            if set(captured) != set(range(len(decoder_layers))):
                raise SuffixReplayUnavailable(
                    "Qwen3 baseline trajectory did not capture every layer"
                )

            baseline_hidden = baseline_core.last_hidden_state
            baseline_output = SimpleNamespace(
                logits=lm_head(baseline_hidden), last_hidden_state=baseline_hidden
            )
            row_specs = [spec for spec, _ in conditions]
            for condition_index, (spec, vector) in enumerate(conditions):
                alpha = float(spec.get("alpha", 0.0))
                if vector is None or alpha == 0.0:
                    output = baseline_output
                else:
                    delta = backend._expand_conditions(item_batch, [(spec, vector)])
                    hidden = captured[layer_index] + delta[:, None, :].to(
                        dtype=captured[layer_index].dtype
                    )
                    replay_cache = backend._clone_and_repeat_cache(prefix_cache, 1)
                    for suffix_index in range(layer_index + 1, len(decoder_layers)):
                        masks = self._attention_masks(
                            core,
                            hidden,
                            attention,
                            cache_position,
                            replay_cache,
                            query_positions,
                        )
                        position_embeddings = core.rotary_emb(hidden, query_positions)
                        hidden = decoder_layers[suffix_index](
                            hidden,
                            attention_mask=masks[decoder_layers[suffix_index].attention_type],
                            position_ids=query_positions,
                            past_key_values=replay_cache,
                            use_cache=True,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                    hidden = core.norm(hidden)
                    output = SimpleNamespace(logits=lm_head(hidden), last_hidden_state=hidden)
                row_outputs = backend._candidate_outputs(
                    output,
                    item_batch,
                    [row_specs[condition_index]] * len(item_batch),
                    "cached_suffix_replay",
                    logit_positions=torch.zeros(
                        (len(item_batch),), dtype=torch.long, device=backend.device
                    ),
                )
                for item, row_output in zip(item_batch, row_outputs, strict=True):
                    results[(item.item_id, str(spec["condition"]))] = row_output

        return [
            (item, spec, results[(item.item_id, str(spec["condition"]))])
            for item in prepared_items
            for spec, _vector in conditions
        ]

    def replay(self, *_args: Any, **_kwargs: Any) -> Any:
        """Compatibility entry point for callers that need a batch scorer."""

        self.require_supported()
