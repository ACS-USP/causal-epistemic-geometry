"""Synthetic Torch/Transformers tests; no pretrained weights or benchmark rows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from epistemic_geometry.backends.tiny import TinyRandomTransformerBackend  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.types import BenchmarkItem, Intervention, SteeringVector  # noqa: E402


@pytest.fixture()
def tiny_backend() -> TinyRandomTransformerBackend:
    backend = TinyRandomTransformerBackend(
        BackendConfig(
            type="tiny_transformer",
            model_id="TINY_RANDOM_GPT2_CONFIG_ONLY",
            hidden_size=32,
            device="cpu",
            dtype="float32",
            layer=0,
            layer_path="transformer.h",
            prompt_mode="plain",
            max_new_tokens=12,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            enable_thinking=False,
        ),
        seed=911,
    )
    backend.config = replace(backend.config, max_new_tokens=12)
    return backend


def test_token_stop_preserves_seeded_prefix_and_generation_config(tiny_backend) -> None:
    item = BenchmarkItem(id="synthetic-stop", prompt="alpha beta gamma", target="none")
    full = tiny_backend.generate_reasoning(item, sampling_seed=772, max_new_tokens=12)
    stopped = tiny_backend.generate_reasoning(
        item,
        sampling_seed=772,
        max_new_tokens=12,
        token_stop_predicate=lambda values: len(values) >= 5,
        token_stop_name="SYNTHETIC_LENGTH_FIVE",
    )
    assert stopped.metadata["generated_token_ids"] == full.metadata["generated_token_ids"][:5]
    assert stopped.metadata["generation_seed"] == full.metadata["generation_seed"] == 772
    assert stopped.metadata["generation"] == full.metadata["generation"]
    assert stopped.metadata["terminal_policy"] == {
        "name": "SYNTHETIC_LENGTH_FIVE",
        "triggered": True,
        "trigger_token_count": 5,
    }


def test_token_stop_preserves_sustained_hook_scope_and_cleanup(tiny_backend) -> None:
    item = BenchmarkItem(id="synthetic-hook", prompt="alpha beta gamma", target="none")
    vector = SteeringVector(
        values=np.linspace(-0.5, 0.5, 32),
        layer=0,
        constructor="synthetic",
        normalization="none",
        hash="synthetic-vector",
    )
    intervention = Intervention(0, 0.4, vector.hash, "last_token", vector)
    with tiny_backend.steer_sustained_current_token(intervention) as trace:
        output = tiny_backend.generate_reasoning(
            item,
            sampling_seed=773,
            max_new_tokens=12,
            token_stop_predicate=lambda values: len(values) >= 5,
            token_stop_name="SYNTHETIC_LENGTH_FIVE",
        )
    assert output.metadata["generated_token_count"] == 5
    assert trace["prefill_applications"] >= 1
    assert trace["prefill_applications"] + trace["decode_applications"] == trace["forward_count"]
    assert len(tiny_backend.layer_module(0)._forward_hooks) == 0
