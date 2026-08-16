import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from epistemic_geometry.backends.tiny import TinyRandomTransformerBackend  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.inference.autotune import (  # noqa: E402
    benchmark_batch_grid,
    choose_fastest_safe,
)
from epistemic_geometry.types import BenchmarkItem  # noqa: E402


def test_autotuner_restores_config_and_selects_safe_result() -> None:
    config = BackendConfig(
        type="tiny_transformer",
        model_id="tiny",
        hidden_size=32,
        device="cpu",
        dtype="float32",
        layer=0,
        layer_path="transformer.h",
        prompt_mode="plain",
        inference_mode="choice_loglikelihood",
        execution_mode="cached_decode",
        item_batch_size=2,
        condition_chunk_size=2,
    )
    backend = TinyRandomTransformerBackend(config, seed=19)
    items = [
        BenchmarkItem(
            id=f"auto-{index}",
            prompt="Choose A alpha " * (index + 1),
            target="A",
            metadata={"candidate_labels": ["A", "B"]},
        )
        for index in range(2)
    ]
    prepared = backend.prepare_choice_items(items)
    rows = benchmark_batch_grid(
        backend,
        prepared,
        [({"condition": "baseline", "alpha": 0.0, "layer": 0}, None)],
        item_batch_sizes=(1, 2),
        condition_chunk_sizes=(1, 2),
    )
    assert len(rows) == 4
    assert backend.config == config
    assert choose_fastest_safe(rows)["status"] == "PASS"
