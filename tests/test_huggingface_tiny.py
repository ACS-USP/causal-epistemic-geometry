"""Actual Torch/Transformers hook integration tests; no pretrained weights."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.backends.tiny import TinyRandomTransformerBackend  # noqa: E402
from epistemic_geometry.benchmarks.mock import MockBenchmark  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.steering.constructors import difference_of_means  # noqa: E402
from epistemic_geometry.steering.vector import load_vector, save_vector  # noqa: E402
from epistemic_geometry.types import BenchmarkItem, Intervention, SteeringVector  # noqa: E402


@pytest.fixture()
def tiny_backend() -> TinyRandomTransformerBackend:
    return TinyRandomTransformerBackend(
        BackendConfig(
            type="tiny_transformer",
            model_id="TINY_RANDOM_GPT2_CONFIG_ONLY",
            hidden_size=32,
            device="cpu",
            dtype="float32",
            layer=0,
            layer_path="transformer.h",
            prompt_mode="plain",
            max_new_tokens=1,
        ),
        seed=19,
    )


def _ids(backend: TinyRandomTransformerBackend):
    return backend.tokenizer("alpha beta gamma", return_tensors="pt")


def _intervention(alpha: float, scope: str, values: np.ndarray | None = None) -> Intervention:
    vector = SteeringVector(
        values=np.ones(32) if values is None else values,
        layer=0,
        constructor="test",
        normalization="none",
        hash="test-vector",
    )
    return Intervention(0, alpha, "test-vector", scope, vector)


def _layer_hidden(backend: TinyRandomTransformerBackend, intervention: Intervention | None = None):
    captured = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden.detach().clone())
        return output

    context = backend.steer(intervention) if intervention is not None else _null_context()
    with context:
        handle = backend.layer_module(0).register_forward_hook(capture)
        try:
            with torch.inference_mode():
                backend.model(**_ids(backend), use_cache=False)
        finally:
            handle.remove()
    return captured[0]


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return False


def test_alpha_zero_and_zero_vector_are_identity(tiny_backend) -> None:
    ids = _ids(tiny_backend)
    with torch.inference_mode():
        baseline = tiny_backend.model(**ids, use_cache=False).logits
        with tiny_backend.steer(_intervention(0.0, "all_tokens")):
            alpha_zero = tiny_backend.model(**ids, use_cache=False).logits
        with tiny_backend.steer(_intervention(1.5, "all_tokens", np.zeros(32))):
            vector_zero = tiny_backend.model(**ids, use_cache=False).logits
    assert torch.equal(baseline, alpha_zero)
    assert torch.equal(baseline, vector_zero)


def test_exact_last_token_shift_and_token_isolation(tiny_backend) -> None:
    baseline = _layer_hidden(tiny_backend)
    vector = np.arange(32, dtype=np.float64) / 10
    alpha = 0.25
    steered = _layer_hidden(tiny_backend, _intervention(alpha, "last_token", vector))
    expected = torch.tensor(vector, dtype=baseline.dtype) * alpha
    assert torch.allclose(steered[:, :-1, :], baseline[:, :-1, :])
    assert torch.allclose(steered[:, -1, :] - baseline[:, -1, :], expected)


def test_all_tokens_shift_and_tuple_output_path(tiny_backend) -> None:
    baseline = _layer_hidden(tiny_backend)
    vector = np.full(32, 0.125)
    steered = _layer_hidden(tiny_backend, _intervention(-0.5, "all_tokens", vector))
    expected = torch.tensor(vector, dtype=baseline.dtype) * -0.5
    assert torch.allclose(steered - baseline, expected.view(1, 1, -1).expand_as(baseline))


def test_hook_cleanup_and_repeated_contexts_do_not_accumulate(tiny_backend) -> None:
    ids = _ids(tiny_backend)
    vector_a = _intervention(0.2, "last_token", np.ones(32))
    vector_b = _intervention(-0.3, "last_token", np.full(32, 2.0))
    with torch.inference_mode():
        clean_a = tiny_backend.model(**ids, use_cache=False).logits
        with tiny_backend.steer(vector_a):
            _ = tiny_backend.model(**ids, use_cache=False).logits
        clean_b = tiny_backend.model(**ids, use_cache=False).logits
        with tiny_backend.steer(vector_b):
            _ = tiny_backend.model(**ids, use_cache=False).logits
        clean_c = tiny_backend.model(**ids, use_cache=False).logits
    assert torch.equal(clean_a, clean_b)
    assert torch.equal(clean_b, clean_c)
    assert len(tiny_backend.layer_module(0)._forward_hooks) == 0


def test_dimension_and_layer_errors_are_loud(tiny_backend) -> None:
    bad = SteeringVector(np.zeros(31), 0, "bad", "none", hash="bad")
    with pytest.raises(ValueError, match="dimension"):
        with tiny_backend.steer(Intervention(0, 1.0, "bad", "last_token", bad)):
            pass
    with pytest.raises(ValueError, match="outside"):
        tiny_backend.layer_module(99)
    bad_config = replace(tiny_backend.config, layer_path="missing.layer.path")
    with pytest.raises(RuntimeError, match="Could not locate"):
        HuggingFaceBackend(
            bad_config,
            model=tiny_backend.model,
            tokenizer=tiny_backend.tokenizer,
            model_identifier="injected",
            tokenizer_identifier="injected",
        )


def test_activation_extraction_is_deterministic_and_graph_free(tiny_backend) -> None:
    item = MockBenchmark(1, seed=8).items()[0]
    first = tiny_backend.extract_activation(item)
    second = tiny_backend.extract_activation(item)
    assert first.shape == (32,)
    assert np.array_equal(first, second)
    assert first.dtype == np.float32


def test_production_inference_has_no_grad_graph(tiny_backend) -> None:
    item = MockBenchmark(1, seed=8).items()[0]
    encoded = tiny_backend.tokenizer(item.prompt, return_tensors="pt")
    with torch.inference_mode():
        output = tiny_backend.model(**encoded, use_cache=False)
    assert output.logits.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in tiny_backend.model.parameters())


def test_choice_loglikelihood_scores_complete_candidates_and_targets_prompt_token(
    tiny_backend,
) -> None:
    tiny_backend.config = replace(tiny_backend.config, inference_mode="choice_loglikelihood")
    item = BenchmarkItem(
        id="choice-1",
        prompt="Choose one answer",
        target="A",
        metadata={"candidate_labels": ["A", "LONG"]},
    )
    baseline = tiny_backend.predict(item)
    assert baseline.raw_output in {"A", "LONG"}
    assert set(baseline.metadata["candidate_scores"]) == {"A", "LONG"}
    assert all(np.isfinite(value) for value in baseline.metadata["candidate_scores"].values())
    assert baseline.metadata["candidate_token_counts"]["LONG"] >= 1
    vector = _intervention(0.1, "last_token")
    with tiny_backend.steer(vector):
        steered = tiny_backend.predict(item)
    assert set(steered.metadata["candidate_scores"]) == {"A", "LONG"}


def test_last_non_padding_activation_policy(tiny_backend) -> None:
    base_tokenizer = tiny_backend.tokenizer

    class LeftPaddedTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        eos_token = "<eos>"
        pad_token = "<pad>"

        def __call__(self, _text, return_tensors="pt"):
            ids = base_tokenizer("alpha beta", return_tensors=return_tensors)
            ids["input_ids"] = torch.cat(
                [torch.zeros((1, 1), dtype=torch.long), ids["input_ids"]], dim=1
            )
            ids["attention_mask"] = torch.cat(
                [torch.zeros((1, 1), dtype=torch.long), ids["attention_mask"]], dim=1
            )
            return ids

    tiny_backend.tokenizer = LeftPaddedTokenizer()
    item = MockBenchmark(1, seed=8).items()[0]
    activation = tiny_backend.extract_activation(item)
    encoded = tiny_backend._encode_item(item)[0]
    captured = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden.detach())
        return output

    handle = tiny_backend.layer_module(0).register_forward_hook(capture)
    try:
        with torch.inference_mode():
            tiny_backend.model(**encoded, use_cache=False)
    finally:
        handle.remove()
    last_index = int(encoded["attention_mask"][0].sum().item()) - 1
    assert np.allclose(activation, captured[0][0, last_index, :].numpy())


def test_difference_of_means_tiny_vector_provenance_and_roundtrip(tiny_backend, tmp_path) -> None:
    benchmark = MockBenchmark(4, seed=8)
    vector = difference_of_means(
        tiny_backend,
        benchmark.items()[:2],
        benchmark.items()[2:],
        layer=0,
        seed=8,
    )
    assert vector.dimension == 32
    assert np.isclose(np.linalg.norm(vector.values), 1.0)
    assert vector.metadata["creation_seed"] == 8
    assert vector.metadata["source_item_ids"] == [item.id for item in benchmark.items()]
    assert vector.metadata["model_provenance"]["model_revision"] == "local-config"
    assert vector.hash
    vector_path, metadata_path = save_vector(
        vector, tmp_path / "tiny.npz", git_commit="test-commit"
    )
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["source_item_ids"] == [item.id for item in benchmark.items()]
    assert saved_metadata["model_identifier"] == "tiny-random-gpt2-config"
    assert saved_metadata["model_revision"] == "local-config"
    assert saved_metadata["git_commit"] == "test-commit"
    restored = load_vector(vector_path, metadata_path)
    assert np.array_equal(vector.values, restored.values)
    assert restored.hash == vector.hash


def test_injected_backend_provenance_is_explicit(tiny_backend) -> None:
    provenance = tiny_backend.provenance()
    assert provenance["injected_test_model"] is True
    assert provenance["num_layers"] == 2
    assert provenance["hidden_size"] == 32
    assert provenance["model_revision"] == "local-config"
