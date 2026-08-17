"""Actual Torch/Transformers hook integration tests; no pretrained weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.backends.qwen3_replay import (  # noqa: E402
    Qwen3CachedSuffixReplayEngine,
    SuffixReplayUnavailable,
)
from epistemic_geometry.backends.tiny import TinyRandomTransformerBackend  # noqa: E402
from epistemic_geometry.benchmarks.mock import MockBenchmark  # noqa: E402
from epistemic_geometry.benchmarks.reasoning.base import ReasoningView  # noqa: E402
from epistemic_geometry.benchmarks.reasoning.rollouts import (  # noqa: E402
    rollout_record_from_output,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.steering.constructors import difference_of_means  # noqa: E402
from epistemic_geometry.steering.vector import load_vector, save_vector  # noqa: E402
from epistemic_geometry.types import BenchmarkItem, Intervention, SteeringVector  # noqa: E402


class _BatchEncodingTokenizer:
    """Wrap the tiny tokenizer like the real Transformers tokenizer API."""

    def __init__(self, tokenizer):
        from transformers.tokenization_utils_base import BatchEncoding

        self._tokenizer = tokenizer
        self._batch_encoding = BatchEncoding

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._batch_encoding(self._tokenizer(*args, **kwargs))


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


def test_steer_preserves_choice_prompt_position_until_inference_finishes(tiny_backend) -> None:
    tiny_backend._choice_prompt_index = 2
    with tiny_backend.steer(_intervention(0.1, "last_token")):
        assert tiny_backend._choice_prompt_index == 2
    assert tiny_backend._choice_prompt_index is None


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


def test_seeded_reasoning_generation_records_raw_trajectory_and_parse_fields(tiny_backend) -> None:
    tiny_backend.config = replace(
        tiny_backend.config,
        enable_thinking=True,
        do_sample=True,
        max_new_tokens=3,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
    item = BenchmarkItem(id="reasoning-tiny", prompt="alpha beta gamma", target="3")
    torch.manual_seed(999)
    expected_after_generation = torch.rand(4)
    torch.manual_seed(999)
    output_a = tiny_backend.generate_reasoning(item, sampling_seed=123)
    output_b = tiny_backend.generate_reasoning(item, sampling_seed=123)
    actual_after_generation = torch.rand(4)
    assert output_a.raw_output == output_b.raw_output
    assert torch.equal(expected_after_generation, actual_after_generation)
    assert output_a.metadata["generation_seed"] == 123
    assert output_a.metadata["generated_token_ids"]

    view = ReasoningView(
        latent_id="MODREG-R:depth_4:tiny",
        view_id="MODREG-R:depth_4:tiny:canonical",
        family="MODREG-R",
        cell="depth_4",
        surface="canonical",
        answer=3,
        prompt="alpha beta gamma",
        prompt_hash=hashlib.sha256(b"alpha beta gamma").hexdigest(),
        template_hash="template",
    )
    record = rollout_record_from_output(
        view,
        output_a,
        intervention_id="baseline",
        rollout_index=0,
        sampling_seed=123,
        generation_config={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    )
    assert record.raw_text == output_a.raw_output
    assert record.token_ids == tuple(output_a.metadata["generated_token_ids"])
    assert record.metadata["model_revision"] == "local-config"
    view_output = tiny_backend.generate_reasoning_view(view, sampling_seed=123)
    assert view_output.metadata["view_id"] == view.view_id
    assert view_output.metadata["source_prompt_hash"] == view.prompt_hash


def test_batched_reasoning_preserves_per_row_seed_streams(tiny_backend) -> None:
    tiny_backend.config = replace(
        tiny_backend.config,
        enable_thinking=True,
        do_sample=True,
        max_new_tokens=5,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        # Right padding plus an explicit last-real-token gather is the
        # portable exact path for decoder-only batch prefill.  Left-padding
        # behavior remains a model-specific benchmark choice and is audited
        # separately on the real Qwen backend.
        padding_side="right",
    )
    views = [
        ReasoningView(
            latent_id="FSM-R:length_4:batch-a",
            view_id="FSM-R:length_4:batch-a:canonical",
            family="FSM-R",
            cell="length_4",
            surface="canonical",
            answer=3,
            prompt="alpha beta",
            prompt_hash=hashlib.sha256(b"alpha beta").hexdigest(),
            template_hash="template",
        ),
        ReasoningView(
            latent_id="FSM-R:length_4:batch-b",
            view_id="FSM-R:length_4:batch-b:canonical",
            family="FSM-R",
            cell="length_4",
            surface="canonical",
            answer=3,
            prompt="alpha beta gamma delta",
            prompt_hash=hashlib.sha256(b"alpha beta gamma delta").hexdigest(),
            template_hash="template",
        ),
    ]
    seeds = [101, 202]
    serial = [
        tiny_backend.generate_reasoning_view(view, sampling_seed=seed, max_new_tokens=5)
        for view, seed in zip(views, seeds, strict=True)
    ]
    batched = tiny_backend.generate_reasoning_batch(
        list(zip(views, seeds, strict=True)),
        max_new_tokens=5,
        batch_size=2,
        max_prefill_tokens=256,
    )
    assert [output.metadata["generated_token_ids"] for output in batched] == [
        output.metadata["generated_token_ids"] for output in serial
    ]
    shuffled = tiny_backend.generate_reasoning_batch(
        list(zip(reversed(views), reversed(seeds), strict=True)),
        max_new_tokens=5,
        batch_size=2,
        max_prefill_tokens=256,
    )
    assert shuffled[0].metadata["generated_token_ids"] == serial[1].metadata[
        "generated_token_ids"
    ]
    assert shuffled[1].metadata["generated_token_ids"] == serial[0].metadata[
        "generated_token_ids"
    ]


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


def test_prepare_choice_item_accepts_transformers_batch_encoding(tiny_backend) -> None:
    config = replace(tiny_backend.config, inference_mode="choice_loglikelihood")
    backend = HuggingFaceBackend(
        config,
        model=tiny_backend.model,
        tokenizer=_BatchEncodingTokenizer(tiny_backend.tokenizer),
        model_identifier="injected-batch-encoding",
        tokenizer_identifier="injected-batch-encoding",
    )
    item = BenchmarkItem(
        id="batch-encoding-choice",
        prompt="Choose one answer",
        target="A",
        metadata={"candidate_labels": ["A", "B"]},
    )
    prepared = backend.prepare_choice_item(item)
    assert prepared.prompt_length > 0
    assert prepared.candidate_token_ids["A"]
    assert isinstance(prepared.context_compatible_candidate_ids["A"], tuple)


def test_choice_log_softmax_promotes_logits_to_fp32_without_grad(tiny_backend) -> None:
    logits = torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.bfloat16)
    with torch.inference_mode():
        log_probs = tiny_backend._choice_log_softmax(logits)
    assert log_probs.dtype == torch.float32
    assert log_probs.requires_grad is False
    assert all(parameter.dtype == torch.float32 for parameter in tiny_backend.model.parameters())


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


def _choice_backend(
    tiny_backend, *, execution_mode: str, candidate_head_mode: str = "full_vocab_reference"
):
    tiny_backend.config = replace(
        tiny_backend.config,
        inference_mode="choice_loglikelihood",
        execution_mode=execution_mode,
        candidate_head_mode=candidate_head_mode,
        item_batch_size=2,
        condition_chunk_size=2,
        padding_side="left",
    )
    return tiny_backend


def _choice_items() -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            id="batch-short",
            prompt="Choose one answer alpha beta",
            target="A",
            metadata={"candidate_labels": ["A", "B", "C", "D"]},
        ),
        BenchmarkItem(
            id="batch-long",
            prompt="Choose one answer alpha beta gamma delta epsilon",
            target="B",
            metadata={"candidate_labels": ["A", "B", "C", "D"]},
        ),
    ]


def _serial_choice_output(backend, item, vector=None, alpha=0.0):
    if vector is None:
        return backend.predict(item)
    intervention = Intervention(0, alpha, vector.hash, "last_token", vector)
    with backend.steer(intervention):
        return backend.predict(item)


@pytest.mark.parametrize("execution_mode", ["full_prompt_batched", "cached_decode"])
def test_optimized_choice_engines_match_serial_predictions(tiny_backend, execution_mode) -> None:
    items = _choice_items()
    vector = SteeringVector(
        values=np.linspace(-0.4, 0.4, 32),
        layer=0,
        constructor="test",
        normalization="none",
        hash="batch-vector",
    )
    backend = _choice_backend(tiny_backend, execution_mode=execution_mode)
    prepared = backend.prepare_choice_items(items)
    conditions = [
        ({"condition": "baseline", "alpha": 0.0, "layer": 0}, None),
        ({"condition": "plus", "alpha": 0.8, "layer": 0}, vector),
        ({"condition": "minus", "alpha": -0.8, "layer": 0}, vector),
    ]
    optimized = backend.predict_choice_batch(prepared, conditions)
    optimized_by_key = {
        (item.item_id, spec["condition"]): output
        for item, spec, output in optimized
    }
    for item in items:
        for spec, condition_vector in conditions:
            serial = _serial_choice_output(
                backend, item, condition_vector, float(spec["alpha"])
            )
            actual = optimized_by_key[(item.id, spec["condition"])]
            assert actual.raw_output == serial.raw_output
            assert set(actual.metadata["candidate_scores"]) == set(
                serial.metadata["candidate_scores"]
            )
            for label, value in serial.metadata["candidate_scores"].items():
                assert actual.metadata["candidate_scores"][label] == pytest.approx(
                    value, abs=2e-5
                )
            assert actual.metadata["candidate_score_semantics"] == (
                "full_vocab_log_probability"
            )


def test_serial_shape_reference_matches_candidatewise_scores(tiny_backend) -> None:
    items = _choice_items()[:1]
    vector = SteeringVector(
        values=np.linspace(-0.4, 0.4, 32),
        layer=0,
        constructor="test",
        normalization="none",
        hash="serial-shape-vector",
    )
    backend = _choice_backend(tiny_backend, execution_mode="full_prompt_batched")
    backend.config = replace(
        backend.config,
        serial_shape_reference=True,
        item_batch_size=1,
        condition_chunk_size=1,
    )
    conditions = [
        ({"condition": "baseline", "alpha": 0.0, "layer": 0}, None),
        ({"condition": "plus", "alpha": 0.8, "layer": 0}, vector),
    ]
    outputs = backend.predict_choice_batch(backend.prepare_choice_items(items), conditions)
    by_key = {(item.item_id, spec["condition"]): output for item, spec, output in outputs}
    for spec, condition_vector in conditions:
        serial = _serial_choice_output(backend, items[0], condition_vector, float(spec["alpha"]))
        actual = by_key[(items[0].id, spec["condition"])]
        assert actual.raw_output == serial.raw_output
        for label, value in serial.metadata["candidate_scores"].items():
            assert actual.metadata["candidate_scores"][label] == pytest.approx(value, abs=1e-6)


def test_candidate_only_head_preserves_ranking_and_margins(tiny_backend) -> None:
    items = _choice_items()
    vector = SteeringVector(
        values=np.linspace(-0.7, 0.7, 32),
        layer=0,
        constructor="test",
        normalization="none",
        hash="candidate-head-vector",
    )
    conditions = [
        ({"condition": "baseline", "alpha": 0.0, "layer": 0}, None),
        ({"condition": "plus", "alpha": 1.1, "layer": 0}, vector),
    ]
    reference = _choice_backend(
        tiny_backend, execution_mode="cached_decode", candidate_head_mode="full_vocab_reference"
    )
    full = reference.predict_choice_batch(reference.prepare_choice_items(items), conditions)
    candidate = _choice_backend(
        tiny_backend, execution_mode="cached_decode", candidate_head_mode="candidate_only"
    )
    only = candidate.predict_choice_batch(candidate.prepare_choice_items(items), conditions)
    full_by_key = {(item.item_id, spec["condition"]): output for item, spec, output in full}
    only_by_key = {(item.item_id, spec["condition"]): output for item, spec, output in only}
    for key, full_output in full_by_key.items():
        only_output = only_by_key[key]
        full_scores = full_output.metadata["candidate_scores"]
        only_scores = only_output.metadata["candidate_scores"]
        assert only_output.raw_output == full_output.raw_output
        assert only_output.metadata["candidate_score_semantics"] == (
            "candidate_logits_no_vocab_normalization"
        )
        full_order = sorted(full_scores, key=full_scores.get, reverse=True)
        only_order = sorted(only_scores, key=only_scores.get, reverse=True)
        assert only_order == full_order
        assert (only_scores[full_order[0]] - only_scores[full_order[1]]) == pytest.approx(
            full_scores[full_order[0]] - full_scores[full_order[1]], abs=2e-5
        )


def test_full_prompt_right_padding_targets_each_real_last_token(tiny_backend) -> None:
    backend = _choice_backend(tiny_backend, execution_mode="full_prompt_batched")
    backend.config = replace(backend.config, padding_side="right")
    items = _choice_items()
    vector = SteeringVector(
        values=np.linspace(-0.5, 0.5, 32),
        layer=0,
        constructor="test",
        normalization="none",
        hash="right-pad-vector",
    )
    conditions = [({"condition": "plus", "alpha": 0.9, "layer": 0}, vector)]
    optimized = backend.predict_choice_batch(backend.prepare_choice_items(items), conditions)
    by_id = {item.id: item for item in items}
    for prepared, spec, output in optimized:
        serial = _serial_choice_output(
            backend, by_id[prepared.item_id], vector, float(spec["alpha"])
        )
        assert output.raw_output == serial.raw_output


def test_cached_decode_rejects_unsafe_right_padding(tiny_backend) -> None:
    backend = _choice_backend(tiny_backend, execution_mode="cached_decode")
    backend.config = replace(backend.config, padding_side="right")
    with pytest.raises(ValueError, match="left padding"):
        backend.predict_choice_batch(
            backend.prepare_choice_items(_choice_items()),
            [({"condition": "baseline", "alpha": 0.0, "layer": 0}, None)],
        )


def test_multitoken_choice_fallback_reuses_prefix_and_matches_serial(tiny_backend) -> None:
    backend = _choice_backend(tiny_backend, execution_mode="cached_decode")
    item = BenchmarkItem(
        id="multi-token-choice",
        prompt="Choose one answer alpha beta",
        target="A",
        metadata={"candidate_labels": ["A", "LONG LABEL"]},
    )
    prepared = backend.prepare_choice_item(item)
    assert not prepared.all_candidates_single_token
    vector = SteeringVector(
        values=np.linspace(-0.2, 0.2, 32),
        layer=0,
        constructor="test",
        normalization="none",
        hash="multitoken-vector",
    )
    condition = {"condition": "plus", "alpha": 0.4, "layer": 0}
    actual = backend.predict_choice_batch([(prepared)], [(condition, vector)])[0][2]
    serial = _serial_choice_output(backend, item, vector, 0.4)
    assert actual.raw_output == serial.raw_output
    assert actual.metadata["execution_engine"] == "cached_decode_multitoken"


def test_batched_activation_extraction_matches_itemwise_and_captures_layers(tiny_backend) -> None:
    items = _choice_items()
    batched = tiny_backend.extract_activations_batch(items, layers=[0, 1])
    assert set(batched) == {0, 1}
    assert batched[0].shape == (2, 32)
    assert batched[1].shape == (2, 32)
    for index, item in enumerate(items):
        assert np.allclose(batched[0][index], tiny_backend.extract_activation(item), atol=1e-6)


def test_cached_decode_forward_accounting_exposes_prefix_reuse(tiny_backend) -> None:
    backend = _choice_backend(tiny_backend, execution_mode="cached_decode")
    vector = SteeringVector(
        values=np.ones(32),
        layer=0,
        constructor="test",
        normalization="none",
        hash="accounting-vector",
    )
    conditions = [
        ({"condition": f"c{index}", "alpha": float(index), "layer": 0}, vector)
        for index in range(3)
    ]
    backend.reset_execution_stats()
    backend.predict_choice_batch(backend.prepare_choice_items(_choice_items()), conditions)
    stats = backend.execution_stats()
    assert stats["prefill_forwards"] == 1
    assert stats["decode_forwards"] == 2
    assert stats["forward_calls"] == 3


def test_heterogeneous_layers_are_grouped_without_row_contamination(tiny_backend) -> None:
    backend = _choice_backend(tiny_backend, execution_mode="cached_decode")
    items = _choice_items()
    vector_a = SteeringVector(np.ones(32), 0, "test", "none", hash="layer-a")
    vector_b = SteeringVector(np.full(32, 2.0), 1, "test", "none", hash="layer-b")
    conditions = [
        ({"condition": "layer-0", "alpha": 0.2, "layer": 0}, vector_a),
        ({"condition": "layer-1", "alpha": -0.3, "layer": 1}, vector_b),
    ]
    optimized = backend.predict_choice_batch(backend.prepare_choice_items(items), conditions)
    by_key = {(item.item_id, spec["condition"]): output for item, spec, output in optimized}
    for item in items:
        for spec, vector in conditions:
            serial = _serial_choice_output(backend, item, vector, float(spec["alpha"]))
            assert by_key[(item.id, spec["condition"])].raw_output == serial.raw_output


def test_qwen3_suffix_replay_guard_fails_closed_on_tiny_gpt2(tiny_backend) -> None:
    engine = Qwen3CachedSuffixReplayEngine(tiny_backend)
    assert engine.status.supported is False
    with pytest.raises(SuffixReplayUnavailable, match="not Qwen3"):
        engine.require_supported()
