import numpy as np

from epistemic_geometry.backends.mock import MockBackend
from epistemic_geometry.benchmarks.mock import MockBenchmark
from epistemic_geometry.types import Intervention


def test_mock_representation_is_deterministic_and_intervention_is_literal() -> None:
    benchmark = MockBenchmark(n_items=3, seed=11)
    item = benchmark.items()[0]
    backend = MockBackend(seed=11)
    vector = backend.fixture_vector("random", seed=7)
    intervention = Intervention(
        layer=0,
        alpha=0.75,
        vector_id=vector.hash,
        token_scope="all_tokens",
        vector=vector,
    )

    first = backend.extract_activation(item)
    second = backend.extract_activation(item)
    assert np.array_equal(first, second)
    assert np.allclose(backend.activation_for(item, intervention), first + 0.75 * vector.values)


def test_mock_steering_context_restores_baseline_state() -> None:
    benchmark = MockBenchmark(n_items=2, seed=12)
    item = benchmark.items()[0]
    backend = MockBackend(seed=12)
    vector = backend.fixture_vector("destructive")
    intervention = Intervention(0, 1.0, vector.hash, "last_token", vector)

    baseline_before = backend.predict(item)
    with backend.steer(intervention):
        treated = backend.predict(item)
        assert treated.metadata["intervened"] is True
    baseline_after = backend.predict(item)

    assert baseline_before.raw_output == baseline_after.raw_output
    assert baseline_before.metadata["intervened"] is False
    assert treated.metadata["intervened"] is True

