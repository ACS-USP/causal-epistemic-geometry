from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.gate6_3 import (
    bank_geometry,
    single_layer_random_bank,
    standardized_delta,
    vector_sha256,
)


def test_single_layer_random_bank_is_deterministic_and_orthogonal() -> None:
    meaningful = np.arange(64, dtype=np.float64) + 1.0
    seeds = (11, 22, 33, 44)
    first = single_layer_random_bank(meaningful, seeds=seeds)
    second = single_layer_random_bank(meaningful, seeds=seeds)
    assert set(first) == {"R0", "R1", "R2", "R3"}
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    checks = bank_geometry(meaningful, first)
    assert checks["unit_norm_pass"] is True
    assert checks["meaningful_orthogonality_pass"] is True
    assert checks["random_pairwise_orthogonality_pass"] is True


def test_standardized_delta_matches_frozen_energy() -> None:
    direction = np.zeros(8, dtype=np.float64)
    direction[3] = 1.0
    delta = standardized_delta(direction, eta=2.5, reference_scale=4.0)
    assert np.linalg.norm(delta) == 10.0
    assert delta[3] == 10.0


def test_vector_hash_is_stable_for_float64_representation() -> None:
    vector = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    assert vector_sha256(vector) == vector_sha256(vector.copy())
