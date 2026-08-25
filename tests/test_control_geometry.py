from __future__ import annotations

import numpy as np

from epistemic_geometry.analysis.control_geometry import (
    aggregate_directional_gram,
    categorical_fisher,
    directional_fisher_gram,
    effective_rank,
    gram_radii_angles_distances,
    jensen_shannon,
    kl_divergence,
    linear_r_squared,
    softmax,
    squared_hellinger,
)


def test_fisher_gram_is_psd_and_invariant_to_logit_shift() -> None:
    probabilities = softmax(np.asarray([0.3, -0.7, 1.2, 0.1]))
    tangents = np.asarray([[1.0, -0.2, 0.4, 0.8], [-0.5, 0.3, 1.1, -0.9]])
    gram = directional_fisher_gram(tangents, probabilities)
    explicit = tangents @ categorical_fisher(probabilities) @ tangents.T
    shifted = directional_fisher_gram(
        tangents + np.asarray([[7.0], [-3.0]]), probabilities
    )
    assert np.allclose(gram, explicit, atol=1e-14)
    assert np.allclose(gram, shifted, atol=1e-14)
    assert np.min(np.linalg.eigvalsh(gram)) >= -1e-14


def test_local_kl_hellinger_and_js_constants_match_fisher() -> None:
    logits = np.asarray([0.2, -0.4, 0.9, 0.1, -1.0])
    tangent = np.asarray([0.8, -0.2, 0.5, -1.1, 0.4])
    p0 = softmax(logits)
    q = float(tangent @ categorical_fisher(p0) @ tangent)
    epsilon = 1e-4
    p1 = softmax(logits + epsilon * tangent)
    assert np.isclose(2.0 * kl_divergence(p0, p1) / epsilon**2, q, rtol=3e-4)
    assert np.isclose(8.0 * squared_hellinger(p0, p1) / epsilon**2, q, rtol=3e-4)
    assert np.isclose(8.0 * jensen_shannon(p0, p1) / epsilon**2, q, rtol=3e-4)


def test_gram_radial_angular_identity_and_polarization() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    metric = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    gram = vectors @ metric @ vectors.T
    result = gram_radii_angles_distances(gram)
    expected = np.asarray(
        [
            [(left - right) @ metric @ (left - right) for right in vectors]
            for left in vectors
        ]
    )
    assert np.allclose(result["squared_distances"], expected)
    q0, q1, qsum = gram[0, 0], gram[1, 1], (vectors[0] + vectors[1]) @ metric @ (
        vectors[0] + vectors[1]
    )
    assert np.isclose((qsum - q0 - q1) / 2.0, gram[0, 1])
    assert effective_rank(gram) <= 2.0 + 1e-12


def test_aggregate_gram_and_linear_r_squared() -> None:
    probabilities = np.asarray([[0.4, 0.6], [0.7, 0.3]])
    tangents = np.asarray(
        [
            [[1.0, -1.0], [0.5, -0.5]],
            [[0.2, -0.2], [-1.0, 1.0]],
        ]
    )
    aggregate = aggregate_directional_gram(tangents, probabilities)
    expected = np.mean(
        [directional_fisher_gram(t, p) for t, p in zip(tangents, probabilities, strict=True)],
        axis=0,
    )
    assert np.allclose(aggregate, expected)
    x = np.arange(10, dtype=float)
    assert np.isclose(linear_r_squared(x, 3.0 + 2.0 * x), 1.0)
