"""CPU-only primitives for directional categorical control geometry.

These functions define mathematical objects used in prospective design. They do
not load a model, evaluate semantic outcomes, or qualify a numerical engine.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return a float64 softmax along the final axis."""

    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def categorical_fisher(probabilities: np.ndarray) -> np.ndarray:
    """Return ``diag(p) - p p^T`` for one categorical distribution."""

    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if np.any(p <= 0.0) or not np.isclose(np.sum(p), 1.0, atol=1e-12):
        raise ValueError("probabilities must be positive and sum to one")
    return np.diag(p) - np.outer(p, p)


def directional_fisher_gram(
    logit_tangents: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Return the Fisher Gram matrix for rows of logit directional derivatives.

    If row ``i`` is ``J_z v_i``, the result is
    ``Gamma[i,j] = (J_z v_i)^T (diag(p)-pp^T) (J_z v_j)``.
    Centering the tangents under ``p`` avoids materializing the vocabulary-size
    Fisher matrix and makes invariance to a constant logit shift explicit.
    """

    tangents = np.asarray(logit_tangents, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if tangents.ndim != 2 or tangents.shape[1] != p.size:
        raise ValueError("tangents must be controller-by-vocabulary")
    if np.any(p <= 0.0) or not np.isclose(np.sum(p), 1.0, atol=1e-12):
        raise ValueError("probabilities must be positive and sum to one")
    centered = tangents - (tangents @ p)[:, None]
    gram = (centered * p[None, :]) @ centered.T
    return 0.5 * (gram + gram.T)


def aggregate_directional_gram(
    tangent_batches: np.ndarray, probability_batches: np.ndarray
) -> np.ndarray:
    """Average Fisher Gram matrices over probe/checkpoint rows."""

    tangents = np.asarray(tangent_batches, dtype=np.float64)
    probabilities = np.asarray(probability_batches, dtype=np.float64)
    if tangents.ndim != 3 or probabilities.ndim != 2:
        raise ValueError("expected row-by-controller-by-vocabulary tangents")
    if tangents.shape[0] != probabilities.shape[0] or tangents.shape[2] != probabilities.shape[1]:
        raise ValueError("tangent and probability batches are incompatible")
    grams = [
        directional_fisher_gram(row_tangents, row_probabilities)
        for row_tangents, row_probabilities in zip(tangents, probabilities, strict=True)
    ]
    return np.mean(grams, axis=0)


def gram_radii_angles_distances(gram: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose a positive-semidefinite Gram matrix into radii, angles, and distances."""

    values = np.asarray(gram, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("gram must be square")
    values = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("gram is not positive semidefinite")
    squared_radii = np.maximum(np.diag(values), 0.0)
    radii = np.sqrt(squared_radii)
    denominator = np.outer(radii, radii)
    cosine = np.full_like(values, np.nan)
    nonzero = denominator > 0.0
    cosine[nonzero] = np.clip(values[nonzero] / denominator[nonzero], -1.0, 1.0)
    squared_distances = np.maximum(
        squared_radii[:, None] + squared_radii[None, :] - 2.0 * values,
        0.0,
    )
    np.fill_diagonal(squared_distances, 0.0)
    return {
        "radii": radii,
        "cosine": cosine,
        "squared_distances": squared_distances,
        "distances": np.sqrt(squared_distances),
    }


def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return KL(left || right) for positive categorical distributions."""

    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def squared_hellinger(left: np.ndarray, right: np.ndarray) -> float:
    """Return ``H^2 = 1/2 ||sqrt(p)-sqrt(q)||_2^2``."""

    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    return float(0.5 * np.sum(np.square(np.sqrt(p) - np.sqrt(q))))


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Return equal-weight Jensen-Shannon divergence with natural logarithms."""

    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def effective_rank(gram: np.ndarray) -> float:
    """Return participation-ratio effective rank of a PSD Gram matrix."""

    eigenvalues = np.maximum(np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64)), 0.0)
    denominator = float(np.sum(np.square(eigenvalues)))
    if denominator == 0.0:
        return 0.0
    return float(np.square(np.sum(eigenvalues)) / denominator)


def linear_r_squared(features: np.ndarray, target: np.ndarray) -> float:
    """Return in-sample affine least-squares R^2 for a design diagnostic."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != y.size or y.size < 3:
        raise ValueError("features and target must contain matching rows")
    design = np.column_stack([np.ones(y.size), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    total = y - np.mean(y)
    denominator = float(total @ total)
    return 0.0 if denominator == 0.0 else float(1.0 - (residual @ residual) / denominator)


__all__ = [
    "aggregate_directional_gram",
    "categorical_fisher",
    "directional_fisher_gram",
    "effective_rank",
    "gram_radii_angles_distances",
    "jensen_shannon",
    "kl_divergence",
    "linear_r_squared",
    "softmax",
    "squared_hellinger",
]
