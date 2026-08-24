"""Outcome-free geometry estimators for the controller-held-out Q2 pilot."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _unit_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("controller vectors must be a matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("controller vectors must have finite positive norms")
    return matrix / norms[:, None]


def flat_geometry(vectors: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return cosine and normalized-Euclidean distance matrices."""

    unit = _unit_rows(vectors)
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    cosine_distance = 1.0 - cosine
    normalized_euclidean = np.sqrt(np.maximum(2.0 * cosine_distance, 0.0))
    np.fill_diagonal(cosine_distance, 0.0)
    np.fill_diagonal(normalized_euclidean, 0.0)
    identity_error = float(
        np.max(np.abs(np.square(normalized_euclidean) - 2.0 * cosine_distance))
    )
    return {
        "cosine_distance": cosine_distance,
        "normalized_euclidean": normalized_euclidean,
        "algebraic_identity_max_error": identity_error,
    }


@dataclass(frozen=True)
class WhitenedFit:
    """Low-rank representation of a prospectively regularized covariance."""

    mean: np.ndarray
    right_singular_vectors: np.ndarray
    eigenvalues: np.ndarray
    isotropic_variance: float
    regularization_fraction: float
    regularization_value: float
    effective_rank: float
    condition_number: float
    fit_hash: str


def fit_whitening(
    activations: np.ndarray,
    *,
    regularization_fraction: float = 0.10,
) -> WhitenedFit:
    """Fit Sigma_lambda=(1-lambda)Sigma+lambda*meanvar*I without labels."""

    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("whitening requires at least three activation rows")
    if not 0.0 < regularization_fraction <= 1.0:
        raise ValueError("regularization fraction must lie in (0, 1]")
    mean = values.mean(axis=0)
    centered = values - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = np.square(singular) / (len(values) - 1)
    total_variance = float(np.sum(np.var(values, axis=0, ddof=1)))
    mean_variance = total_variance / values.shape[1]
    ridge = regularization_fraction * mean_variance
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("whitening covariance is numerically degenerate")
    shrunk = (1.0 - regularization_fraction) * eigenvalues
    adjusted = shrunk + ridge
    condition = float((float(np.max(adjusted)) if len(adjusted) else ridge) / ridge)
    effective_rank = float(
        np.square(np.sum(eigenvalues)) / np.sum(np.square(eigenvalues))
    ) if np.any(eigenvalues) else 0.0
    digest = hashlib.sha256()
    for array in (mean, vt, eigenvalues):
        digest.update(str(array.shape).encode())
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    digest.update(np.asarray([regularization_fraction, ridge], dtype=np.float64).tobytes())
    return WhitenedFit(
        mean=mean,
        right_singular_vectors=vt,
        eigenvalues=eigenvalues,
        isotropic_variance=mean_variance,
        regularization_fraction=regularization_fraction,
        regularization_value=ridge,
        effective_rank=effective_rank,
        condition_number=condition,
        fit_hash=digest.hexdigest(),
    )


def whitened_inner_products(vectors: np.ndarray, fit: WhitenedFit) -> np.ndarray:
    """Apply the inverse regularized covariance through its low-rank eigensystem."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != fit.mean.size:
        raise ValueError("vectors and whitening fit occupy different spaces")
    ridge = fit.regularization_value
    basis = fit.right_singular_vectors
    projected = values @ basis.T
    eigen_adjusted = (
        (1.0 - fit.regularization_fraction) * fit.eigenvalues + ridge
    )
    # ridge^-1 I plus the correction within the observed covariance span.
    gram = (values @ values.T) / ridge
    correction = (1.0 / eigen_adjusted) - (1.0 / ridge)
    gram += (projected * correction[None, :]) @ projected.T
    return 0.5 * (gram + gram.T)


def whitened_geometry(vectors: np.ndarray, fit: WhitenedFit) -> dict[str, np.ndarray]:
    """Return normalized distances in the frozen inverse-covariance inner product."""

    gram = whitened_inner_products(vectors, fit)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    if np.any(norms <= 0):
        raise ValueError("one or more controllers has zero whitened norm")
    cosine = np.clip(gram / np.outer(norms, norms), -1.0, 1.0)
    cosine_distance = 1.0 - cosine
    normalized_euclidean = np.sqrt(np.maximum(2.0 * cosine_distance, 0.0))
    np.fill_diagonal(cosine_distance, 0.0)
    np.fill_diagonal(normalized_euclidean, 0.0)
    return {
        "cosine_distance": cosine_distance,
        "normalized_euclidean": normalized_euclidean,
    }


def _js_from_logits(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_log = left64 - np.logaddexp.reduce(left64)
    right_log = right64 - np.logaddexp.reduce(right64)
    mixture_log = np.logaddexp(left_log, right_log) - np.log(2.0)
    left_p = np.exp(left_log)
    right_p = np.exp(right_log)
    return float(
        0.5 * np.sum(left_p * (left_log - mixture_log))
        + 0.5 * np.sum(right_p * (right_log - mixture_log))
    )


def finite_secant_geometry(
    logits: Mapping[str, np.ndarray],
    controller_ids: Sequence[str],
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Aggregate full-vocabulary pairwise JS over frozen probe/checkpoint rows.

    Each array has shape ``(probe_checkpoint, vocabulary)``. The primary M2
    distance is ``sqrt(mean(JS))``, matching the prospectively frozen squared
    secant definition while remaining on a distance scale.
    """

    names = tuple(controller_ids)
    if set(logits) != set(names):
        raise ValueError("finite-secant archive does not match the controller bank")
    shapes = {np.asarray(logits[name]).shape for name in names}
    if len(shapes) != 1:
        raise ValueError("finite-secant controller arrays have unequal shapes")
    shape = shapes.pop()
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("finite-secant logits require checkpoint-by-vocabulary matrices")
    mean_js = np.zeros((len(names), len(names)), dtype=np.float64)
    for left_index, left_name in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right_name = names[right_index]
            values = [
                _js_from_logits(left_row, right_row)
                for left_row, right_row in zip(
                    np.asarray(logits[left_name]), np.asarray(logits[right_name]), strict=True
                )
            ]
            mean_js[left_index, right_index] = float(np.mean(values))
            mean_js[right_index, left_index] = mean_js[left_index, right_index]
    distance = np.sqrt(np.maximum(mean_js, 0.0))
    return {
        "mean_js": mean_js,
        "sqrt_mean_js": distance,
        "metadata": {
            "aggregation": "equal-weight mean over frozen probe/checkpoint rows",
            "distance": "sqrt(mean full-vocabulary Jensen-Shannon divergence)",
            "probe_checkpoint_rows": int(shape[0]),
            "vocabulary_size": int(shape[1]),
            "output_reduction_dtype": "float64",
        },
    }


def matrix_checks(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    return {
        "square": values.ndim == 2 and values.shape[0] == values.shape[1],
        "finite": bool(np.all(np.isfinite(values))),
        "symmetry_max_error": float(np.max(np.abs(values - values.T))),
        "diagonal_max_absolute": float(np.max(np.abs(np.diag(values)))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


__all__ = [
    "WhitenedFit",
    "finite_secant_geometry",
    "fit_whitening",
    "flat_geometry",
    "matrix_checks",
    "whitened_geometry",
    "whitened_inner_products",
]
