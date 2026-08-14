"""Pure functions for binary error-vector summaries."""

from __future__ import annotations

import numpy as np


def _binary(values: np.ndarray | list[bool]) -> np.ndarray:
    array = np.asarray(values, dtype=bool).reshape(-1)
    return array


def accuracy(correct: np.ndarray | list[bool]) -> float:
    """Return the fraction of correct paired predictions."""

    values = _binary(correct)
    return float(values.mean()) if values.size else float("nan")


def phi_correlation(errors_a: np.ndarray | list[bool], errors_b: np.ndarray | list[bool]) -> float:
    """Return Pearson/phi correlation with explicit constant-vector conventions.

    If both error vectors are identical constants, the result is 1.0. If only
    one is constant, or both constants differ, the result is 0.0 rather than
    propagating an undefined NaN into a run summary.
    """

    a = _binary(errors_a).astype(float)
    b = _binary(errors_b).astype(float)
    if a.size != b.size or a.size == 0:
        raise ValueError("Error vectors must be non-empty and have equal length")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a_centered, b_centered) / denominator)


def error_jaccard(errors_a: np.ndarray | list[bool], errors_b: np.ndarray | list[bool]) -> float:
    """Return error-set Jaccard similarity; empty union is defined as 1.0."""

    a = _binary(errors_a)
    b = _binary(errors_b)
    if a.size != b.size:
        raise ValueError("Error vectors must have equal length")
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(intersection / union)


def double_fault(errors_a: np.ndarray | list[bool], errors_b: np.ndarray | list[bool]) -> float:
    """Return ``P(e_a=1 and e_b=1)``."""

    a = _binary(errors_a)
    b = _binary(errors_b)
    if a.size != b.size or a.size == 0:
        raise ValueError("Error vectors must be non-empty and have equal length")
    return float(np.logical_and(a, b).mean())

