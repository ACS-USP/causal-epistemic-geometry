"""Outcome-independent Gate-4 construction and estimands.

All functions in this module are CPU/numpy utilities.  They intentionally do
not know about model outputs or benchmark selection, which makes them safe to
unit-test before the real run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from epistemic_geometry.reproducibility import stable_seed


def _unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must have a finite non-zero norm")
    return array / norm


def construct_paired_direction(
    careful: np.ndarray,
    direct: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return unit careful-minus-direct direction, Delta, and raw mean diff."""

    positive = np.asarray(careful, dtype=np.float64)
    negative = np.asarray(direct, dtype=np.float64)
    if positive.ndim != 2 or negative.ndim != 2 or positive.shape != negative.shape:
        raise ValueError("careful/direct activations must be equally shaped 2-D arrays")
    raw = (positive - negative).mean(axis=0)
    direction = _unit(raw)
    gaps = (positive - negative) @ direction
    return direction, float(gaps.mean()), raw


def heldout_signed_gaps(
    direction: np.ndarray, careful: np.ndarray, direct: np.ndarray
) -> np.ndarray:
    """Compute held-out signed careful-minus-direct projections."""

    positive = np.asarray(careful, dtype=np.float64)
    negative = np.asarray(direct, dtype=np.float64)
    vector = _unit(direction)
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("held-out activations must be equally shaped 2-D arrays")
    return (positive - negative) @ vector


def random_orthogonal_direction(direction: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Draw a deterministic unit vector orthogonal to ``direction``."""

    vector = _unit(direction)
    rng = np.random.default_rng(stable_seed("MICRO-Q1-RANDOM-DIRECTION", seed, vector.size))
    draw = rng.normal(size=vector.size).astype(np.float64)
    draw -= vector * float(np.dot(draw, vector))
    control = _unit(draw)
    return control, float(np.dot(control, vector))


def vector_sha256(values: np.ndarray) -> str:
    """Hash canonical float64 vector bytes."""

    return hashlib.sha256(np.asarray(values, dtype=np.float64).reshape(-1).tobytes()).hexdigest()


def _errors(rows: Sequence[Sequence[int | bool]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.int8)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("each condition must provide a T x 2 error matrix")
    if not np.isin(array, [0, 1]).all():
        raise ValueError("errors must be binary")
    return array


def pair_estimands(
    baseline: Sequence[Sequence[int | bool]], condition: Sequence[Sequence[int | bool]]
) -> dict[str, float]:
    """Compute unbiased Gate-4 pair estimands for one intervention."""

    e0 = _errors(baseline)
    ej = _errors(condition)
    if e0.shape != ej.shape or e0.shape[0] < 2:
        raise ValueError("baseline and condition must have equal T x 2 shape with T >= 2")
    b00 = float(np.mean(e0[:, 0] * e0[:, 1]))
    bjj = float(np.mean(ej[:, 0] * ej[:, 1]))
    b0j = float(np.mean((e0[:, :, None] * ej[:, None, :]).reshape(-1)))
    q0 = e0.mean(axis=1, dtype=np.float64)
    qj = ej.mean(axis=1, dtype=np.float64)
    mask = ~np.eye(e0.shape[0], dtype=bool)
    u00 = float((q0[:, None] * q0[None, :])[mask].mean())
    u0j = float((q0[:, None] * qj[None, :])[mask].mean())
    r = float(np.mean((e0[:, :, None] * (1 - ej[:, None, :])).reshape(-1)))
    m = float(np.mean(((1 - e0[:, :, None]) * ej[:, None, :]).reshape(-1)))
    result = {
        "B00": b00,
        "Bjj": bjj,
        "B0j": b0j,
        "O00": 1.0 - b00,
        "O0j": 1.0 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": float(
            np.mean(
                e0[:, 0] * e0[:, 1]
                + ej[:, 0] * ej[:, 1]
                - e0[:, 0] * ej[:, 1]
                - e0[:, 1] * ej[:, 0]
            )
        ),
        "rescue": r,
        "damage": m,
        "accuracy_baseline": 1.0 - float(e0.mean()),
        "accuracy_condition": 1.0 - float(ej.mean()),
    }
    if not np.isclose(
        result["rescue"] - result["damage"],
        result["accuracy_condition"] - result["accuracy_baseline"],
    ):
        raise AssertionError("rescue minus damage identity failed")
    return result


def all_pair_estimands(
    baseline: Sequence[Sequence[int | bool]],
    plus: Sequence[Sequence[int | bool]],
    minus: Sequence[Sequence[int | bool]],
    random: Sequence[Sequence[int | bool]],
) -> dict[str, dict[str, float]]:
    """Return baseline and the three condition estimand dictionaries."""

    output = {
        "plus": pair_estimands(baseline, plus),
        "minus": pair_estimands(baseline, minus),
        "random": pair_estimands(baseline, random),
    }
    baseline_array = _errors(baseline)
    b00 = float(np.mean(baseline_array[:, 0] * baseline_array[:, 1]))
    output["baseline"] = {"B00": b00, "O00": 1.0 - b00, "resampling_gain": 0.0}
    for name in ("plus", "minus"):
        output[name]["Delta_G"] = output[name]["G"] - output["random"]["G"]
        output[name]["Delta_C"] = output[name]["C"] - output["random"]["C"]
        output[name]["Delta_D"] = output[name]["D"] - output["random"]["D"]
    return output


def bootstrap_pair_estimands(
    baseline: np.ndarray,
    conditions: dict[str, np.ndarray],
    *,
    resamples: int = 5000,
    seed: int = 20260819,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Item-cluster percentile bootstrap for the Gate-4 estimands."""

    e0 = _errors(baseline)
    arrays = {name: _errors(value) for name, value in conditions.items()}
    if any(value.shape != e0.shape for value in arrays.values()):
        raise ValueError("all bootstrap conditions must have the baseline shape")
    rng = np.random.default_rng(seed)
    names = tuple(arrays)
    metrics = {name: {key: [] for key in ("G", "C", "D", "rescue", "damage")} for name in names}
    for name in names:
        if name != "random":
            metrics[name].update({key: [] for key in ("Delta_G", "Delta_C", "Delta_D")})
    for _ in range(resamples):
        indices = rng.integers(0, e0.shape[0], size=e0.shape[0])
        sampled = {
            name: pair_estimands(e0[indices], value[indices]) for name, value in arrays.items()
        }
        for name, values in sampled.items():
            for key in metrics[name]:
                if key in values:
                    metrics[name][key].append(values[key])
        for name in ("plus", "minus"):
            if name not in sampled or "random" not in sampled:
                continue
            for key in ("Delta_G", "Delta_C", "Delta_D"):
                metrics[name][key].append(
                    sampled[name][key.replace("Delta_", "")]
                    - sampled["random"][key.replace("Delta_", "")]
                )
    return {
        name: {
            key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
            for key, values in entries.items()
        }
        for name, entries in metrics.items()
    }
