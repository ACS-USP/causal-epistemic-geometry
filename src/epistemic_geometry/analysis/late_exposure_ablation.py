"""Outcome-free planning primitives for the D75 late-exposure ablation.

The proposed scientific comparison is between a trajectory with D75 active for
all 4096 generated-token predictions and one that shares D75 through a fixed
2048-token prefix, then has D75 disabled.  This module intentionally operates
only on synthetic binary arrays supplied by its caller.  It neither materializes
items nor reads any empirical journal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True, slots=True)
class PlanningCell:
    """One outcome-free operating point for the paired decision rule."""

    n_latents: int
    p_early_stop: float
    late_exposure_harm: float
    shared_randomness: float


def _unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def latent_contrasts(
    early_stop: np.ndarray | Iterable[Iterable[int]],
    full_exposure: np.ndarray | Iterable[Iterable[int]],
) -> np.ndarray:
    """Return one unconditional correctness contrast per latent.

    Arrays are ``[latent, rollout]`` and must contain exactly two binary
    rollouts.  A positive contrast means that turning D75 off after the fixed
    prefix improved delivery correctness.  No filter is applied to unfinished
    trajectories; their correctness value is simply zero.
    """

    early = np.asarray(early_stop, dtype=float)
    full = np.asarray(full_exposure, dtype=float)
    if early.ndim != 2 or early.shape[1] != 2 or early.shape[0] == 0:
        raise ValueError("early_stop must have shape [positive_latents, 2]")
    if full.shape != early.shape:
        raise ValueError("full_exposure must have the same shape as early_stop")
    if not np.all(np.isin(early, (0.0, 1.0))) or not np.all(np.isin(full, (0.0, 1.0))):
        raise ValueError("correctness arrays must contain only 0 and 1")
    return (early - full).mean(axis=1)


def paired_evalues(contrasts: np.ndarray | Iterable[float], *, delta: float) -> dict[str, float]:
    """Return fixed-sample e-values for and against relevant late harm.

    Each contrast lies in ``[-1, 1]``.  The first e-value can support a mean
    improvement above ``delta`` for switching D75 off; the second can exclude
    a mean improvement at least ``delta``.  They are valid under independent
    latent items.  Threshold allocation remains a protocol decision.
    """

    values = np.asarray(tuple(contrasts), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("contrasts must be a non-empty finite vector")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("contrasts must lie in [-1, 1]")
    relevant = _unit_interval(delta, name="delta")
    support_factors = 1.0 + (values - relevant) / 2.0
    exclude_factors = 1.0 + (relevant - values) / 2.0
    return {
        "support_relevant_late_harm": float(prod(support_factors)),
        "exclude_relevant_late_harm": float(prod(exclude_factors)),
    }


def simulate_pair(
    cell: PlanningCell,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate paired two-rollout correctness without empirical calibration.

    ``shared_randomness`` is a transparent sensitivity parameter: a proportion
    of rollout pairs use the same uniform draw, while the remainder use
    independent uniforms.  Under shared draws, full exposure has probability
    ``p_early_stop - late_exposure_harm`` and is nested in early-stop success.
    This is a planning model, not a model of Qwen behaviour.
    """

    n_latents = _positive_int(cell.n_latents, name="n_latents")
    p_early = _unit_interval(cell.p_early_stop, name="p_early_stop")
    harm = _unit_interval(cell.late_exposure_harm, name="late_exposure_harm")
    shared = _unit_interval(cell.shared_randomness, name="shared_randomness")
    if harm > p_early:
        raise ValueError("late_exposure_harm cannot exceed p_early_stop")
    p_full = p_early - harm
    shape = (n_latents, 2)
    shared_mask = rng.random(shape) < shared
    shared_uniform = rng.random(shape)
    early_uniform = np.where(shared_mask, shared_uniform, rng.random(shape))
    full_uniform = np.where(shared_mask, shared_uniform, rng.random(shape))
    return (early_uniform < p_early).astype(int), (full_uniform < p_full).astype(int)


def estimate_operating_characteristics(
    cells: Iterable[PlanningCell],
    *,
    delta: float,
    threshold: float,
    replications: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Estimate a complete, caller-chosen planning grid with fixed simulation seed."""

    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 1.0:
        raise ValueError("threshold must be finite and greater than one")
    reps = _positive_int(replications, name="replications")
    rng = np.random.default_rng(seed)
    results: list[dict[str, float | int]] = []
    for cell in cells:
        support = 0
        exclude = 0
        contrast_sum = 0.0
        for _ in range(reps):
            early, full = simulate_pair(cell, rng=rng)
            contrasts = latent_contrasts(early, full)
            evalues = paired_evalues(contrasts, delta=delta)
            contrast_sum += float(contrasts.mean())
            support += evalues["support_relevant_late_harm"] >= threshold
            exclude += evalues["exclude_relevant_late_harm"] >= threshold
        results.append(
            {
                "n_latents": cell.n_latents,
                "p_early_stop": cell.p_early_stop,
                "late_exposure_harm": cell.late_exposure_harm,
                "shared_randomness": cell.shared_randomness,
                "replications": reps,
                "mean_observed_contrast": contrast_sum / reps,
                "support_rate": support / reps,
                "exclude_rate": exclude / reps,
            }
        )
    return results
