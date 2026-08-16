"""Small deterministic paired bootstrap summaries for development diagnostics."""

from __future__ import annotations

import numpy as np

from epistemic_geometry.metrics.complementarity import _paired_arrays
from epistemic_geometry.reproducibility import stable_seed
from epistemic_geometry.types import Prediction


def bootstrap_paired_metrics(
    predictions: list[Prediction],
    seed: int,
    n_resamples: int = 200,
    confidence: float = 0.95,
    treatment_condition: str = "steered",
) -> dict[str, object]:
    """Bootstrap descriptive intervals over items; not a hypothesis test."""

    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be positive and confidence must lie in (0, 1)")
    base_errors, treatment_errors, _counts = _paired_arrays(
        predictions, treatment_condition=treatment_condition
    )
    n_items = base_errors.size
    rng = np.random.default_rng(stable_seed("bootstrap", seed, n_items, n_resamples))
    deltas: list[float] = []
    rescue_minus_damage: list[float] = []
    headrooms: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n_items, size=n_items)
        base = base_errors[indices]
        treatment = treatment_errors[indices]
        base_accuracy = float((~base).mean())
        treatment_accuracy = float((~treatment).mean())
        baseline_errors = int(base.sum())
        baseline_successes = int((~base).sum())
        rescue = (
            float(np.logical_and(base, ~treatment).sum() / baseline_errors)
            if baseline_errors
            else float("nan")
        )
        damage = (
            float(np.logical_and(~base, treatment).sum() / baseline_successes)
            if baseline_successes
            else float("nan")
        )
        oracle = float(np.logical_or(~base, ~treatment).mean())
        deltas.append(treatment_accuracy - base_accuracy)
        rescue_minus_damage.append(
            rescue - damage
            if np.isfinite(rescue) and np.isfinite(damage)
            else float("nan")
        )
        headrooms.append(oracle - max(base_accuracy, treatment_accuracy))

    lower = (1.0 - confidence) / 2.0 * 100
    upper = (1.0 + confidence) / 2.0 * 100

    def interval(values: list[float]) -> list[float | None]:
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            return [None, None]
        return [float(np.percentile(finite, lower)), float(np.percentile(finite, upper))]

    return {
        "method": "item_bootstrap_descriptive",
        "seed": seed,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "treatment_condition": treatment_condition,
        "delta_accuracy_interval": interval(deltas),
        "rescue_minus_damage_interval": interval(rescue_minus_damage),
        "complementarity_headroom_interval": interval(headrooms),
    }
