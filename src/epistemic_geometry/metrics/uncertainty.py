"""Small deterministic paired bootstrap summaries for development diagnostics."""

from __future__ import annotations

import numpy as np

from epistemic_geometry.metrics.complementarity import _paired_arrays
from epistemic_geometry.reproducibility import stable_seed
from epistemic_geometry.types import Prediction


def cluster_bootstrap_mean(
    values: np.ndarray | list[float],
    cluster_ids: np.ndarray | list[str],
    *,
    seed: int,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Bootstrap an equally weighted mean over nested scientific clusters.

    Test cases or rendered views inside one latent problem are reduced to one
    cluster mean before resampling. This prevents a problem with many redundant
    tests from masquerading as many independent scientific units.
    """

    observations = np.asarray(values, dtype=float).reshape(-1)
    clusters = np.asarray(cluster_ids, dtype=str).reshape(-1)
    if observations.size == 0 or observations.shape != clusters.shape:
        raise ValueError("values and cluster_ids must be non-empty and have equal shape")
    if np.any(~np.isfinite(observations)):
        raise ValueError("cluster bootstrap values must be finite")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be positive and confidence must lie in (0, 1)")
    unique = np.unique(clusters)
    cluster_means = np.asarray(
        [observations[clusters == cluster].mean() for cluster in unique],
        dtype=float,
    )
    rng = np.random.default_rng(
        stable_seed("cluster-bootstrap", seed, n_resamples, *unique.tolist())
    )
    samples = cluster_means[
        rng.integers(0, cluster_means.size, size=(n_resamples, cluster_means.size))
    ].mean(axis=1)
    lower = (1.0 - confidence) / 2.0 * 100
    upper = (1.0 + confidence) / 2.0 * 100
    return {
        "method": "equal_cluster_bootstrap_descriptive",
        "seed": seed,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "n_clusters": int(unique.size),
        "n_nested_observations": int(observations.size),
        "estimate": float(cluster_means.mean()),
        "interval": [
            float(np.percentile(samples, lower)),
            float(np.percentile(samples, upper)),
        ],
    }


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
    conditional_rescue_minus_damage: list[float] = []
    net_flip_fractions: list[float] = []
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
        conditional_rescue_minus_damage.append(
            rescue - damage
            if np.isfinite(rescue) and np.isfinite(damage)
            else float("nan")
        )
        net_flip_fractions.append(
            float(
                (
                    np.logical_and(base, ~treatment).sum()
                    - np.logical_and(~base, treatment).sum()
                )
                / n_items
            )
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
        "net_flip_fraction_interval": interval(net_flip_fractions),
        "conditional_rescue_minus_damage_interval": interval(
            conditional_rescue_minus_damage
        ),
        # Compatibility only. Conditional rescue and damage have different
        # denominators, so their difference is not a net accuracy effect.
        "rescue_minus_damage_interval": interval(conditional_rescue_minus_damage),
        "rescue_minus_damage_interval_status": (
            "DEPRECATED_CONDITIONAL_RATES_DIFFERENT_DENOMINATORS"
        ),
        "complementarity_headroom_interval": interval(headrooms),
    }
