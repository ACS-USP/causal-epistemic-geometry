"""Dependence-aware planning for the Q1 LiveCodeBench Amendment 1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epistemic_geometry.experiments import q1_second_task_power as original


def unequal_cluster_design_effect(family_sizes: Sequence[int], rho: float) -> float:
    """Kish-style design effect for a row-weighted clustered mean."""

    sizes = np.asarray(family_sizes, dtype=np.float64)
    if sizes.ndim != 1 or len(sizes) == 0 or np.any(sizes < 1):
        raise ValueError("family sizes must be a nonempty positive vector")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0,1]")
    mean = float(sizes.mean())
    cv2 = float((sizes.std(ddof=0) / mean) ** 2)
    return 1.0 + (((1.0 + cv2) * mean) - 1.0) * rho


def family_average_variance_factor(family_sizes: Sequence[int], rho: float) -> float:
    """Variance factor for an equal-family mean of all observed tests."""

    sizes = np.asarray(family_sizes, dtype=np.float64)
    if sizes.ndim != 1 or len(sizes) == 0 or np.any(sizes < 1):
        raise ValueError("family sizes must be a nonempty positive vector")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0,1]")
    return float(np.mean(rho + (1.0 - rho) / sizes))


def _simulate(
    *,
    design: str,
    independent_units: int,
    raw_rows: int,
    rollouts: int,
    random_controls: int,
    transfer_fraction: float,
    se_multiplier: float,
    rho: float | None,
    replicates: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    rng = np.random.default_rng(seed)
    se_c = original.scaled_se(
        original.HISTORICAL_C_INTERVAL, n=independent_units, rollouts=rollouts
    ) * se_multiplier
    se_delta = original.scaled_se(
        original.HISTORICAL_DELTA_INTERVAL, n=independent_units, rollouts=rollouts
    ) * se_multiplier
    null_mean = float(original.HISTORICAL_NULL_C.mean())
    true_delta = float(transfer_fraction * original.HISTORICAL_DELTA_C)
    true_c = null_mean + true_delta
    z_c = rng.normal(size=replicates)
    z_delta = original.TEST_CORRELATION * z_c + np.sqrt(
        1.0 - original.TEST_CORRELATION**2
    ) * rng.normal(size=replicates)
    c_estimate = true_c + se_c * z_c
    delta_estimate = true_delta + se_delta * z_delta
    c_lower_positive = c_estimate - original.Z_975 * se_c > 0.0
    delta_lower_positive = delta_estimate - original.Z_975 * se_delta > 0.0

    null_indices = rng.integers(
        0, len(original.HISTORICAL_NULL_C), size=(replicates, random_controls)
    )
    null_true = original.HISTORICAL_NULL_C[null_indices]
    null_estimate = null_true + se_c * rng.normal(size=null_true.shape)
    above_max = c_estimate > null_estimate.max(axis=1)

    half_se_c = original.scaled_se(
        original.HISTORICAL_C_INTERVAL, n=independent_units, rollouts=2
    ) * se_multiplier
    half_se_delta = original.scaled_se(
        original.HISTORICAL_DELTA_INTERVAL, n=independent_units, rollouts=2
    ) * se_multiplier
    half_c = true_c + half_se_c * rng.normal(size=(replicates, 2))
    half_delta = true_delta + half_se_delta * rng.normal(size=(replicates, 2))
    split_consistent = np.all(half_c > 0.0, axis=1) & np.all(half_delta > 0.0, axis=1)
    joint = c_lower_positive & delta_lower_positive & above_max & split_consistent
    joint_probability = float(joint.mean())
    return {
        "design": design,
        "independent_units": independent_units,
        "raw_rows": raw_rows,
        "rollouts": rollouts,
        "random_controls": random_controls,
        "transfer_fraction": transfer_fraction,
        "within_family_rho": rho,
        "se_multiplier": se_multiplier,
        "effective_units": independent_units / (se_multiplier**2),
        "replicates": replicates,
        "true_c": true_c,
        "true_delta_c": true_delta,
        "expected_c_ci_width": 2.0 * original.Z_975 * se_c,
        "expected_delta_c_ci_width": 2.0 * original.Z_975 * se_delta,
        "power_c_lower_gt_zero": float(c_lower_positive.mean()),
        "power_delta_c_lower_gt_zero": float(delta_lower_positive.mean()),
        "probability_c_above_every_random": float(above_max.mean()),
        "split_half_sign_consistency": float(split_consistent.mean()),
        "joint_frozen_rule_probability": joint_probability,
        "joint_probability_monte_carlo_se": float(
            np.sqrt(joint_probability * (1.0 - joint_probability) / replicates)
        ),
    }


def simulate_one_row_per_family(
    families: int,
    *,
    transfer_fraction: float,
    replicates: int = 100_000,
    seed: int = 2026083001,
) -> dict[str, float | int | str | None]:
    return _simulate(
        design="C_ONE_ROW_PER_FAMILY",
        independent_units=families,
        raw_rows=families,
        rollouts=4,
        random_controls=8,
        transfer_fraction=transfer_fraction,
        se_multiplier=1.0,
        rho=None,
        replicates=replicates,
        seed=seed,
    )


def simulate_family_balanced(
    family_sizes: Sequence[int],
    *,
    rho: float,
    transfer_fraction: float,
    replicates: int = 100_000,
    seed: int = 2026083001,
) -> dict[str, float | int | str | None]:
    factor = family_average_variance_factor(family_sizes, rho)
    return _simulate(
        design="B_EQUAL_FAMILY_ALL_ROWS",
        independent_units=len(family_sizes),
        raw_rows=sum(family_sizes),
        rollouts=4,
        random_controls=8,
        transfer_fraction=transfer_fraction,
        se_multiplier=float(np.sqrt(factor)),
        rho=rho,
        replicates=replicates,
        seed=seed,
    )


def simulate_row_weighted(
    family_sizes: Sequence[int],
    *,
    rho: float,
    transfer_fraction: float,
    replicates: int = 100_000,
    seed: int = 2026083001,
) -> dict[str, float | int | str | None]:
    effect = unequal_cluster_design_effect(family_sizes, rho)
    return _simulate(
        design="A_ROW_WEIGHTED_CLUSTER_SENSITIVITY",
        independent_units=sum(family_sizes),
        raw_rows=sum(family_sizes),
        rollouts=4,
        random_controls=8,
        transfer_fraction=transfer_fraction,
        se_multiplier=float(np.sqrt(effect)),
        rho=rho,
        replicates=replicates,
        seed=seed,
    )


__all__ = [
    "family_average_variance_factor",
    "simulate_family_balanced",
    "simulate_one_row_per_family",
    "simulate_row_weighted",
    "unequal_cluster_design_effect",
]
