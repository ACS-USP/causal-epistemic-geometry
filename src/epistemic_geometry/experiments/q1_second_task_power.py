"""Deterministic planning calculations for the Q1 second-task design.

The approximation is calibrated only from the frozen Qwen confirmatory point
estimates and percentile intervals.  It is planning evidence, not a new Q1
outcome and not a claim about LiveCodeBench performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HISTORICAL_N = 57
HISTORICAL_R = 2
HISTORICAL_MEANINGFUL_C = 0.05435463659147877
HISTORICAL_NULL_C = np.asarray(
    [
        0.024122807017543935,
        0.014097744360902331,
        0.016369047619047616,
        0.007988721804511378,
    ],
    dtype=np.float64,
)
HISTORICAL_DELTA_C = 0.03871005639097745
HISTORICAL_C_INTERVAL = (0.014411027568922305, 0.09680451127819548)
HISTORICAL_DELTA_INTERVAL = (0.00601112155388471, 0.07456140350877193)
PLANNING_ICC = 0.876655701754386
TEST_CORRELATION = 0.65
Z_975 = 1.959963984540054


@dataclass(frozen=True)
class PlanningCell:
    n: int
    rollouts: int
    random_controls: int
    transfer_fraction: float


def _historical_se(interval: tuple[float, float]) -> float:
    return (interval[1] - interval[0]) / (2.0 * Z_975)


def scaled_se(interval: tuple[float, float], *, n: int, rollouts: int) -> float:
    if n < 2 or rollouts < 2:
        raise ValueError("planning requires N>=2 and R>=2")
    item_factor = np.sqrt(HISTORICAL_N / n)
    rollout_variance = PLANNING_ICC + (1.0 - PLANNING_ICC) / rollouts
    historical_variance = PLANNING_ICC + (1.0 - PLANNING_ICC) / HISTORICAL_R
    return float(
        _historical_se(interval)
        * item_factor
        * np.sqrt(rollout_variance / historical_variance)
    )


def simulate_cell(
    cell: PlanningCell,
    *,
    replicates: int = 100_000,
    seed: int = 2026082901,
) -> dict[str, float | int]:
    """Approximate the frozen conjunctive rule under a transfer fraction.

    ``transfer_fraction=0`` makes the meaningful controller exchangeable with
    the null-bank mean for the null-specific contrast.  Values 0.5, 0.75 and
    1.0 retain the corresponding fraction of the historical Qwen excess C.
    """

    rng = np.random.default_rng(seed)
    se_c = scaled_se(HISTORICAL_C_INTERVAL, n=cell.n, rollouts=cell.rollouts)
    se_delta = scaled_se(HISTORICAL_DELTA_INTERVAL, n=cell.n, rollouts=cell.rollouts)
    null_mean = float(HISTORICAL_NULL_C.mean())
    true_delta = float(cell.transfer_fraction * HISTORICAL_DELTA_C)
    true_c = null_mean + true_delta
    z_c = rng.normal(size=replicates)
    z_delta = TEST_CORRELATION * z_c + np.sqrt(1.0 - TEST_CORRELATION**2) * rng.normal(
        size=replicates
    )
    c_estimate = true_c + se_c * z_c
    delta_estimate = true_delta + se_delta * z_delta
    c_lower_positive = c_estimate - Z_975 * se_c > 0.0
    delta_lower_positive = delta_estimate - Z_975 * se_delta > 0.0

    null_indices = rng.integers(0, len(HISTORICAL_NULL_C), size=(replicates, cell.random_controls))
    null_true = HISTORICAL_NULL_C[null_indices]
    null_estimate = null_true + se_c * rng.normal(size=null_true.shape)
    above_max = c_estimate > null_estimate.max(axis=1)

    half_se_c = scaled_se(HISTORICAL_C_INTERVAL, n=cell.n, rollouts=2)
    half_se_delta = scaled_se(HISTORICAL_DELTA_INTERVAL, n=cell.n, rollouts=2)
    half_c = true_c + half_se_c * rng.normal(size=(replicates, 2))
    half_delta = true_delta + half_se_delta * rng.normal(size=(replicates, 2))
    split_consistent = np.all(half_c > 0.0, axis=1) & np.all(half_delta > 0.0, axis=1)
    joint = c_lower_positive & delta_lower_positive & above_max & split_consistent
    return {
        "n": cell.n,
        "rollouts": cell.rollouts,
        "random_controls": cell.random_controls,
        "transfer_fraction": cell.transfer_fraction,
        "replicates": replicates,
        "true_c": true_c,
        "true_delta_c": true_delta,
        "expected_c_ci_width": 2.0 * Z_975 * se_c,
        "expected_delta_c_ci_width": 2.0 * Z_975 * se_delta,
        "power_c_lower_gt_zero": float(c_lower_positive.mean()),
        "power_delta_c_lower_gt_zero": float(delta_lower_positive.mean()),
        "probability_c_above_every_random": float(above_max.mean()),
        "split_half_sign_consistency": float(split_consistent.mean()),
        "joint_frozen_rule_probability": float(joint.mean()),
    }


def planning_grid(
    *,
    ns: tuple[int, ...] = (100, 120, 150, 200),
    rollouts: tuple[int, ...] = (2, 4),
    random_controls: tuple[int, ...] = (4, 8),
    transfer_fractions: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0),
    replicates: int = 100_000,
    seed: int = 2026082901,
) -> list[dict[str, float | int]]:
    records = []
    index = 0
    for n in ns:
        for rollout_count in rollouts:
            for random_count in random_controls:
                for fraction in transfer_fractions:
                    records.append(
                        simulate_cell(
                            PlanningCell(n, rollout_count, random_count, fraction),
                            replicates=replicates,
                            seed=seed + index,
                        )
                    )
                    index += 1
    return records


__all__ = [
    "HISTORICAL_C_INTERVAL",
    "HISTORICAL_DELTA_C",
    "HISTORICAL_DELTA_INTERVAL",
    "HISTORICAL_MEANINGFUL_C",
    "HISTORICAL_N",
    "HISTORICAL_NULL_C",
    "HISTORICAL_R",
    "PLANNING_ICC",
    "PlanningCell",
    "planning_grid",
    "scaled_se",
    "simulate_cell",
]
