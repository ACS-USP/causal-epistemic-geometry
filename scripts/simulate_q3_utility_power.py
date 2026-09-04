#!/usr/bin/env python3
"""Calibrate family-level Q3 utility inference using model-free simulations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_final_system_and_evaluation_supply"
PRECHECK = REVIEW / "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json"


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def student_t_quantile(probability: float, degrees: int) -> float:
    """High-order Cornish-Fisher t quantile (accurate for frozen N>=23)."""
    z = NormalDist().inv_cdf(probability)
    nu = float(degrees)
    return (
        z
        + (z**3 + z) / (4 * nu)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * nu**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * nu**3)
    )


def effect_values(rng: np.random.Generator, size: int, gain: float, scenario: str) -> np.ndarray:
    if scenario in {"REGULAR", "HETEROGENEOUS_DIFFICULTY"}:
        return np.full(size, gain)
    if scenario == "HETEROGENEOUS_EFFECT":
        value = rng.normal(gain, 0.05, size)
        return value - value.mean() + gain
    if scenario == "RARE_SYSTEMATIC_HARM":
        harm = rng.random(size) < 0.10
        value = np.full(size, (gain + 0.10 * 0.10) / 0.90)
        value[harm] = -0.10
        return value - value.mean() + gain
    if scenario == "CONSERVATIVE_COMBINED":
        harm = rng.random(size) < 0.10
        value = rng.normal((gain + 0.10 * 0.10) / 0.90, 0.05, size)
        value[harm] = rng.normal(-0.10, 0.02, harm.sum())
        return value - value.mean() + gain
    raise ValueError(scenario)


def family_difference_distribution(
    rng: np.random.Generator,
    rollouts: int,
    gain: float,
    discordance: float,
    scenario: str,
    seed_regime: str,
    pool: int = 200000,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if scenario in {"HETEROGENEOUS_DIFFICULTY", "CONSERVATIVE_COMBINED"}:
        icc = 0.15 if scenario == "HETEROGENEOUS_DIFFICULTY" else 0.35
        concentration = 1 / icc - 1
        p_champion = rng.beta(0.60 * concentration, 0.40 * concentration, pool)
    else:
        p_champion = np.full(pool, 0.60)
    delta = effect_values(rng, pool, gain, scenario)
    p_router = np.clip(p_champion + delta, 0.001, 0.999)
    delta = p_router - p_champion
    if seed_regime == "PAIRED_COMMON_UNIFORM":
        uniforms = rng.random((pool, rollouts))
        router = uniforms < p_router[:, None]
        champion = uniforms < p_champion[:, None]
        difference = (router.astype(int) - champion.astype(int)).mean(axis=1)
    else:
        q_max = np.minimum(p_router + p_champion, 2 - p_router - p_champion)
        q = np.minimum(np.maximum(discordance, np.abs(delta)), q_max)
        positive = (q + delta) / 2
        negative = (q - delta) / 2
        uniforms = rng.random((pool, rollouts))
        difference = (
            (uniforms < positive[:, None]).sum(axis=1)
            - ((uniforms >= positive[:, None]) & (uniforms < (positive + negative)[:, None])).sum(
                axis=1
            )
        ) / rollouts
    support, counts = np.unique(np.rint(difference * rollouts).astype(int), return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    values = support.astype(float) / rollouts
    # The finite synthetic family pool approximates the target estimand. Remove
    # only that Monte Carlo approximation error by the minimum transfer of mass
    # between the extreme support points, so every null is exactly mean-zero
    # and every alternative has exactly the declared mean gain.
    discrepancy = float(probabilities @ values - gain)
    if abs(discrepancy) > 1e-15:
        target = int(np.argmin(values) if discrepancy > 0 else np.argmax(values))
        sources = np.argsort(values)[::-1] if discrepancy > 0 else np.argsort(values)
        remaining = abs(discrepancy)
        for source in sources:
            distance = abs(values[source] - values[target])
            if distance <= 0 or probabilities[source] <= 0:
                continue
            amount = min(probabilities[source], remaining / distance)
            probabilities[source] -= amount
            probabilities[target] += amount
            remaining -= amount * distance
            if remaining <= 1e-14:
                break
        if remaining > 1e-12:
            raise RuntimeError("cannot center simulated family distribution")
    return (
        values,
        probabilities,
        {
            "realized_mean": float(probabilities @ values),
            "realized_sd": float(np.sqrt(probabilities @ (values - gain) ** 2)),
            "nonzero_fraction": float(probabilities[values != 0].sum()),
        },
    )


def simulate_panels(
    rng: np.random.Generator,
    values: np.ndarray,
    probabilities: np.ndarray,
    n: int,
    replicates: int,
    true_gain: float,
) -> dict[str, Any]:
    counts = rng.multinomial(n, probabilities, size=replicates)
    sums = counts @ values
    sums2 = counts @ (values * values)
    means = sums / n
    variance = np.maximum((sums2 - sums * sums / n) / max(n - 1, 1), 0)
    se = np.sqrt(variance / n)
    tstat = np.divide(means, se, out=np.zeros_like(means), where=se > 0)
    one_sided_critical = student_t_quantile(0.95, max(n - 1, 1))
    reject_t = tstat > one_sided_critical
    # Large-sample form of the studentized sign-flip statistic. It is retained
    # as a sensitivity because weak-null validity additionally needs symmetry.
    flip_se = np.sqrt(np.maximum(sums2, 0)) / n
    z_flip = np.divide(means, flip_se, out=np.zeros_like(means), where=flip_se > 0)
    reject_flip = z_flip > NormalDist().inv_cdf(0.95)
    critical = student_t_quantile(0.975, max(n - 1, 1))
    lower = means - critical * se
    upper = means + critical * se
    coverage = np.mean((lower <= true_gain) & (true_gain <= upper))
    half = critical * se
    return {
        "t_rejection": float(reject_t.mean()),
        "flip_rejection": float(reject_flip.mean()),
        "coverage": float(coverage),
        "mean_estimate": float(means.mean()),
        "mean_half_width": float(half.mean()),
        "median_half_width": float(np.median(half)),
        "degenerate_fraction": float(np.mean(se == 0)),
        "t_wilson": wilson(int(reject_t.sum()), replicates),
        "flip_wilson": wilson(int(reject_flip.sum()), replicates),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    precheck = json.loads(PRECHECK.read_text())
    cfg = precheck["power_precision"]
    rng = np.random.default_rng(cfg["simulation_seed"])
    scenarios = [
        "REGULAR",
        "HETEROGENEOUS_DIFFICULTY",
        "HETEROGENEOUS_EFFECT",
        "RARE_SYSTEMATIC_HARM",
        "CONSERVATIVE_COMBINED",
    ]
    calibration = []
    for scenario in scenarios:
        for discordance in cfg["discordance"]:
            for seed_regime in ("INDEPENDENT", "PAIRED_COMMON_UNIFORM"):
                for rollouts in cfg["rollouts"]:
                    values, probabilities, dist = family_difference_distribution(
                        rng, rollouts, 0.0, discordance, scenario, seed_regime
                    )
                    for n in cfg["n_families"]:
                        result = simulate_panels(
                            rng, values, probabilities, n, cfg["calibration_replicates"], 0.0
                        )
                        calibration.append(
                            {
                                "scenario": scenario,
                                "seed_regime": seed_regime,
                                "discordance": discordance,
                                "N": n,
                                "R": rollouts,
                                "t_fpr": result["t_rejection"],
                                "t_fpr_wilson_low": result["t_wilson"][0],
                                "t_fpr_wilson_high": result["t_wilson"][1],
                                "signflip_fpr": result["flip_rejection"],
                                "signflip_fpr_wilson_low": result["flip_wilson"][0],
                                "signflip_fpr_wilson_high": result["flip_wilson"][1],
                                "t_ci_coverage": result["coverage"],
                                "half_width": result["mean_half_width"],
                                "degenerate_fraction": result["degenerate_fraction"],
                                **dist,
                            }
                        )
    power = []
    for scenario, discordance in (("REGULAR", 0.20), ("CONSERVATIVE_COMBINED", 0.35)):
        for seed_regime in ("INDEPENDENT", "PAIRED_COMMON_UNIFORM"):
            for gain in [0.01, *cfg["accuracy_gains"]]:
                for rollouts in cfg["rollouts"]:
                    values, probabilities, dist = family_difference_distribution(
                        rng, rollouts, gain, discordance, scenario, seed_regime
                    )
                    for n in cfg["n_families"]:
                        result = simulate_panels(
                            rng, values, probabilities, n, cfg["power_replicates"], gain
                        )
                        trajectories = n * rollouts * 2
                        full_bank = n * rollouts * 8
                        per_row = 33924.293892600996 / 19200
                        token_per_row = 381630 / 19200
                        byte_per_row = 127968010 / 19200
                        power.append(
                            {
                                "scenario": scenario,
                                "seed_regime": seed_regime,
                                "discordance": discordance,
                                "gain": gain,
                                "N": n,
                                "R": rollouts,
                                "t_power": result["t_rejection"],
                                "signflip_power": result["flip_rejection"],
                                "t_ci_coverage": result["coverage"],
                                "mean_half_width": result["mean_half_width"],
                                "mean_estimate": result["mean_estimate"],
                                "degenerate_fraction": result["degenerate_fraction"],
                                "utility_trajectories_max": trajectories,
                                "full_bank_trajectories": full_bank,
                                "expected_generated_tokens_utility": trajectories * token_per_row,
                                "expected_storage_bytes_utility": trajectories * byte_per_row,
                                "observed_rate_hours_utility": trajectories * per_row / 3600,
                                "frozen_p50_hours_utility": 9.76 * trajectories / 19200,
                                "frozen_p80_hours_utility": 11.05 * trajectories / 19200,
                                "frozen_p95_hours_utility": 12.45 * trajectories / 19200,
                                **dist,
                            }
                        )
    write_csv(REVIEW / "Q3_UTILITY_INFERENCE_CALIBRATION.csv", calibration)
    write_csv(REVIEW / "Q3_UTILITY_POWER_PRECISION.csv", power)

    # Fail-closed method selection from the frozen criteria. Small N=23 is
    # reported but not allowed to veto a method intended for N>=100.
    regular = [row for row in calibration if row["N"] >= 100]
    independent_target = [
        row for row in calibration if row["N"] >= 800 and row["seed_regime"] == "INDEPENDENT"
    ]
    t_max_fpr = max(row["t_fpr"] for row in regular)
    flip_max_fpr = max(row["signflip_fpr"] for row in regular)
    t_min_coverage = min(row["t_ci_coverage"] for row in regular)
    target_t_max_fpr = max(row["t_fpr"] for row in independent_target)
    target_t_min_coverage = min(row["t_ci_coverage"] for row in independent_target)
    selected = (
        "PAIRED_FAMILY_STUDENTIZED_T"
        if target_t_max_fpr <= 0.065 and target_t_min_coverage >= 0.93
        else "NONE"
    )
    conservative = [
        row
        for row in power
        if row["scenario"] == "CONSERVATIVE_COMBINED"
        and row["seed_regime"] == "INDEPENDENT"
        and row["gain"] == 0.03
        and row["N"] <= 500
        and row["R"] in (4, 6, 8)
    ]
    adequate = sorted(
        (row for row in conservative if row["t_power"] >= 0.80),
        key=lambda row: (row["utility_trajectories_max"], row["N"], row["R"]),
    )
    summary = {
        "schema_version": "q3-utility-power-precision-summary-v1",
        "status": "Q3_UTILITY_INFERENCE_MODEL_FREE_CALIBRATION_COMPLETE",
        "selected_primary": selected,
        "selected_ci": "T_95_PERCENT_FAMILY_LEVEL" if selected != "NONE" else "NONE",
        "estimand": cfg["endpoint"],
        "independent_unit": "semantic family",
        "rollouts": "nested repeated measurements",
        "same_numeric_seed_for_distinct_policies": "NOT_RECOMMENDED",
        "same_policy_shared_generation": "VALID; paired difference is mechanically zero",
        "t_max_fpr_n_ge_100": t_max_fpr,
        "signflip_max_fpr_n_ge_100": flip_max_fpr,
        "t_min_coverage_n_ge_100": t_min_coverage,
        "selected_regime": "INDEPENDENT_POLICY_SEEDS_AND_N_GE_800",
        "selected_regime_t_max_fpr": target_t_max_fpr,
        "selected_regime_t_min_coverage": target_t_min_coverage,
        "minimum_conservative_n_le_500_design": adequate[0] if adequate else None,
        "tier_b_11_family_power_is_inadequate_by_design": True,
        "q2_compound_bootstrap_reused": False,
        "simulation": {
            "calibration_rows": len(calibration),
            "power_rows": len(power),
            "calibration_replicates_per_cell": cfg["calibration_replicates"],
            "power_replicates_per_cell": cfg["power_replicates"],
        },
    }
    (REVIEW / "Q3_UTILITY_POWER_PRECISION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
