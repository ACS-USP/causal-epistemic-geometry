#!/usr/bin/env python3
"""CPU-only reviewer-hardening audit for prospective Q2 OOS V2.

This script uses only public coefficient geometry, presemantic historical
safety labels summarized by the accepted V2 audit, and synthetic planning
panels. It never derives the future V2 stream seed, generates future V2
controllers, loads benchmark content, or reads semantic outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    SHELLS,
    angular_cross_block,
    cross_block_shape,
    fresh_row_permutations,
    leave_one_fresh_out,
    spearman_flat,
)
from epistemic_geometry.experiments.q2_v4 import average_ranks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/v2_reviewer_hardening"
PRECHECK = OUT / "REVIEWER_HARDENING_PRECHECK.json"
REFERENCE_MANIFEST = (
    ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
)
SAFETY_MODEL_PATH = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_gate_audit/HISTORICAL_SAFETY_GEOMETRY.json"
)
SOURCE_COMMIT = "ff7ede3785e3e4a203cf64f4260e7cc6b819918b"
DIMENSION = 8
REFERENCE_A0_RHO = 0.5638183484033006
SAFETY_HOURS_PER_CANDIDATE = 1.5267696481606668 / 24.0
SEMANTIC_HOURS_PER_CONTROLLER = 27.4018783902909 / 10.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(label: str) -> int:
    payload = f"Q2-OOS-V2-REVIEWER-HARDENING-{label}|{SOURCE_COMMIT}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def unit_sphere(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.standard_normal(shape)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def binomial_tail(n: int, k: int, p: float) -> float:
    return float(sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1)))


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return center - half, center + half


def solve_intercept(coordinate: np.ndarray, slope: float, target: float) -> float:
    low, high = -20.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2.0
        probability = 1.0 / (1.0 + np.exp(-(middle + slope * coordinate)))
        if float(np.mean(probability)) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def matrix_metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    scatter = np.einsum("bni,bnj->bij", values, values)
    eigenvalues = np.maximum(np.linalg.eigvalsh(scatter), 0.0)
    probabilities = eigenvalues / np.sum(eigenvalues, axis=1, keepdims=True)
    effective_rank = np.exp(
        -np.sum(
            np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0),
            axis=1,
        )
    )
    gram = np.einsum("bni,bmi->bnm", values, values)
    diagonal = np.arange(values.shape[1])
    gram[:, diagonal, diagonal] = 0.0
    return {
        "rank": np.sum(eigenvalues > 1e-20, axis=1),
        "effective_rank": effective_rank,
        "condition_number": np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0]),
        "maximum_absolute_pair_cosine": np.max(np.abs(gram), axis=(1, 2)),
    }


def candidate_integrity(values: np.ndarray) -> np.ndarray:
    metrics = matrix_metrics(values)
    return (metrics["rank"] == 8) & (metrics["maximum_absolute_pair_cosine"] < 0.98)


def selected_gate(
    values: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    metrics = matrix_metrics(values)
    a0 = 1.0 - np.einsum("bki,ri->bkr", values, reference)
    flattened = a0.reshape(len(values), -1)
    q10, q90 = np.quantile(flattened, (0.10, 0.90), axis=1)
    metrics["a0_q90_minus_q10"] = q90 - q10
    passed = (
        (metrics["rank"] == 8)
        & (metrics["effective_rank"] >= 4.8)
        & (metrics["maximum_absolute_pair_cosine"] < 0.98)
        & (metrics["a0_q90_minus_q10"] >= 0.20)
    )
    return passed, metrics


def first_safe(values: np.ndarray, safe: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.sum(safe, axis=1) >= k
    positions = np.broadcast_to(np.arange(values.shape[1]), safe.shape)
    indices = np.sort(np.where(safe, positions, values.shape[1]), axis=1)[:, :k]
    selected = values[eligible][np.arange(int(np.sum(eligible)))[:, None], indices[eligible]]
    return selected, eligible


def qualification_cell(
    *,
    k: int,
    n: int,
    replicates: int,
    scenario: str,
    reference: np.ndarray,
    slope: float,
    intercept: float,
    coordinate_mean: float,
    coordinate_sd: float,
) -> dict[str, Any]:
    seed = derived_seed(f"QUALIFICATION-K{k}-N{n}-{scenario}")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    reserve_successes = 0
    final_successes = 0
    selected_successes = 0
    batch_size = 1000
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        values = unit_sphere(rng, (size, n, DIMENSION))
        uniforms = rng.random((size, n))
        if scenario == "INDEPENDENT":
            probabilities = np.full((size, n), 0.60)
        elif scenario == "MODERATE_HISTORICAL_AXIS4":
            coordinate = (values[:, :, 4] - coordinate_mean) / coordinate_sd
            probabilities = 1.0 / (1.0 + np.exp(-(intercept + slope * coordinate)))
        else:
            raise ValueError(f"unknown safety scenario: {scenario}")
        selected, eligible = first_safe(values, uniforms < probabilities, k)
        reserve_successes += int(np.sum(eligible))
        if not np.any(eligible):
            continue
        selected_pass, _metrics = selected_gate(selected, reference)
        selected_successes += int(np.sum(selected_pass))
        final_successes += int(np.sum(selected_pass & candidate_integrity(values[eligible])))
    low, high = wilson(final_successes, replicates)
    return {
        "K": k,
        "candidate_count": n,
        "scenario": scenario,
        "replicates": replicates,
        "exact_binomial_reserve_probability": binomial_tail(n, k, 0.60),
        "simulated_reserve_probability": reserve_successes / replicates,
        "selected_gate_probability_given_reserve": (
            selected_successes / reserve_successes if reserve_successes else float("nan")
        ),
        "unconditional_inference_aligned_qualification": final_successes / replicates,
        "qualification_ci95_low": low,
        "qualification_ci95_high": high,
        "qualification_mc_se": math.sqrt(
            (final_successes / replicates) * (1.0 - final_successes / replicates) / replicates
        ),
    }


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = average_ranks(np.asarray(values, dtype=np.float64).reshape(-1))
    ranks -= np.mean(ranks)
    return ranks / np.linalg.norm(ranks)


def choose_latent(
    combined: np.ndarray,
    k: int,
    target_rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    signal = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    nuisance = unit_sphere(rng, signal.shape)
    target_geometry = angular_cross_block(signal[:k], signal[k:])
    best: tuple[float, float, np.ndarray] | None = None
    for weight in np.linspace(0.0, 1.0, 101):
        latent = np.concatenate(
            [np.sqrt(weight) * signal, np.sqrt(1.0 - weight) * nuisance], axis=1
        )
        latent /= np.linalg.norm(latent, axis=1, keepdims=True)
        achieved = spearman_flat(
            target_geometry,
            angular_cross_block(latent[:k], latent[k:]),
        )
        candidate = (abs(achieved - target_rho), achieved, latent)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[2], float(best[1])


def qap_cache(geometry: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    cache = np.empty((len(permutations), geometry.size), dtype=np.float64)
    for index, permutation in enumerate(permutations):
        cache[index] = normalized_ranks(geometry[permutation, :])
    return cache


def simulate_panel(
    *,
    latent: np.ndarray,
    k: int,
    geometry: np.ndarray,
    permutation_cache: np.ndarray,
    seed: int,
) -> dict[str, float | bool]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    item_features = rng.standard_normal((300, latent.shape[1]))
    base_logit = rng.normal(0.0, 0.75, size=300)
    response = latent @ item_features.T
    outcome_by_shell: dict[str, np.ndarray] = {}
    true_by_shell: dict[str, np.ndarray] = {}
    for shell, amplitude in (("MEDIUM", 0.75), ("STRONG", 1.15)):
        probability = 1.0 / (1.0 + np.exp(-(base_logit[None, :] + amplitude * response)))
        errors = (rng.random((len(latent), 300, 2)) < probability[:, :, None]).astype(float)
        outcome_by_shell[shell] = cross_block_shape(errors[:k], errors[k:])
        centered = probability - np.mean(probability, axis=1, keepdims=True)
        true_by_shell[shell] = np.mean(
            np.square(centered[:k, None, :] - centered[None, k:, :]), axis=2
        )
    permutation_statistics = np.mean(
        np.stack(
            [permutation_cache @ normalized_ranks(outcome_by_shell[shell]) for shell in SHELLS]
        ),
        axis=0,
    )
    observed = float(permutation_statistics[0])
    p_value = float(np.sum(permutation_statistics >= observed) / len(permutation_statistics))
    geometry_by_shell = {shell: geometry for shell in SHELLS}
    lofo = leave_one_fresh_out(geometry_by_shell, outcome_by_shell)
    true_rho = float(
        np.mean([spearman_flat(geometry, true_by_shell[shell]) for shell in SHELLS])
    )
    return {
        "observed_rho": observed,
        "true_rho": true_rho,
        "permutation_pass": bool(observed > 0.0 and p_value <= 0.05),
        "all_lofo_positive": bool(np.all(lofo > 0.0)),
        "lofo_min": float(np.min(lofo)),
    }


def draw_planning_banks(
    *,
    k: int,
    n: int,
    count: int,
    reference: np.ndarray,
    slope: float,
    intercept: float,
    coordinate_mean: float,
    coordinate_sd: float,
) -> list[np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(derived_seed(f"POWER-BANKS-K{k}-N{n}")))
    banks: list[np.ndarray] = []
    attempts = 0
    while len(banks) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("could not draw enough synthetic planning banks")
        values = unit_sphere(rng, (1, n, DIMENSION))
        coordinate = (values[:, :, 4] - coordinate_mean) / coordinate_sd
        probabilities = 1.0 / (1.0 + np.exp(-(intercept + slope * coordinate)))
        selected, eligible = first_safe(values, rng.random((1, n)) < probabilities, k)
        if not eligible[0] or not candidate_integrity(values)[0]:
            continue
        passed, _metrics = selected_gate(selected, reference)
        if passed[0]:
            banks.append(selected[0])
    return banks


def summarize_power(results: list[dict[str, float | bool]]) -> dict[str, float]:
    observed = np.asarray([float(row["observed_rho"]) for row in results])
    passed = np.asarray([bool(row["permutation_pass"]) for row in results])
    lofo = np.asarray([bool(row["all_lofo_positive"]) for row in results])
    probability = float(np.mean(passed))
    return {
        "mean_true_rho": float(np.mean([float(row["true_rho"]) for row in results])),
        "mean_observed_rho": float(np.mean(observed)),
        "sampling_q025": float(np.quantile(observed, 0.025)),
        "sampling_q975": float(np.quantile(observed, 0.975)),
        "sampling_95_width": float(np.quantile(observed, 0.975) - np.quantile(observed, 0.025)),
        "permutation_power_or_fpr": probability,
        "permutation_mc_se": math.sqrt(probability * (1.0 - probability) / len(results)),
        "all_lofo_positive_rate": float(np.mean(lofo)),
        "mean_lofo_min": float(np.mean([float(row["lofo_min"]) for row in results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-replicates", type=int)
    parser.add_argument("--power-replicates", type=int)
    args = parser.parse_args()
    precheck = read_json(PRECHECK)
    if precheck["source_commit"] != SOURCE_COMMIT or not precheck["frozen_before_final_simulation"]:
        raise RuntimeError("reviewer-hardening precheck is absent or not frozen")
    qualification_replicates = args.qualification_replicates or int(
        precheck["comparison"]["qualification_replicates_per_cell"]
    )
    power_replicates = args.power_replicates or int(precheck["power"]["replicates_per_K_effect"])
    if power_replicates % int(precheck["power"]["planning_banks_per_K"]):
        raise ValueError("power replicates must be divisible by planning-bank count")

    reference_payload = read_json(REFERENCE_MANIFEST)
    reference = np.asarray(
        [row["coefficients"] for row in reference_payload["directions"]], dtype=float
    )
    if reference.shape != (31, 8):
        raise RuntimeError("frozen reference coefficient bank is not 31x8")
    safety = read_json(SAFETY_MODEL_PATH)
    coordinate_mean = float(safety["coordinate_mean"])
    coordinate_sd = float(safety["coordinate_standard_deviation"])
    slope = float(safety["logistic_slope"])
    population_rng = np.random.Generator(np.random.PCG64DXSM(derived_seed("SAFETY-INTERCEPT")))
    population = unit_sphere(population_rng, (250000, DIMENSION))
    standardized_population = (population[:, 4] - coordinate_mean) / coordinate_sd
    moderate_intercept = solve_intercept(standardized_population, slope, 0.60)

    qualification_rows: list[dict[str, Any]] = []
    selected_n: dict[int, int] = {}
    for k in precheck["comparison"]["K_values"]:
        counts = precheck["comparison"]["candidate_count_search"][f"K{k}"]
        for n in counts:
            for scenario in precheck["comparison"]["qualification_scenarios"]:
                qualification_rows.append(
                    qualification_cell(
                        k=int(k),
                        n=int(n),
                        replicates=qualification_replicates,
                        scenario=str(scenario),
                        reference=reference,
                        slope=slope,
                        intercept=moderate_intercept,
                        coordinate_mean=coordinate_mean,
                        coordinate_sd=coordinate_sd,
                    )
                )
        rows_by_n = {
            int(n): [
                row
                for row in qualification_rows
                if row["K"] == k and row["candidate_count"] == n
            ]
            for n in counts
        }
        eligible = [
            int(n)
            for n, rows in rows_by_n.items()
            if rows[0]["exact_binomial_reserve_probability"] >= 0.95
            and all(row["unconditional_inference_aligned_qualification"] >= 0.95 for row in rows)
        ]
        if not eligible:
            raise RuntimeError(f"candidate search range did not qualify K={k}")
        selected_n[int(k)] = min(eligible)
    write_csv(OUT / "QUALIFICATION_SEARCH.csv", qualification_rows)

    power_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    effects = precheck["power"]["effect_grid"]
    banks_per_k = int(precheck["power"]["planning_banks_per_K"])
    maps = int(precheck["power"]["planning_row_permutations"])
    for k in precheck["comparison"]["K_values"]:
        k = int(k)
        n = selected_n[k]
        banks = draw_planning_banks(
            k=k,
            n=n,
            count=banks_per_k,
            reference=reference,
            slope=slope,
            intercept=moderate_intercept,
            coordinate_mean=coordinate_mean,
            coordinate_sd=coordinate_sd,
        )
        results_by_effect: dict[str, list[dict[str, float | bool]]] = {
            name: [] for name in effects
        }
        for bank_index, fresh in enumerate(banks):
            geometry = angular_cross_block(fresh, reference)
            permutations = fresh_row_permutations(
                k,
                maps,
                seed=derived_seed(f"POWER-QAP-K{k}-BANK{bank_index}"),
            )
            cache = qap_cache(geometry, permutations)
            for effect_name, target in effects.items():
                latent_rng = np.random.Generator(
                    np.random.PCG64DXSM(derived_seed(f"LATENT-K{k}-BANK{bank_index}-{effect_name}"))
                )
                latent, latent_rho = choose_latent(
                    np.vstack([fresh, reference]), k, float(target), latent_rng
                )
                bank_results = [
                    simulate_panel(
                        latent=latent,
                        k=k,
                        geometry=geometry,
                        permutation_cache=cache,
                        seed=derived_seed(
                            f"POWER-PANEL-K{k}-BANK{bank_index}-{effect_name}-R{replicate}"
                        ),
                    )
                    for replicate in range(power_replicates // banks_per_k)
                ]
                results_by_effect[effect_name].extend(bank_results)
                bank_rows.append(
                    {
                        "K": k,
                        "candidate_count": n,
                        "bank_index": bank_index,
                        "effect": effect_name,
                        "target_rho": target,
                        "achieved_latent_rho": latent_rho,
                        "replicates": len(bank_results),
                        **summarize_power(bank_results),
                    }
                )
        for effect_name, target in effects.items():
            power_rows.append(
                {
                    "K": k,
                    "candidate_count": n,
                    "effect": effect_name,
                    "target_rho": target,
                    "planning_banks": banks_per_k,
                    "replicates": len(results_by_effect[effect_name]),
                    "planning_permutations": maps,
                    "final_permutations": 50000,
                    **summarize_power(results_by_effect[effect_name]),
                }
            )
    write_csv(OUT / "POWER_COMPARISON.csv", power_rows)
    write_csv(OUT / "POWER_BY_PLANNING_BANK.csv", bank_rows)

    comparison_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for k in (10, 12, 16):
        n = selected_n[k]
        qualifying = [
            row
            for row in qualification_rows
            if row["K"] == k and row["candidate_count"] == n
        ]
        power = {row["effect"]: row for row in power_rows if row["K"] == k}
        projected_safety_hours = n * SAFETY_HOURS_PER_CANDIDATE
        projected_semantic_hours = k * SEMANTIC_HOURS_PER_CONTROLLER
        comparison = {
            "K": k,
            "candidate_count": n,
            "reserve_probability_p0_60": binomial_tail(n, k, 0.60),
            "independent_unconditional_qualification": next(
                row["unconditional_inference_aligned_qualification"]
                for row in qualifying
                if row["scenario"] == "INDEPENDENT"
            ),
            "moderate_axis_unconditional_qualification": next(
                row["unconditional_inference_aligned_qualification"]
                for row in qualifying
                if row["scenario"] == "MODERATE_HISTORICAL_AXIS4"
            ),
            "safety_trajectories": 48 * n,
            "semantic_trajectories": 1200 * k,
            "fresh_controller_identities": k,
            "fresh_old_dyads": 31 * k,
            "fresh_fresh_dyads": math.comb(k, 2),
            "projected_safety_gpu_hours": projected_safety_hours,
            "projected_semantic_gpu_hours": projected_semantic_hours,
            "projected_safety_plus_semantic_gpu_hours": (
                projected_safety_hours + projected_semantic_hours
            ),
            "half_prior_power": power["HALF_PRIOR_A0"]["permutation_power_or_fpr"],
            "quarter_prior_power": power["QUARTER_PRIOR_A0"]["permutation_power_or_fpr"],
            "rho_0_15_power": power["RHO_0_15"]["permutation_power_or_fpr"],
            "null_fpr": power["NULL"]["permutation_power_or_fpr"],
            "half_prior_all_lofo_positive_rate": power["HALF_PRIOR_A0"][
                "all_lofo_positive_rate"
            ],
            "rho_0_15_all_lofo_positive_rate": power["RHO_0_15"][
                "all_lofo_positive_rate"
            ],
            "half_prior_sampling_95_width": power["HALF_PRIOR_A0"]["sampling_95_width"],
            "rho_0_15_sampling_95_width": power["RHO_0_15"]["sampling_95_width"],
        }
        checks = {
            "reserve_probability": comparison["reserve_probability_p0_60"] >= 0.95,
            "independent_qualification": comparison[
                "independent_unconditional_qualification"
            ]
            >= 0.95,
            "moderate_axis_qualification": comparison[
                "moderate_axis_unconditional_qualification"
            ]
            >= 0.95,
            "null_fpr": comparison["null_fpr"] <= 0.075,
            "gpu_hours": comparison["projected_safety_plus_semantic_gpu_hours"] <= 50.0,
            "semantic_trajectories": comparison["semantic_trajectories"] <= 20000,
            "half_prior_power": comparison["half_prior_power"] >= 0.80,
            "rho_0_15_power": comparison["rho_0_15_power"] >= 0.60,
            "half_prior_lofo": comparison["half_prior_all_lofo_positive_rate"] >= 0.80,
            "half_prior_width": comparison["half_prior_sampling_95_width"] <= 0.40,
        }
        comparison_rows.append(comparison)
        decisions.append({"K": k, "checks": checks, "pass": bool(all(checks.values()))})
    write_csv(OUT / "K_COMPARISON.csv", comparison_rows)
    passing = [row["K"] for row in decisions if row["pass"]]
    if not passing:
        recommended_k = None
    else:
        recommended_k = max(passing)
        ordered = sorted(passing)
        if len(ordered) > 1:
            previous = ordered[-2]
            cost_by_k = {
                row["K"]: row["projected_safety_plus_semantic_gpu_hours"]
                for row in comparison_rows
            }
            increase = cost_by_k[recommended_k] / cost_by_k[previous] - 1.0
            if increase > 0.50:
                recommended_k = previous

    half_bank_rows = [row for row in bank_rows if row["effect"] == "HALF_PRIOR_A0"]
    bank_gate_failures = [
        row
        for row in half_bank_rows
        if row["permutation_power_or_fpr"] < 0.80
        or row["all_lofo_positive_rate"] < 0.80
    ]
    bank_gate_failure_fraction = len(bank_gate_failures) / len(half_bank_rows)
    power_gate_ruling = (
        "RETAIN_AS_TERMINAL_GATE"
        if bank_gate_failure_fraction > 0.05
        else "RETAIN_AS_MANDATORY_DIAGNOSTIC_ONLY"
    )
    permutation = {
        str(k): {
            "fresh_controller_permutation_group_size": math.factorial(k),
            "exact_enumeration_feasible": k == 10,
            "final_convention": "identity plus 49,999 unique sampled non-identity maps",
            "fresh_fresh_action": "P_pi A0 P_pi^T; Dshape fixed; same pi across shells",
        }
        for k in (10, 12, 16)
    }
    write_json(OUT / "PERMUTATION_FEASIBILITY.json", permutation)
    summary = {
        "schema_version": "q2-oos-v2-reviewer-hardening-summary-v1",
        "source_commit": SOURCE_COMMIT,
        "historical_v1_classification": "Q2_OOS_FRESH_CONTROLLER_DESIGN_BLOCKED",
        "accepted_route": "ROUTE_C_INFERENCE_ALIGNED_GATES",
        "decisions": decisions,
        "recommended_K": recommended_k,
        "recommended_candidate_count": selected_n.get(recommended_k),
        "predicted_power_gate_ruling": power_gate_ruling,
        "half_prior_planning_bank_gate_failure_fraction": bank_gate_failure_fraction,
        "fresh_fresh_secondary": precheck["fresh_fresh_secondary"],
        "new_v2_stream_generated": False,
        "actual_v2_seed_derived": False,
        "model_inference": 0,
        "semantic_trajectories": 0,
        "correctness_inspected": False,
        "spark1_used": False,
        "spark2_used": False,
        "artifact_hashes": {
            name: sha256(OUT / name)
            for name in (
                "REVIEWER_HARDENING_PRECHECK.json",
                "QUALIFICATION_SEARCH.csv",
                "POWER_COMPARISON.csv",
                "POWER_BY_PLANNING_BANK.csv",
                "K_COMPARISON.csv",
                "PERMUTATION_FEASIBILITY.json",
            )
        },
    }
    write_json(OUT / "SIMULATION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
