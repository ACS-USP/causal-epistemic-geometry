#!/usr/bin/env python3
# ruff: noqa: E501
"""Model-free Q2 OOS heterogeneity diagnosis and inference tournament."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.heterogeneity_robust import (
    exact_sign_test,
    node_jackknife_test,
    rank_cluster_regression,
    row_spearman,
    studentized_mean_test,
)
from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    angular_cross_block,
    cross_block_shape,
    fresh_row_permutations,
    spearman_flat,
)
from scripts.calibrate_q2_oos_v2_row_qap import (
    normalized_ranks,
    qap_cache,
    reference_coefficients,
    stress_panel_ranks,
    stress_setup,
    strict_panel_ranks,
    wilson,
)
from scripts.review_q2_oos_v2_k import choose_latent, unit_sphere

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/heterogeneity_robust_inference"
PRECHECK_COMMIT = "d1166eaa202fddc68af8da5a98c8f18f747939e6"
K = 16
R = 31
SHELL_NAMES = ("MEDIUM", "STRONG")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def derived_seed(base: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{label}".encode()).digest()[:16], "big")


def rank_matrix(values: np.ndarray) -> np.ndarray:
    ranks = normalized_ranks(values)
    return ranks.reshape(values.shape)


def geometry_cache(geometry: np.ndarray, seed: int, maps: int = 1000) -> np.ndarray:
    permutations = fresh_row_permutations(K, maps, seed=seed)
    return qap_cache(geometry, permutations)


def qap_result(cache: np.ndarray, outcomes: dict[str, np.ndarray]) -> tuple[float, float]:
    ranks = np.column_stack([normalized_ranks(outcomes[shell]) for shell in SHELL_NAMES])
    statistics = np.mean(cache @ ranks, axis=1)
    observed = float(statistics[0])
    p_value = float(np.mean(statistics >= observed))
    return observed, p_value


def binary_panel(
    latent: np.ndarray,
    seed: int,
    *,
    intercepts: bool,
    shared_items: bool,
    coupled_shells: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    controller_intercept = rng.normal(0.0, 0.25, size=(len(latent), 1)) if intercepts else 0.0
    shared_features = rng.standard_normal((300, latent.shape[1]))
    shared_base = rng.normal(0.0, 0.75, size=300)
    results: dict[str, np.ndarray] = {}
    for shell, amplitude in (("MEDIUM", 0.75), ("STRONG", 1.15)):
        if coupled_shells:
            features = shared_features
            base = shared_base
        else:
            features = rng.standard_normal((300, latent.shape[1]))
            base = rng.normal(0.0, 0.75, size=300)
        if shared_items:
            response = latent @ features.T
            logit = base[None, :] + amplitude * response + controller_intercept
        else:
            features_by_controller = rng.standard_normal((len(latent), 300, latent.shape[1]))
            response = np.einsum("cd,ctd->ct", latent, features_by_controller)
            controller_base = rng.normal(0.0, 0.75, size=(len(latent), 300))
            logit = controller_base + amplitude * response + controller_intercept
        probability = 1.0 / (1.0 + np.exp(-logit))
        errors = (rng.random((len(latent), 300, 2)) < probability[:, :, None]).astype(float)
        results[shell] = cross_block_shape(errors[:K], errors[K:])
    return results


def ablation_setup() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    reference = reference_coefficients()
    banks, latents, _caches = stress_setup(reference)
    return reference, banks, latents


def ablation_panel(
    scenario: str,
    index: int,
    reference: np.ndarray,
    banks: list[np.ndarray],
    stress_latents: list[np.ndarray],
    base_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    safety = "SAFETY" in scenario or scenario == "FROZEN_NONEXCHANGEABLE_STRESS_NULL"
    bank_index = index % len(banks)
    if safety:
        fresh = banks[bank_index]
    else:
        rng_geometry = np.random.Generator(
            np.random.PCG64DXSM(derived_seed(base_seed, f"{scenario}|GEOMETRY"))
        )
        fresh = unit_sphere(rng_geometry, (K, 8))
    geometry = angular_cross_block(fresh, reference)
    if scenario == "FROZEN_NONEXCHANGEABLE_STRESS_NULL":
        pair = stress_panel_ranks(
            latent=stress_latents[bank_index],
            k=K,
            seed=derived_seed(base_seed, f"{scenario}|{bank_index}|{index // len(banks)}"),
        )
        return geometry, {shell: pair[pos].reshape(K, R) for pos, shell in enumerate(SHELL_NAMES)}, False

    rng = np.random.Generator(np.random.PCG64DXSM(derived_seed(base_seed, f"{scenario}|{index}")))
    combined = np.vstack([fresh, reference])
    geometry_nuisance = "GEOMETRY_NUISANCE" in scenario or scenario.startswith("SAFETY_GEOMETRY")
    if geometry_nuisance:
        latent, _rho = choose_latent(combined, K, 0.0, rng)
    else:
        latent = unit_sphere(rng, combined.shape)
    if scenario == "NO_FRESH_LATENT_HETEROGENEITY":
        latent[:K] = np.mean(latent[:K], axis=0)
        latent[:K] /= np.linalg.norm(latent[:K], axis=1, keepdims=True)
    if scenario == "NO_REFERENCE_LATENT_HETEROGENEITY":
        latent[K:] = np.mean(latent[K:], axis=0)
        latent[K:] /= np.linalg.norm(latent[K:], axis=1, keepdims=True)
    outcomes = binary_panel(
        latent,
        derived_seed(base_seed, f"{scenario}|PANEL|{index}"),
        intercepts=scenario not in {"NO_CONTROLLER_INTERCEPTS", "SAFETY_GEOMETRY_WITHOUT_INTERCEPTS"},
        shared_items=scenario != "NO_SHARED_ITEM_FACTORS",
        coupled_shells=scenario not in {"INDEPENDENT_SHELLS", "SAFETY_GEOMETRY_INDEPENDENT_SHELLS"},
    )
    exchangeable = not safety and not geometry_nuisance
    return geometry, outcomes, exchangeable


def summarize_rejections(values: list[bool]) -> dict[str, Any]:
    successes = int(np.sum(values))
    low, high = wilson(successes, len(values))
    return {
        "replicates": len(values),
        "rejections": successes,
        "rate": successes / len(values),
        "Wilson_95_low": low,
        "Wilson_95_high": high,
    }


def run_ablation(replicates: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    precheck = read_json(OUT / "NONEXCHANGEABILITY_ABLATION_PRECHECK.json")
    base_seed = int(precheck["design"]["base_seed"])
    reference, banks, stress_latents = ablation_setup()
    rows: list[dict[str, Any]] = []
    summaries = []
    for specification in precheck["scenarios"]:
        scenario = specification["id"]
        qap_rejections: list[bool] = []
        globals_: list[float] = []
        row_values: list[float] = []
        exchangeable_values: list[bool] = []
        cache_by_geometry: dict[bytes, np.ndarray] = {}
        for index in range(replicates):
            geometry, outcomes, exchangeable = ablation_panel(
                scenario, index, reference, banks, stress_latents, base_seed
            )
            key = hashlib.sha256(geometry.tobytes()).digest()
            if key not in cache_by_geometry:
                cache_by_geometry[key] = geometry_cache(
                    geometry, derived_seed(base_seed, f"{scenario}|QAP|{key.hex()}")
                )
            observed, p_value = qap_result(cache_by_geometry[key], outcomes)
            row = row_spearman({shell: geometry for shell in SHELL_NAMES}, outcomes)
            qap_rejections.append(bool(observed > 0.0 and p_value <= 0.05))
            globals_.append(observed)
            row_values.extend(row.tolist())
            exchangeable_values.append(exchangeable)
        rejection = summarize_rejections(qap_rejections)
        global_array = np.asarray(globals_)
        row_array = np.asarray(row_values)
        summary = {
            "scenario": scenario,
            **rejection,
            "global_rho_mean": float(np.mean(global_array)),
            "global_rho_q05": float(np.quantile(global_array, 0.05)),
            "global_rho_q50": float(np.quantile(global_array, 0.50)),
            "global_rho_q95": float(np.quantile(global_array, 0.95)),
            "row_rho_q05": float(np.quantile(row_array, 0.05)),
            "row_rho_q50": float(np.quantile(row_array, 0.50)),
            "row_rho_q95": float(np.quantile(row_array, 0.95)),
            "fresh_rows_exchangeable_by_construction": bool(all(exchangeable_values)),
            "global_zero_and_row_exchangeability_coincide": bool(
                abs(np.mean(global_array)) <= 0.03 and all(exchangeable_values)
            ),
        }
        rows.append(summary)
        summaries.append(summary)
    result = {
        "schema_version": "q2-oos-nonexchangeability-ablation-result-v1",
        "precheck_commit": PRECHECK_COMMIT,
        "scenarios": summaries,
        "model_inference": 0,
        "semantic_outcomes": 0,
    }
    return rows, result


def continuous_panel(
    geometry: np.ndarray,
    seed: int,
    scenario: str,
    target: float,
) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    standardized = (geometry - np.mean(geometry, axis=1, keepdims=True)) / np.std(
        geometry, axis=1, keepdims=True
    )
    if scenario == "ROW_HETEROGENEITY_NULL":
        beta = np.asarray([0.65] * 8 + [-0.65] * 8)
        rng.shuffle(beta)
    elif scenario == "HEAVY_HETEROGENEITY_NULL":
        magnitudes = np.clip(np.abs(rng.standard_cauchy(8)), 0.25, 2.5)
        beta = np.concatenate([magnitudes, -magnitudes])
        rng.shuffle(beta)
    elif target > 0.0:
        beta = target + rng.normal(0.0, 0.12, K)
    else:
        beta = np.zeros(K)
    results = {}
    for shell in SHELL_NAMES:
        noise = rng.standard_normal((K, R))
        if scenario == "HEAVY_HETEROGENEITY_NULL":
            noise *= np.clip(np.abs(rng.standard_cauchy((K, 1))), 0.5, 3.0)
        reference_effect = rng.normal(0.0, 0.35, (1, R))
        row_effect = rng.normal(0.0, 0.35, (K, 1))
        results[shell] = beta[:, None] * standardized + noise + reference_effect + row_effect
    return results


def cluster_bootstrap_global(
    geometry: np.ndarray,
    outcomes: dict[str, np.ndarray],
    samples: np.ndarray,
) -> dict[str, float | bool]:
    observed = np.mean([spearman_flat(geometry, outcomes[shell]) for shell in SHELL_NAMES])
    bootstrap = np.empty(len(samples), dtype=np.float64)
    for index, sample in enumerate(samples):
        bootstrap[index] = np.mean(
            [spearman_flat(geometry[sample], outcomes[shell][sample]) for shell in SHELL_NAMES]
        )
    q025, q975 = np.quantile(bootstrap, (0.025, 0.975))
    percentile_lower = float(q025)
    basic_lower = float(2.0 * observed - q975)
    return {
        "estimate": float(observed),
        "percentile_lower": percentile_lower,
        "basic_lower": basic_lower,
        "percentile_reject": bool(percentile_lower > 0.0),
        "basic_reject": bool(basic_lower > 0.0),
        "percentile_covers_zero": bool(q025 <= 0.0 <= q975),
        "basic_covers_zero": bool(2.0 * observed - q975 <= 0.0 <= 2.0 * observed - q025),
    }


def method_results(
    geometry: np.ndarray,
    outcomes: dict[str, np.ndarray],
    cache: np.ndarray,
    bootstrap_samples: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    observed, qap_p = qap_result(cache, outcomes)
    rows = row_spearman({shell: geometry for shell in SHELL_NAMES}, outcomes)
    sign = exact_sign_test(rows)
    student = studentized_mean_test(rows)
    regression = rank_cluster_regression(
        {shell: geometry for shell in SHELL_NAMES}, outcomes
    )
    lofo_means = np.asarray([np.mean(np.delete(rows, index)) for index in range(K)])
    values: dict[str, dict[str, Any]] = {
        "A_ORIGINAL_ROW_QAP": {"estimate": observed, "reject": observed > 0.0 and qap_p <= 0.05, "coverage": None},
        "B_ROW_SPEARMAN_SIGN": {"estimate": sign["median"], "reject": sign["reject_0_05"], "coverage": None},
        "C_STUDENTIZED_MEAN_ROW_ASSOCIATION": {"estimate": student["mean"], "reject": student["reject_0_05"], "coverage": not student["reject_0_05"]},
        "E_RANK_REGRESSION_CLUSTER": {"estimate": regression["slope"], "reject": regression["reject_0_05"], "coverage": not regression["reject_0_05"]},
    }
    for value in values.values():
        value["lofo_min"] = float(np.min(lofo_means))
        value["lofo_max"] = float(np.max(lofo_means))
        value["lofo_all_positive"] = bool(np.all(lofo_means > 0.0))
    if bootstrap_samples is not None:
        bootstrap = cluster_bootstrap_global(geometry, outcomes, bootstrap_samples)
        values["D_CLUSTER_BOOTSTRAP_PERCENTILE"] = {
            "estimate": bootstrap["estimate"],
            "reject": bootstrap["percentile_reject"],
            "coverage": bootstrap["percentile_covers_zero"],
            "lofo_min": float(np.min(lofo_means)),
            "lofo_max": float(np.max(lofo_means)),
            "lofo_all_positive": bool(np.all(lofo_means > 0.0)),
        }
        values["D_CLUSTER_BOOTSTRAP_BASIC"] = {
            "estimate": bootstrap["estimate"],
            "reject": bootstrap["basic_reject"],
            "coverage": bootstrap["basic_covers_zero"],
            "lofo_min": float(np.min(lofo_means)),
            "lofo_max": float(np.max(lofo_means)),
            "lofo_all_positive": bool(np.all(lofo_means > 0.0)),
        }
    return values


def tournament_panel(
    scenario: str,
    index: int,
    reference: np.ndarray,
    banks: list[np.ndarray],
    stress_latents: list[np.ndarray],
    seed: int,
    target: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bank_index = index % len(banks)
    if scenario in {"FROZEN_NONEXCHANGEABLE_STRESS_NULL", "SAFETY_CONDITIONED_NULL"}:
        fresh = banks[bank_index]
    else:
        rng = np.random.Generator(np.random.PCG64DXSM(derived_seed(seed, "GEOMETRY")))
        fresh = unit_sphere(rng, (K, 8))
    geometry = angular_cross_block(fresh, reference)
    if scenario == "STRICT_EXCHANGEABLE_NULL":
        pair = strict_panel_ranks(seed=derived_seed(seed, f"PANEL|{index}"), k=K)
        outcomes = {shell: pair[pos].reshape(K, R) for pos, shell in enumerate(SHELL_NAMES)}
    elif scenario == "FROZEN_NONEXCHANGEABLE_STRESS_NULL":
        pair = stress_panel_ranks(
            latent=stress_latents[bank_index],
            k=K,
            seed=derived_seed(seed, f"PANEL|{bank_index}|{index // len(banks)}"),
        )
        outcomes = {shell: pair[pos].reshape(K, R) for pos, shell in enumerate(SHELL_NAMES)}
    else:
        outcomes = continuous_panel(geometry, derived_seed(seed, f"PANEL|{index}"), scenario, target)
    return geometry, outcomes


def run_tournament(scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    precheck = read_json(OUT / "HETEROGENEITY_ROBUST_INFERENCE_PRECHECK.json")
    reference, banks, stress_latents = ablation_setup()
    result_rows: list[dict[str, Any]] = []
    for scenario_spec in precheck["scenarios"]:
        scenario = scenario_spec["id"]
        target = float(scenario_spec.get("target", 0.0))
        total = max(20, int(scenario_spec["replicates_A_B_C_E"] * scale))
        bootstrap_total = max(10, int(scenario_spec["replicates_D"] * scale))
        seed_name = {
            "STRICT_EXCHANGEABLE_NULL": "strict",
            "FROZEN_NONEXCHANGEABLE_STRESS_NULL": "stress",
            "ROW_HETEROGENEITY_NULL": "row_heterogeneity",
            "SAFETY_CONDITIONED_NULL": "safety",
            "HEAVY_HETEROGENEITY_NULL": "heavy",
            "POSITIVE_25_PERCENT_CLOSED_A0": "alternative_25",
            "POSITIVE_50_PERCENT_CLOSED_A0": "alternative_50",
            "POSITIVE_RHO_LIKE_0_15": "alternative_015",
        }[scenario]
        seed = int(precheck["seeds"][seed_name])
        bootstrap_rng = np.random.Generator(
            np.random.PCG64DXSM(derived_seed(int(precheck["seeds"]["bootstrap"]), scenario))
        )
        bootstrap_samples = bootstrap_rng.integers(0, K, size=(499, K))
        by_method: dict[str, list[dict[str, Any]]] = {}
        cache_by_geometry: dict[bytes, np.ndarray] = {}
        for index in range(total):
            geometry, outcomes = tournament_panel(
                scenario, index, reference, banks, stress_latents, seed, target
            )
            key = hashlib.sha256(geometry.tobytes()).digest()
            if key not in cache_by_geometry:
                cache_by_geometry[key] = geometry_cache(
                    geometry, derived_seed(seed, f"QAP|{key.hex()}")
                )
            methods = method_results(
                geometry,
                outcomes,
                cache_by_geometry[key],
                bootstrap_samples if index < bootstrap_total else None,
            )
            for method, values in methods.items():
                by_method.setdefault(method, []).append(values)
        for method, values in by_method.items():
            rejection = summarize_rejections([bool(value["reject"]) for value in values])
            estimates = np.asarray([float(value["estimate"]) for value in values])
            coverages = [value["coverage"] for value in values if value["coverage"] is not None]
            result_rows.append(
                {
                    "scenario": scenario,
                    "kind": scenario_spec["kind"],
                    "method": method,
                    **rejection,
                    "mean_estimate": float(np.mean(estimates)),
                    "signed_bias_vs_nominal_target": float(np.mean(estimates) - target),
                    "coverage": float(np.mean(coverages)) if coverages else "NA",
                    "lofo_all_positive_rate": float(
                        np.mean([bool(value["lofo_all_positive"]) for value in values])
                    ),
                    "lofo_range_p95": float(
                        np.quantile(
                            [float(value["lofo_max"]) - float(value["lofo_min"]) for value in values],
                            0.95,
                        )
                    ),
                }
            )
    nulls = [row for row in result_rows if row["kind"] == "null"]
    alternatives = [row for row in result_rows if row["kind"] == "alternative"]
    method_ids = sorted({row["method"] for row in result_rows})
    decisions = []
    for method in method_ids:
        method_nulls = [row for row in nulls if row["method"] == method]
        calibrated = bool(
            len(method_nulls) == 5
            and all(float(row["rate"]) <= 0.065 and float(row["Wilson_95_low"]) <= 0.055 for row in method_nulls)
        )
        power = {row["scenario"]: float(row["rate"]) for row in alternatives if row["method"] == method}
        adequate = bool(
            power.get("POSITIVE_50_PERCENT_CLOSED_A0", 0.0) >= 0.60
            and power.get("POSITIVE_RHO_LIKE_0_15", 0.0) >= 0.60
        )
        decisions.append({"method": method, "calibrated_all_nulls": calibrated, "preferred_power_adequate": adequate, "power": power})
    precedence = [
        "B_ROW_SPEARMAN_SIGN",
        "C_STUDENTIZED_MEAN_ROW_ASSOCIATION",
        "D_CLUSTER_BOOTSTRAP_BASIC",
        "D_CLUSTER_BOOTSTRAP_PERCENTILE",
        "E_RANK_REGRESSION_CLUSTER",
    ]
    selected = next(
        (method for method in precedence if any(row["method"] == method and row["calibrated_all_nulls"] and row["preferred_power_adequate"] for row in decisions)),
        None,
    )
    summary = {
        "schema_version": "q2-oos-heterogeneity-robust-method-selection-v1",
        "precheck_commit": PRECHECK_COMMIT,
        "method_decisions": decisions,
        "selected_primary": selected,
        "global_cross_block_rho_role": "DESCRIPTIVE_EFFECT_SIZE",
        "original_row_qap_role": "SECONDARY_DIAGNOSTIC_ONLY" if selected else "NOT_SELECTED",
        "terminal_recommendation": "Q2_OOS_HETEROGENEITY_ROBUST_INFERENCE_READY" if selected else "Q2_OOS_INFERENCE_REQUIRES_FURTHER_THEORY",
    }
    return result_rows, summary


def symmetric_qap_cache(geometry: np.ndarray, seed: int, maps: int = 1000) -> np.ndarray:
    permutations = fresh_row_permutations(len(geometry), maps, seed=seed)
    upper = np.triu_indices(len(geometry), 1)
    cache = np.empty((len(permutations), len(upper[0])), dtype=np.float64)
    for index, permutation in enumerate(permutations):
        cache[index] = normalized_ranks(geometry[np.ix_(permutation, permutation)][upper])
    return cache


def symmetric_panel(geometry: np.ndarray, seed: int, scenario: str, target: float) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    upper = np.triu_indices(len(geometry), 1)
    geometry_z = (geometry - np.mean(geometry[upper])) / np.std(geometry[upper])
    if scenario == "NODE_HETEROGENEITY_NULL":
        beta = np.asarray([0.8] * 8 + [-0.8] * 8)
        rng.shuffle(beta)
    elif scenario == "HEAVY_NODE_HETEROGENEITY_NULL":
        magnitude = np.clip(np.abs(rng.standard_cauchy(8)), 0.25, 2.5)
        beta = np.concatenate([magnitude, -magnitude])
        rng.shuffle(beta)
    else:
        beta = np.full(len(geometry), target)
    results = {}
    for shell in SHELL_NAMES:
        noise = rng.standard_normal(geometry.shape)
        noise = 0.5 * (noise + noise.T)
        coefficient = 0.5 * (beta[:, None] + beta[None, :])
        outcome = coefficient * geometry_z + noise
        np.fill_diagonal(outcome, 0.0)
        results[shell] = outcome
    return results


def run_fresh_fresh(scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    precheck = read_json(OUT / "FRESH_FRESH_SECONDARY_PRECHECK.json")
    seed = int(precheck["design"]["seed"])
    rng = np.random.Generator(np.random.PCG64DXSM(derived_seed(seed, "GEOMETRY")))
    coefficients = unit_sphere(rng, (K, 8))
    geometry = 1.0 - np.clip(coefficients @ coefficients.T, -1.0, 1.0)
    upper = np.triu_indices(K, 1)
    cache = symmetric_qap_cache(geometry, derived_seed(seed, "QAP"))
    scenarios = [
        ("STRICT_EXCHANGEABLE_NODE_NULL", 0.0, max(20, int(5000 * scale))),
        ("NODE_HETEROGENEITY_NULL", 0.0, max(20, int(5000 * scale))),
        ("HEAVY_NODE_HETEROGENEITY_NULL", 0.0, max(20, int(5000 * scale))),
        ("POSITIVE_RHO_LIKE_0_15", 0.15, max(20, int(3000 * scale))),
    ]
    rows = []
    for scenario, target, replicates in scenarios:
        qap_reject = []
        jackknife_reject = []
        estimates = []
        for index in range(replicates):
            outcomes = symmetric_panel(
                geometry, derived_seed(seed, f"{scenario}|{index}"), scenario, target
            )
            outcome_ranks = np.column_stack(
                [normalized_ranks(outcomes[shell][upper]) for shell in SHELL_NAMES]
            )
            statistics = np.mean(cache @ outcome_ranks, axis=1)
            qap_reject.append(bool(statistics[0] > 0.0 and np.mean(statistics >= statistics[0]) <= 0.05))
            jackknife = node_jackknife_test(
                {shell: geometry for shell in SHELL_NAMES}, outcomes
            )
            jackknife_reject.append(bool(jackknife["reject_0_05"]))
            estimates.append(float(jackknife["full_association"]))
        for method, rejected in (
            ("HISTORICAL_CONJUGATION_QAP", qap_reject),
            ("NODE_JACKKNIFE_PSEUDOVALUE_T", jackknife_reject),
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    **summarize_rejections(rejected),
                    "mean_association": float(np.mean(estimates)),
                }
            )
    null_rows = [row for row in rows if "NULL" in row["scenario"]]
    jackknife_nulls = [row for row in null_rows if row["method"] == "NODE_JACKKNIFE_PSEUDOVALUE_T"]
    calibrated = bool(
        all(row["rate"] <= 0.065 and row["Wilson_95_low"] <= 0.055 for row in jackknife_nulls)
    )
    result = {
        "schema_version": "q2-oos-fresh-fresh-secondary-calibration-v1",
        "formal_method": "NODE_JACKKNIFE_PSEUDOVALUE_T" if calibrated else None,
        "status": "FORMAL_SECONDARY_CALIBRATED" if calibrated else "DESCRIPTIVE_SECONDARY_ONLY",
        "cannot_rescue_primary": True,
        "node_bootstrap": "NOT_RETAINED_DUPLICATED_NODE_RESAMPLES_CREATE_STRUCTURAL_ZERO_DYADS",
    }
    return rows, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.benchmark == args.full:
        raise SystemExit("choose exactly one of --benchmark or --full")
    scale = 0.01 if args.benchmark else 1.0
    started = time.monotonic()
    ablation_rows, ablation_result = run_ablation(max(20, int(5000 * scale)))
    tournament_rows, selection = run_tournament(scale)
    secondary_rows, secondary = run_fresh_fresh(scale)
    elapsed = time.monotonic() - started
    if args.benchmark:
        print(json.dumps({"benchmark_seconds": elapsed, "linear_projection_minutes": elapsed / scale / 60.0, "local_full_run_eligible": elapsed / scale <= 1800.0}, indent=2))
        return
    write_csv(OUT / "NONEXCHANGEABILITY_ABLATION.csv", ablation_rows)
    write_json(OUT / "NONEXCHANGEABILITY_ABLATION.json", ablation_result)
    write_csv(OUT / "METHOD_TOURNAMENT.csv", tournament_rows)
    write_json(OUT / "METHOD_SELECTION.json", selection)
    write_csv(OUT / "FRESH_FRESH_SECONDARY_CALIBRATION.csv", secondary_rows)
    write_json(OUT / "FRESH_FRESH_SECONDARY_CALIBRATION.json", secondary)
    write_json(
        OUT / "SIMULATION_METADATA.json",
        {"precheck_commit": PRECHECK_COMMIT, "elapsed_seconds": elapsed, "model_inference": 0, "semantic_outcomes": 0, "new_controller_streams": 0},
    )
    print(json.dumps({"elapsed_seconds": elapsed, "selection": selection, "secondary": secondary}, indent=2))


if __name__ == "__main__":
    main()
