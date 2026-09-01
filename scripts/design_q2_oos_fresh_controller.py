#!/usr/bin/env python3
"""CPU-only power, precision, and reserve design for Q2 OOS controllers."""

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
    K_CANDIDATES,
    PRIMARY_N,
    QAP_MAPS,
    SHELLS,
    angular_cross_block,
    cross_block_shape,
    fresh_row_permutations,
    leave_one_fresh_out,
    protocol_seed,
    spearman_flat,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "q2_oos_fresh_controller_design"
REFERENCE_MANIFEST = (
    ROOT / "review" / "q2_v4_1_31_safe_bank_review" / "SAFE_31_IMMUTABLE_MANIFEST.json"
)
REFERENCE_COMMIT = "9232a765db4ebded0a997a9969f4691783a57d21"
REFERENCE_A0_RHO = 0.5638183484033006
EFFECT_SCENARIOS = {
    "NULL": 0.0,
    "HALF_PRIOR": 0.50 * REFERENCE_A0_RHO,
    "THREE_QUARTER_PRIOR": 0.75 * REFERENCE_A0_RHO,
    "FULL_PRIOR": REFERENCE_A0_RHO,
}
PLANNING_CRITERIA = {
    "minimum_fresh_controller_vertices": 10,
    "full_prior_permutation_power_min": 0.80,
    "three_quarter_prior_permutation_power_min": 0.60,
    "null_false_positive_rate_max": 0.075,
    "full_prior_all_lofo_positive_rate_min": 0.80,
    "full_prior_sampling_95_width_max": 0.40,
    "safe_bank_probability_min_at_p_0_70": 0.95,
    "selection_rule": "smallest K satisfying every criterion; ties favor lower future trajectories",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    from epistemic_geometry.experiments.q2_v4 import average_ranks

    ranks = average_ranks(np.asarray(values, dtype=np.float64).reshape(-1))
    ranks -= np.mean(ranks)
    return ranks / np.linalg.norm(ranks)


def reference_coefficients() -> np.ndarray:
    manifest = read_json(REFERENCE_MANIFEST)
    rows = np.asarray([row["coefficients"] for row in manifest["directions"]], dtype=np.float64)
    if rows.shape != (31, 8):
        raise RuntimeError("frozen reference coefficient matrix is not 31x8")
    return rows


def planning_fresh_coefficients(k: int) -> np.ndarray:
    seed = protocol_seed(f"Q2-OOS-POWER-COEFFICIENTS-K{k}-V1", REFERENCE_COMMIT)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    values = rng.standard_normal((k, 8))
    return unit_rows(values)


def choose_latent(
    coefficients: np.ndarray,
    fresh_count: int,
    target_rho: float,
    rng: np.random.Generator,
    nuisance_scale: float,
) -> tuple[np.ndarray, float, float]:
    signal = unit_rows(coefficients)
    nuisance = unit_rows(rng.standard_normal(coefficients.shape))
    target_geometry = angular_cross_block(signal[:fresh_count], signal[fresh_count:])
    weights = np.linspace(0.0, 1.0, 101)
    best: tuple[float, float, np.ndarray] | None = None
    for weight in weights:
        latent = unit_rows(
            np.concatenate(
                [np.sqrt(weight) * signal, nuisance_scale * np.sqrt(1.0 - weight) * nuisance],
                axis=1,
            )
        )
        achieved = spearman_flat(
            target_geometry,
            angular_cross_block(latent[:fresh_count], latent[fresh_count:]),
        )
        candidate = (abs(achieved - target_rho), achieved, latent)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[2], float(best[1]), float(weights[0] if target_rho == 0.0 else np.nan)


def qap_rank_cache(geometry: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    cache = np.empty((len(permutations), geometry.size), dtype=np.float64)
    for index, permutation in enumerate(permutations):
        cache[index] = normalized_ranks(geometry[permutation, :])
    return cache


def simulate_once(
    fresh: np.ndarray,
    reference: np.ndarray,
    geometry: np.ndarray,
    qap_cache: np.ndarray,
    *,
    target_rho: float,
    seed: int,
    nuisance_scale: float,
    controller_profile_nuisance: float = 0.0,
) -> dict[str, float | bool]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    k = len(fresh)
    combined = np.vstack([fresh, reference])
    latent, latent_rho, _weight = choose_latent(combined, k, target_rho, rng, nuisance_scale)
    item_features = rng.standard_normal((PRIMARY_N, latent.shape[1]))
    base_logit = rng.normal(0.0, 0.75, size=PRIMARY_N)
    response = latent @ item_features.T
    # This additional low-rank controller-by-item term is independent of A0.
    # It is used only in the declared dependence sensitivity analysis to ask
    # how robust power is to shared controller-profile heterogeneity.
    if controller_profile_nuisance > 0.0:
        controller_loading = rng.standard_normal((len(combined), 3))
        shared_item_profile = rng.standard_normal((3, PRIMARY_N))
        response += controller_profile_nuisance * controller_loading @ shared_item_profile
    observed_by_shell: dict[str, np.ndarray] = {}
    true_by_shell: dict[str, np.ndarray] = {}
    for shell, amplitude in (("MEDIUM", 0.75), ("STRONG", 1.15)):
        probability = sigmoid(base_logit[None, :] + amplitude * response)
        error = (rng.random((len(combined), PRIMARY_N, 2)) < probability[:, :, None]).astype(
            np.float64
        )
        observed_by_shell[shell] = cross_block_shape(error[:k], error[k:])
        centered = probability - np.mean(probability, axis=1, keepdims=True)
        fresh_centered = centered[:k]
        reference_centered = centered[k:]
        true_by_shell[shell] = np.mean(
            np.square(fresh_centered[:, None, :] - reference_centered[None, :, :]), axis=2
        )
    observed_rank = {shell: normalized_ranks(observed_by_shell[shell]) for shell in SHELLS}
    permutation_statistics = np.mean(
        np.stack([qap_cache @ observed_rank[shell] for shell in SHELLS], axis=0), axis=0
    )
    observed = float(permutation_statistics[0])
    p_value = float(np.sum(permutation_statistics >= observed) / len(permutation_statistics))
    geometry_shell = {shell: geometry for shell in SHELLS}
    lofo = leave_one_fresh_out(geometry_shell, observed_by_shell)
    true_rho = float(np.mean([spearman_flat(geometry, true_by_shell[shell]) for shell in SHELLS]))
    return {
        "latent_rho": latent_rho,
        "true_rho": true_rho,
        "observed_rho": observed,
        "permutation_p": p_value,
        "permutation_pass": bool(observed > 0.0 and p_value <= 0.05),
        "all_lofo_positive": bool(np.all(lofo > 0.0)),
        "lofo_min": float(np.min(lofo)),
        "lofo_max": float(np.max(lofo)),
    }


def binomial_tail(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1)))


def reserve_count(k: int, *, p: float = 0.70, target: float = 0.95) -> int:
    return next(n for n in range(k, 10 * k + 1) if binomial_tail(k, n, p) >= target)


def summarize_cell(results: list[dict[str, float | bool]]) -> dict[str, float]:
    observed = np.asarray([float(row["observed_rho"]) for row in results])
    permutation = np.asarray([bool(row["permutation_pass"]) for row in results])
    lofo = np.asarray([bool(row["all_lofo_positive"]) for row in results])
    return {
        "mean_latent_rho": float(np.mean([float(row["latent_rho"]) for row in results])),
        "mean_true_rho": float(np.mean([float(row["true_rho"]) for row in results])),
        "mean_observed_rho": float(np.mean(observed)),
        "sampling_q025": float(np.quantile(observed, 0.025)),
        "sampling_q975": float(np.quantile(observed, 0.975)),
        "sampling_95_width": float(np.quantile(observed, 0.975) - np.quantile(observed, 0.025)),
        "permutation_power_or_fpr": float(np.mean(permutation)),
        "permutation_mc_se": float(
            np.sqrt(np.mean(permutation) * (1.0 - np.mean(permutation)) / len(results))
        ),
        "all_lofo_positive_rate": float(np.mean(lofo)),
        "mean_lofo_min": float(np.mean([float(row["lofo_min"]) for row in results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--planning-permutations", type=int, default=999)
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)
    reference = reference_coefficients()
    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    for k in K_CANDIDATES:
        fresh = planning_fresh_coefficients(k)
        geometry = angular_cross_block(fresh, reference)
        permutation_seed = protocol_seed(f"Q2-OOS-POWER-QAP-K{k}-V1", REFERENCE_COMMIT)
        permutations = fresh_row_permutations(k, args.planning_permutations, seed=permutation_seed)
        cache = qap_rank_cache(geometry, permutations)
        for scenario, rho in EFFECT_SCENARIOS.items():
            results = [
                simulate_once(
                    fresh,
                    reference,
                    geometry,
                    cache,
                    target_rho=rho,
                    seed=protocol_seed(
                        f"Q2-OOS-POWER-K{k}-{scenario}-R{replicate}-V1", REFERENCE_COMMIT
                    ),
                    nuisance_scale=1.0,
                )
                for replicate in range(args.replicates)
            ]
            rows.append(
                {
                    "K": k,
                    "scenario": scenario,
                    "target_rho": rho,
                    "replicates": args.replicates,
                    "planning_permutations": len(permutations),
                    "final_permutations": min(QAP_MAPS, math.factorial(k)),
                    "future_trajectories": 1200 * k,
                    **summarize_cell(results),
                }
            )
        candidate_count = reserve_count(k)
        reserve_rows.append(
            {
                "K": k,
                "candidate_count": candidate_count,
                "reserve": candidate_count - k,
                **{
                    f"probability_at_p_{str(p).replace('.', '_')}": binomial_tail(
                        k, candidate_count, p
                    )
                    for p in (0.70, 0.75, 0.775, 0.80, 0.85, 0.90)
                },
            }
        )
        for profile_nuisance in (0.50, 1.00):
            results = [
                simulate_once(
                    fresh,
                    reference,
                    geometry,
                    cache,
                    target_rho=EFFECT_SCENARIOS["THREE_QUARTER_PRIOR"],
                    seed=protocol_seed(
                        f"Q2-OOS-DEPENDENCE-K{k}-S{profile_nuisance}-R{replicate}-V1",
                        REFERENCE_COMMIT,
                    ),
                    nuisance_scale=1.0,
                    controller_profile_nuisance=profile_nuisance,
                )
                for replicate in range(args.replicates)
            ]
            sensitivity_rows.append(
                {
                    "K": k,
                    "scenario": "THREE_QUARTER_PRIOR",
                    "controller_profile_nuisance": profile_nuisance,
                    "replicates": args.replicates,
                    **summarize_cell(results),
                }
            )
    by_key = {(int(row["K"]), str(row["scenario"])): row for row in rows}
    reserve_by_k = {int(row["K"]): row for row in reserve_rows}
    decisions = []
    for k in K_CANDIDATES:
        full = by_key[(k, "FULL_PRIOR")]
        three_quarter = by_key[(k, "THREE_QUARTER_PRIOR")]
        null = by_key[(k, "NULL")]
        checks = {
            "minimum_fresh_controller_vertices": k
            >= PLANNING_CRITERIA["minimum_fresh_controller_vertices"],
            "full_prior_permutation_power": full["permutation_power_or_fpr"]
            >= PLANNING_CRITERIA["full_prior_permutation_power_min"],
            "three_quarter_prior_permutation_power": three_quarter["permutation_power_or_fpr"]
            >= PLANNING_CRITERIA["three_quarter_prior_permutation_power_min"],
            "null_false_positive_rate": null["permutation_power_or_fpr"]
            <= PLANNING_CRITERIA["null_false_positive_rate_max"],
            "full_prior_lofo_sign_stability": full["all_lofo_positive_rate"]
            >= PLANNING_CRITERIA["full_prior_all_lofo_positive_rate_min"],
            "full_prior_precision": full["sampling_95_width"]
            <= PLANNING_CRITERIA["full_prior_sampling_95_width_max"],
            "safe_bank_feasibility": reserve_by_k[k]["probability_at_p_0_7"]
            >= PLANNING_CRITERIA["safe_bank_probability_min_at_p_0_70"],
        }
        decisions.append({"K": k, "checks": checks, "pass": all(checks.values())})
    passing = [int(row["K"]) for row in decisions if row["pass"]]
    recommended = min(passing) if passing else None
    with (REVIEW / "POWER_PRECISION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (REVIEW / "RESERVE_FEASIBILITY.json").write_text(
        json.dumps(reserve_rows, indent=2, sort_keys=True) + "\n"
    )
    with (REVIEW / "CONTROLLER_DEPENDENCE_SENSITIVITY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    payload = {
        "schema_version": "q2-oos-fresh-controller-power-v1",
        "cpu_only": True,
        "prior_item_level_outcomes_accessed": False,
        "planning_input": {
            "A0_aggregate_rho": REFERENCE_A0_RHO,
            "source": "Q2 V4.1 final closeout",
        },
        "effect_scenarios": EFFECT_SCENARIOS,
        "K_values": K_CANDIDATES,
        "N": PRIMARY_N,
        "dependence": (
            "shared 300 items, two rollout blocks, 31 fixed reference columns, "
            "complete fresh-row permutation, both shells coupled"
        ),
        "precision_definition": (
            "empirical 95% width across independent synthetic N=300 panels; final "
            "inference uses the separately frozen item bootstrap"
        ),
        "planning_criteria_frozen_in_code": PLANNING_CRITERIA,
        "decisions": decisions,
        "recommended_K": recommended,
        "recommended_candidate_count": None
        if recommended is None
        else reserve_by_k[recommended]["candidate_count"],
        "power_csv_sha256": sha256(REVIEW / "POWER_PRECISION.csv"),
        "controller_dependence_sensitivity_sha256": sha256(
            REVIEW / "CONTROLLER_DEPENDENCE_SENSITIVITY.csv"
        ),
        "reference_manifest_sha256": sha256(REFERENCE_MANIFEST),
    }
    (REVIEW / "POWER_PRECISION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"recommended_K": recommended, "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
