#!/usr/bin/env python3
"""CPU-only design and dependence-aware power simulation for Q2 V4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from epistemic_geometry.experiments.q2_v4 import (
    average_ranks,
    blind_spot_shape_matrices,
    coefficient_bank_checks,
    controller_permutations,
    orthonormal_source_subspace,
    protocol_seed,
    sample_coefficient_bank,
    spearman,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "q2_v4_intervention_subspace_design"
SOURCE_COMMIT = "691ced3779b7b9de0184b73023bf40cba87f5008"
FAMILIES = {
    "CONTROL_FLOW_PATH_COVERAGE",
    "MUTATION_ALIAS_CAUSALITY",
    "LOOP_BOUNDARY_ACCOUNTING",
    "HYPOTHESIS_BRANCH_ELIMINATION",
}
K_VALUES = (16, 20, 24, 32)
N_VALUES = (200, 300)
RHO_VALUES = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _angular(values: np.ndarray) -> np.ndarray:
    unit = _unit_rows(values)
    result = 1.0 - np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(result, 0.0)
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _candidate_embeddings(coefficients: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rank = coefficients.shape[1]
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    q0, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
    q1, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
    static_transform = q0 @ np.diag(np.geomspace(1.0, 2.0, rank)) @ q0.T
    finite_transform = q1 @ np.diag(np.geomspace(0.6, 2.8, rank)) @ q1.T
    a0 = _unit_rows(coefficients)
    a1 = _unit_rows(coefficients @ static_transform)
    linear = coefficients @ finite_transform
    quadratic = np.square(coefficients @ q0) - np.mean(
        np.square(coefficients @ q0), axis=0, keepdims=True
    )
    a2 = _unit_rows(np.concatenate([linear, 0.65 * quadratic], axis=1))
    return {"A0": a0, "A1": a1, "A2": a2}


def _finite_specific_embeddings(
    coefficients: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Planning scenario where finite response carries genuinely new structure.

    The quadratic lift is not a proposed V4 metric implementation.  It is a
    controller-dependent synthetic alternative used only to test whether the
    design can attribute a finite-response advantage when one truly exists.
    """

    ordinary = _candidate_embeddings(coefficients, seed)
    quadratic = np.einsum("ki,kj->kij", coefficients, coefficients).reshape(
        len(coefficients), -1
    )
    ordinary["A2"] = _unit_rows(quadratic)
    return ordinary


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = average_ranks(values)
    centered = ranks - np.mean(ranks)
    return centered / np.linalg.norm(centered)


def _qap_cache(
    geometries: dict[str, np.ndarray], permutations: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(sorted(geometries))
    cache = np.empty((len(names), len(permutations), len(_upper(next(iter(geometries.values()))))))
    upper = np.triu_indices(next(iter(geometries.values())).shape[0], 1)
    for metric_index, name in enumerate(names):
        matrix = geometries[name]
        for permutation_index, permutation in enumerate(permutations):
            permuted = matrix[np.ix_(permutation, permutation)]
            cache[metric_index, permutation_index] = _normalized_ranks(permuted[upper])
    return names, cache


def _binomial_sign_p(positive: int, total: int) -> float:
    return float(sum(math.comb(total, k) for k in range(positive, total + 1)) / (2**total))


def _choose_mixture(
    finite_embedding: np.ndarray,
    nuisance_embedding: np.ndarray,
    target_rho: float,
) -> tuple[float, float]:
    finite_distance = _upper(_angular(finite_embedding))
    nuisance_distance = _upper(_angular(nuisance_embedding))
    weights = np.linspace(0.0, 1.0, 201)
    mixtures = (
        weights[:, None] * finite_distance[None, :]
        + (1.0 - weights[:, None]) * nuisance_distance[None, :]
    )
    # Continuous random designs have no ties here; stable double argsort gives
    # zero-based ranks and avoids 201 separate Python-level Spearman calls.
    mixed_ranks = np.argsort(np.argsort(mixtures, axis=1, kind="mergesort"), axis=1)
    mixed_centered = mixed_ranks - np.mean(mixed_ranks, axis=1, keepdims=True)
    finite_rank = _normalized_ranks(finite_distance)
    achieved = (mixed_centered @ finite_rank) / np.linalg.norm(mixed_centered, axis=1)
    index = int(np.argmin(np.abs(achieved - target_rho)))
    return float(weights[index]), float(achieved[index])


def _pair_shape(baseline: np.ndarray, condition: np.ndarray) -> float:
    d0 = baseline[:, 0] - condition[:, 0]
    d1 = baseline[:, 1] - condition[:, 1]
    panel = float(np.mean(d0 * d1) - np.mean(d0) * np.mean(d1))
    return panel * len(d0) / (len(d0) - 1.0)


def _simulate_once(
    embeddings: dict[str, np.ndarray],
    qap_cache: np.ndarray,
    metric_names: tuple[str, ...],
    *,
    n_items: int,
    target_rho: float,
    seed: int,
) -> dict[str, float | bool]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    size = len(embeddings["A2"])
    finite = embeddings["A2"]
    nuisance = _unit_rows(rng.standard_normal(finite.shape))
    weight, latent_rho = _choose_mixture(finite, nuisance, target_rho)
    latent = np.concatenate(
        [np.sqrt(weight) * finite, np.sqrt(1.0 - weight) * nuisance], axis=1
    )
    item_features = rng.standard_normal((n_items, latent.shape[1]))
    base_logit = rng.normal(0.0, 0.75, size=n_items)
    # Each controller embedding has unit norm, so its dot product with an
    # isotropic item feature already has unit variance.  Further division by
    # dimension would manufacture a near-zero controller regime.
    response = latent @ item_features.T
    probabilities = {
        "MEDIUM": _sigmoid(base_logit[None, :] + 0.75 * response),
        "STRONG": _sigmoid(base_logit[None, :] + 1.15 * response),
    }
    outcome_matrices: dict[str, np.ndarray] = {}
    error_arrays: dict[str, np.ndarray] = {}
    true_matrices: dict[str, np.ndarray] = {}
    for shell, probability in probabilities.items():
        error = (rng.random((size, n_items, 2)) < probability[:, :, None]).astype(np.float64)
        error_arrays[shell] = error
        outcome_matrices[shell] = blind_spot_shape_matrices(error)["shape_item_population"]
        centered = probability - np.mean(probability, axis=1, keepdims=True)
        true_matrices[shell] = (
            np.mean(np.square(centered[:, None, :] - centered[None, :, :]), axis=2)
        )
    outcome_rank = np.mean(
        np.vstack([_normalized_ranks(_upper(outcome_matrices[s])) for s in ("MEDIUM", "STRONG")]),
        axis=0,
    )
    null = np.einsum("mpe,e->pm", qap_cache, outcome_rank)
    observed = null[0]
    max_null = np.max(null, axis=1)
    observed_max = float(np.max(observed))
    global_p = float(np.sum(max_null >= observed_max) / len(max_null))
    adjusted = np.asarray(
        [np.sum(max_null >= value) / len(max_null) for value in observed], dtype=np.float64
    )
    a2_index = metric_names.index("A2")
    a0_index = metric_names.index("A0")
    a1_index = metric_names.index("A1")
    differences = np.column_stack(
        [null[:, a2_index] - null[:, a0_index], null[:, a2_index] - null[:, a1_index]]
    )
    max_difference_null = np.max(differences, axis=1)
    observed_differences = differences[0]
    superiority_adjusted = np.asarray(
        [
            np.sum(max_difference_null >= value) / len(max_difference_null)
            for value in observed_differences
        ]
    )
    baseline_probability = _sigmoid(base_logit)
    baseline_error = (
        rng.random((n_items, 2)) < baseline_probability[:, None]
    ).astype(np.float64)
    radial = []
    for controller in range(size):
        medium = _pair_shape(baseline_error, error_arrays["MEDIUM"][controller])
        strong = _pair_shape(baseline_error, error_arrays["STRONG"][controller])
        radial.append(strong - medium)
    radial = np.asarray(radial)
    radial_positive = int(np.sum(radial > 0.0))
    true_rho = float(
        np.mean(
            [
                spearman(_upper(_angular(finite)), _upper(true_matrices[shell]))
                for shell in ("MEDIUM", "STRONG")
            ]
        )
    )
    return {
        "target_rho": target_rho,
        "latent_rho": latent_rho,
        "true_rho": true_rho,
        "observed_a0": float(observed[a0_index]),
        "observed_a1": float(observed[a1_index]),
        "observed_a2": float(observed[a2_index]),
        "omnibus_pass": global_p <= 0.05,
        "a0_attribution_pass": adjusted[a0_index] <= 0.05 and observed[a0_index] > 0.0,
        "a1_attribution_pass": adjusted[a1_index] <= 0.05 and observed[a1_index] > 0.0,
        "a2_attribution_pass": adjusted[a2_index] <= 0.05 and observed[a2_index] > 0.0,
        "a2_superiority_pass": bool(
            np.all(observed_differences >= 0.10)
            and np.all(superiority_adjusted <= 0.05)
        ),
        "radial_pass": _binomial_sign_p(radial_positive, size) <= 0.05
        and float(np.median(radial)) > 0.0,
    }


def _source_summary() -> dict[str, object]:
    bank = json.loads(
        (ROOT / "review/q2_v3_amendment1_execution/Q2_V3_DIRECTION_BANK.json").read_text()
    )
    selected = sorted(
        (name, metadata)
        for name, metadata in bank["directions"].items()
        if metadata["family_id"] in FAMILIES
    )
    vectors = [
        np.load(ROOT / metadata["path"]).astype(np.float64).reshape(-1)
        for _, metadata in selected
    ]
    source = np.column_stack(vectors)
    basis, summary = orthonormal_source_subspace(source, relative_singular_threshold=1e-6)
    gram = (source / np.linalg.norm(source, axis=0)[None, :]).T @ (
        source / np.linalg.norm(source, axis=0)[None, :]
    )
    return {
        "source_commit": SOURCE_COMMIT,
        "rank_rule": "retain sigma_i/sigma_1 >= 1e-6; numerical conditioning only",
        "direction_ids": [name for name, _ in selected],
        "direction_paths": [metadata["path"] for _, metadata in selected],
        "vector_hashes": [metadata["vector_hash"] for _, metadata in selected],
        "ambient_dimension": int(source.shape[0]),
        "source_direction_count": int(source.shape[1]),
        "exact_rank": summary.exact_rank,
        "retained_rank": summary.retained_rank,
        "singular_values": summary.singular_values,
        "relative_singular_values": summary.relative_singular_values,
        "condition_number": summary.condition_number,
        "entropy_effective_rank": summary.entropy_effective_rank,
        "stable_rank": summary.stable_rank,
        "unit_gram": gram.tolist(),
        "orthonormality_max_error": float(
            np.max(
                np.abs(basis.T @ basis - np.eye(len(summary.singular_values)))
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=120)
    parser.add_argument("--permutations", type=int, default=499)
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)
    source_summary = _source_summary()
    (REVIEW / "SUBSPACE_NUMERICAL_AUDIT.json").write_text(
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n"
    )

    rows: list[dict[str, object]] = []
    superiority_rows: list[dict[str, object]] = []
    bank_rows: list[dict[str, object]] = []
    rank = int(source_summary["retained_rank"])
    for size in K_VALUES:
        bank_seed = protocol_seed(f"Q2-V4-POWER-BANK-K{size}-V1", SOURCE_COMMIT)
        coefficients = sample_coefficient_bank(rank, size, seed=bank_seed)
        checks = coefficient_bank_checks(coefficients)
        bank_rows.append({"K": size, "seed": bank_seed, **checks})
        embeddings = _candidate_embeddings(coefficients, bank_seed ^ 0xA2A2)
        geometries = {name: _angular(value) for name, value in embeddings.items()}
        permutations = controller_permutations(
            size,
            args.permutations,
            seed=protocol_seed(f"Q2-V4-POWER-QAP-K{size}-V1", SOURCE_COMMIT),
        )
        metric_names, cache = _qap_cache(geometries, permutations)
        for n_items in N_VALUES:
            for rho in RHO_VALUES:
                results = [
                    _simulate_once(
                        embeddings,
                        cache,
                        metric_names,
                        n_items=n_items,
                        target_rho=rho,
                        seed=protocol_seed(
                            f"Q2-V4-SIM-K{size}-N{n_items}-R{rho}-I{replicate}",
                            SOURCE_COMMIT,
                        ),
                    )
                    for replicate in range(args.replicates)
                ]
                a2_values = np.asarray([float(result["observed_a2"]) for result in results])
                a0_values = np.asarray([float(result["observed_a0"]) for result in results])
                a1_values = np.asarray([float(result["observed_a1"]) for result in results])
                row = {
                    "K": size,
                    "N": n_items,
                    "target_rho": rho,
                    "mean_true_rho": float(np.mean([result["true_rho"] for result in results])),
                    "omnibus_rate": float(np.mean([result["omnibus_pass"] for result in results])),
                    "a2_attribution_rate": float(
                        np.mean([result["a2_attribution_pass"] for result in results])
                    ),
                    "a2_superiority_rate": float(
                        np.mean([result["a2_superiority_pass"] for result in results])
                    ),
                    "radial_rate": float(np.mean([result["radial_pass"] for result in results])),
                    "a2_rho_mean": float(np.mean(a2_values)),
                    "a0_rho_mean": float(np.mean(a0_values)),
                    "a1_rho_mean": float(np.mean(a1_values)),
                    "a2_minus_best_static_mean": float(
                        np.mean(a2_values - np.maximum(a0_values, a1_values))
                    ),
                    "a2_rho_mc95_width": float(
                        np.quantile(a2_values, 0.975)
                        - np.quantile(a2_values, 0.025)
                    ),
                    "replicates": args.replicates,
                    "qap_permutations": args.permutations,
                }
                rows.append(row)
        specific_embeddings = _finite_specific_embeddings(coefficients, bank_seed ^ 0xA2A2)
        specific_geometries = {
            name: _angular(value) for name, value in specific_embeddings.items()
        }
        specific_names, specific_cache = _qap_cache(specific_geometries, permutations)
        for n_items in N_VALUES:
            for rho in (0.25, 0.30, 0.40, 0.50, 0.60):
                results = [
                    _simulate_once(
                        specific_embeddings,
                        specific_cache,
                        specific_names,
                        n_items=n_items,
                        target_rho=rho,
                        seed=protocol_seed(
                            f"Q2-V4-SUPERIORITY-K{size}-N{n_items}-R{rho}-I{replicate}",
                            SOURCE_COMMIT,
                        ),
                    )
                    for replicate in range(args.replicates)
                ]
                a0 = np.asarray([float(result["observed_a0"]) for result in results])
                a1 = np.asarray([float(result["observed_a1"]) for result in results])
                a2 = np.asarray([float(result["observed_a2"]) for result in results])
                superiority_rows.append(
                    {
                        "K": size,
                        "N": n_items,
                        "target_rho": rho,
                        "mean_true_rho": float(
                            np.mean([result["true_rho"] for result in results])
                        ),
                        "a0_rho_mean": float(np.mean(a0)),
                        "a1_rho_mean": float(np.mean(a1)),
                        "a2_rho_mean": float(np.mean(a2)),
                        "a2_minus_best_static_mean": float(
                            np.mean(a2 - np.maximum(a0, a1))
                        ),
                        "omnibus_rate": float(
                            np.mean([result["omnibus_pass"] for result in results])
                        ),
                        "a2_attribution_rate": float(
                            np.mean([result["a2_attribution_pass"] for result in results])
                        ),
                        "a2_superiority_rate": float(
                            np.mean([result["a2_superiority_pass"] for result in results])
                        ),
                        "replicates": args.replicates,
                        "qap_permutations": args.permutations,
                    }
                )
    with (REVIEW / "POWER_SIMULATION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    with (REVIEW / "SUPERIORITY_POWER_SIMULATION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(superiority_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(superiority_rows)
    (REVIEW / "COEFFICIENT_BANK_SIMULATION_CHECKS.json").write_text(
        json.dumps(bank_rows, indent=2, sort_keys=True) + "\n"
    )
    metadata = {
        "schema_version": "q2-v4-power-simulation-v1",
        "cpu_only": True,
        "semantic_outcomes_read": False,
        "source_commit": SOURCE_COMMIT,
        "K_values": K_VALUES,
        "N_values": N_VALUES,
        "rho_values": RHO_VALUES,
        "replicates_per_cell": args.replicates,
        "qap_permutations_per_replicate": args.permutations,
        "candidate_metrics": ["A0", "A1", "A2"],
        "dependence": (
            "controller embeddings, shared item logits, shared items, two "
            "Bernoulli rollout blocks, shell-coupled controller permutations"
        ),
        "shape_estimator": (
            "N/(N-1) corrected two-rollout item-population shape estimator"
        ),
        "superiority_margin": 0.10,
        "note": "planning simulation; final V4 QAP uses 50,000 permutations",
        "scenarios": {
            "correlated_metric_ladder": (
                "A2 combines anisotropic linear and quadratic response "
                "features; used for primary K/N power table"
            ),
            "finite_specific_superiority": (
                "A2 uses a synthetic quadratic controller lift with low "
                "static-geometry correlation; used only to assess G3 "
                "attribution when finite response truly adds structure"
            ),
        },
    }
    (REVIEW / "POWER_SIMULATION_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    digest = hashlib.sha256((REVIEW / "POWER_SIMULATION.csv").read_bytes()).hexdigest()
    print(json.dumps({"rows": len(rows), "power_csv_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
