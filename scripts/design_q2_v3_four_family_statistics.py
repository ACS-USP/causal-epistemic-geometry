#!/usr/bin/env python3
"""CPU-only four-family Q2 V3 statistical-design simulation.

This script uses synthetic Bernoulli error panels with two independent
rollouts.  It imports no model, parser, journal, correctness result, or frozen
geometry matrix.  Its purpose is prospective gate calibration and panel-size
planning, not evidence about Q2.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_v3 import stable_rank  # noqa: E402
from epistemic_geometry.experiments.q2_v3_four_family import (  # noqa: E402
    SHELLS,
    cross_family_edges,
    effective_rank,
    exact_qap,
    family_balanced_rho,
    family_leverage,
    lodo_rhos,
)

OUTPUT = ROOT / "review/q2_v3_four_family_statistical_redesign"
LEDGER = ROOT / "review/q2_m3_qualification_cruxeval_provenance/CRUXEVAL_PROVENANCE_LEDGER.jsonl"
AMENDMENT = ROOT / "review/q2_v3_amendment1_freeze"

SEED = 2026082504
RHO_GRID = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40)
N_GRID = (200, 300, 400)
REPETITIONS = 240
MAX_N = max(N_GRID)

SCENARIOS = {
    "BALANCED": {
        "item_noise": 0.55,
        "family_item_noise": 0.20,
        "controller_item_noise": 0.12,
        "family_signal": (1.0, 1.0, 1.0, 1.0),
        "shell_signal": (1.0, 1.0),
    },
    "HETEROGENEOUS": {
        "item_noise": 0.90,
        "family_item_noise": 0.40,
        "controller_item_noise": 0.25,
        "family_signal": (1.15, 0.90, 0.70, 0.55),
        "shell_signal": (1.0, 0.75),
    },
    "ONE_WEAK_FAMILY": {
        "item_noise": 0.70,
        "family_item_noise": 0.30,
        "controller_item_noise": 0.18,
        "family_signal": (1.0, 1.0, 1.0, 0.10),
        "shell_signal": (1.0, 0.90),
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def matrix_from_edges(values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=np.float64)
    for value, (left, right) in zip(values, cross_family_edges(), strict=True):
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def rank_standardize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    centered = ranks - np.mean(ranks)
    return centered / np.std(centered)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def expected_distance_matrices(logits: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    probabilities = sigmoid(logits)
    for shell_index, shell in enumerate(SHELLS):
        matrix = np.zeros((8, 8), dtype=np.float64)
        for left, right in cross_family_edges():
            difference = probabilities[shell_index, left] - probabilities[shell_index, right]
            value = float(difference**2)
            matrix[left, right] = value
            matrix[right, left] = value
        output[shell] = matrix
    return output


def make_geometry(
    target: dict[str, np.ndarray],
    desired_rho: float,
    rng: np.random.Generator,
    family_signal: tuple[float, ...],
    shell_signal: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], float]:
    edges = cross_family_edges()
    noise_by_shell: dict[str, np.ndarray] = {}
    signal_by_shell: dict[str, np.ndarray] = {}
    for shell_index, shell in enumerate(SHELLS):
        target_values = np.asarray([target[shell][left, right] for left, right in edges])
        signal = rank_standardize(target_values)
        weights = np.asarray(
            [
                0.5 * (family_signal[left // 2] + family_signal[right // 2])
                * shell_signal[shell_index]
                for left, right in edges
            ]
        )
        signal_by_shell[shell] = signal * weights
        family_pair_noise = rng.normal(size=(4, 4))
        family_pair_noise = 0.5 * (family_pair_noise + family_pair_noise.T)
        direction_noise = rng.normal(size=8)
        noise = np.asarray(
            [
                0.45 * family_pair_noise[left // 2, right // 2]
                + 0.35 * (direction_noise[left] + direction_noise[right])
                + rng.normal(scale=0.75)
                for left, right in edges
            ]
        )
        noise_by_shell[shell] = rank_standardize(noise)

    if desired_rho == 0.0:
        geometry = {
            shell: matrix_from_edges(noise_by_shell[shell] + 2.5) for shell in SHELLS
        }
        achieved = float(family_balanced_rho(geometry, target)["aggregate"])
        return geometry, achieved

    best: tuple[float, dict[str, np.ndarray], float] | None = None
    for mixture in np.linspace(0.0, 1.0, 101):
        geometry = {
            shell: matrix_from_edges(
                mixture * signal_by_shell[shell]
                + math.sqrt(max(0.0, 1.0 - mixture**2)) * noise_by_shell[shell]
                + 2.5
            )
            for shell in SHELLS
        }
        achieved = float(family_balanced_rho(geometry, target)["aggregate"])
        candidate = (abs(achieved - desired_rho), geometry, achieved)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def generate_panel(
    rng: np.random.Generator,
    scenario: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    family_base = rng.normal(scale=0.65, size=4)
    location_base = rng.normal(scale=0.35, size=2)
    controller_base = rng.normal(scale=0.20, size=8)
    medium = np.asarray(
        [
            family_base[index // 2] + location_base[index % 2] + controller_base[index]
            for index in range(8)
        ]
    )
    radial = rng.normal(loc=0.18, scale=0.15, size=8)
    logits = np.vstack([medium, medium + radial])
    target = expected_distance_matrices(logits)

    global_item = rng.normal(scale=scenario["item_noise"], size=(MAX_N, 1))
    family_item = rng.normal(scale=scenario["family_item_noise"], size=(MAX_N, 4))
    controller_item = rng.normal(scale=scenario["controller_item_noise"], size=(MAX_N, 8))
    contributions = np.empty((MAX_N, 2, len(cross_family_edges())), dtype=np.float64)
    for shell_index in range(2):
        item_logits = (
            logits[shell_index][None, :]
            + global_item
            + family_item[:, np.arange(8) // 2]
            + controller_item
        )
        probabilities = sigmoid(item_logits)
        errors = rng.binomial(1, probabilities[:, :, None], size=(MAX_N, 8, 2))
        for edge_index, (left, right) in enumerate(cross_family_edges()):
            contributions[:, shell_index, edge_index] = (
                (errors[:, left, 0] - errors[:, right, 0])
                * (errors[:, left, 1] - errors[:, right, 1])
            )
    return contributions, target


def outcomes_from_contributions(
    contributions: np.ndarray, item_count: int
) -> dict[str, np.ndarray]:
    return {
        shell: matrix_from_edges(np.mean(contributions[:item_count, shell_index, :], axis=0))
        for shell_index, shell in enumerate(SHELLS)
    }


def bootstrap_family_rhos(
    geometry: dict[str, np.ndarray],
    contributions: np.ndarray,
    item_count: int,
    rng: np.random.Generator,
    resamples: int = 200,
) -> np.ndarray:
    """Fast item-cluster bootstrap used only inside the planning simulation."""

    counts = rng.multinomial(
        item_count,
        np.full(item_count, 1.0 / item_count),
        size=resamples,
    )
    edges = cross_family_edges()
    aggregate = np.zeros(resamples, dtype=np.float64)
    for shell_index, shell in enumerate(SHELLS):
        outcome_edges = counts @ contributions[:item_count, shell_index, :] / item_count
        geometry_edges = np.asarray([geometry[shell][left, right] for left, right in edges])
        for family in range(4):
            incident = np.asarray(
                [
                    index
                    for index, (left, right) in enumerate(edges)
                    if left // 2 == family or right // 2 == family
                ]
            )
            geometry_rank = rank_standardize(geometry_edges[incident])
            values = outcome_edges[:, incident]
            order = np.argsort(values, axis=1, kind="mergesort")
            ranks = np.empty_like(values)
            np.put_along_axis(
                ranks,
                order,
                np.broadcast_to(np.arange(len(incident), dtype=np.float64), values.shape),
                axis=1,
            )
            ranks -= np.mean(ranks, axis=1)[:, None]
            ranks /= np.maximum(np.linalg.norm(ranks, axis=1)[:, None], 1e-12)
            geometry_rank /= np.linalg.norm(geometry_rank)
            aggregate += (ranks @ geometry_rank) / 8.0
    return aggregate


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
    }


def run_power_simulation() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    accumulators: dict[tuple[str, float, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for scenario_name, scenario in SCENARIOS.items():
        for desired_rho in RHO_GRID:
            for _replicate in range(REPETITIONS):
                contributions, target = generate_panel(rng, scenario)
                geometries: dict[str, dict[str, np.ndarray]] = {}
                achieved: dict[str, float] = {}
                for metric, target_rho in (
                    ("M0", max(0.0, desired_rho - 0.10)),
                    ("M1", max(0.0, desired_rho - 0.10)),
                    ("M2", desired_rho),
                ):
                    geometry, actual = make_geometry(
                        target,
                        target_rho,
                        rng,
                        tuple(scenario["family_signal"]),
                        tuple(scenario["shell_signal"]),
                    )
                    geometries[metric] = geometry
                    achieved[metric] = actual
                for item_count in N_GRID:
                    outcomes = outcomes_from_contributions(contributions, item_count)
                    qap = exact_qap(geometries, outcomes)
                    m2 = family_balanced_rho(geometries["M2"], outcomes)
                    lodo = np.asarray(lodo_rhos(geometries["M2"], outcomes))
                    family_values = np.asarray(list(m2["family_summary"].values()))
                    shell_values = np.asarray(list(m2["shell_summary"].values()))
                    bootstrap = bootstrap_family_rhos(
                        geometries["M2"], contributions, item_count, rng
                    )
                    bootstrap_lower = float(np.quantile(bootstrap, 0.025))
                    bootstrap_width = float(
                        np.quantile(bootstrap, 0.975) - bootstrap_lower
                    )
                    key = (scenario_name, desired_rho, item_count)
                    bucket = accumulators[key]
                    observed = float(m2["aggregate"])
                    bucket["achieved_true_rho"].append(achieved["M2"])
                    bucket["observed_rho"].append(observed)
                    bucket["estimation_error"].append(observed - achieved["M2"])
                    bucket["bootstrap_width"].append(bootstrap_width)
                    bucket["bootstrap_lower_positive"].append(float(bootstrap_lower > 0.0))
                    bucket["global_qap_reject"].append(float(qap["global_p"] <= 0.05))
                    bucket["m2_maxT_reject"].append(
                        float(qap["single_step_maxT_adjusted_p"]["M2"] <= 0.05)
                    )
                    bucket["families_3_of_4_positive"].append(float(np.sum(family_values > 0) >= 3))
                    bucket["families_4_of_4_positive"].append(float(np.all(family_values > 0)))
                    bucket["both_shells_positive"].append(float(np.all(shell_values > 0)))
                    bucket["lodo_8_of_8_positive"].append(float(np.all(lodo > 0)))
                    bucket["lodo_7_of_8_positive"].append(float(np.sum(lodo > 0) >= 7))
                    bucket["rho_gate"].append(float(observed >= 0.25))
                    bucket["recommended_full_gate"].append(
                        float(
                            observed >= 0.25
                            and qap["single_step_maxT_adjusted_p"]["M2"] <= 0.05
                            and np.sum(family_values > 0) >= 3
                            and np.all(shell_values > 0)
                            and np.all(lodo > 0)
                            and bootstrap_lower > 0.0
                        )
                    )
    rows: list[dict[str, Any]] = []
    for (scenario, rho, item_count), bucket in sorted(accumulators.items()):
        observed_summary = summarize(bucket["observed_rho"])
        estimation_error = np.asarray(bucket["estimation_error"])
        rows.append(
            {
                "scenario": scenario,
                "nominal_rho": rho,
                "N": item_count,
                "repetitions": REPETITIONS,
                "achieved_true_rho_mean": float(np.mean(bucket["achieved_true_rho"])),
                "observed_rho_mean": observed_summary["mean"],
                "observed_rho_q025": observed_summary["q025"],
                "observed_rho_q975": observed_summary["q975"],
                "observed_interval_width": observed_summary["q975"] - observed_summary["q025"],
                "estimation_bias": float(np.mean(estimation_error)),
                "estimation_rmse": float(np.sqrt(np.mean(np.square(estimation_error)))),
                "median_item_bootstrap_width": float(np.median(bucket["bootstrap_width"])),
                "bootstrap_lower_positive_rate": float(
                    np.mean(bucket["bootstrap_lower_positive"])
                ),
                "global_qap_rejection_rate": float(np.mean(bucket["global_qap_reject"])),
                "m2_maxT_rejection_rate": float(np.mean(bucket["m2_maxT_reject"])),
                "family_3of4_rate": float(np.mean(bucket["families_3_of_4_positive"])),
                "family_4of4_rate": float(np.mean(bucket["families_4_of_4_positive"])),
                "both_shells_positive_rate": float(np.mean(bucket["both_shells_positive"])),
                "lodo_8of8_rate": float(np.mean(bucket["lodo_8_of_8_positive"])),
                "lodo_7of8_rate": float(np.mean(bucket["lodo_7_of_8_positive"])),
                "rho_gate_rate": float(np.mean(bucket["rho_gate"])),
                "recommended_full_gate_rate": float(
                    np.mean(bucket["recommended_full_gate"])
                ),
            }
        )
    return rows


def geometry_simulation() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 1)
    rows: list[dict[str, Any]] = []
    for regime in ("BALANCED", "MODERATE_COMMON_AXIS", "PATHOLOGICAL_DOMINATION"):
        for _ in range(2_000):
            random = rng.normal(size=(8, 64))
            if regime == "BALANCED":
                vectors = random
            elif regime == "MODERATE_COMMON_AXIS":
                common = rng.normal(size=(1, 64))
                vectors = 0.78 * random + 0.63 * common
            else:
                common = rng.normal(size=(1, 64))
                cluster = common + 0.10 * random
                outlier_axis = rng.normal(size=(1, 64))
                cluster[0] = outlier_axis + 0.10 * random[0]
                vectors = cluster
            unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            values = np.asarray(
                [
                    math.sqrt(max(0.0, 2.0 - 2.0 * float(unit[left] @ unit[right])))
                    for left, right in cross_family_edges()
                ]
            )
            leverage = family_leverage(values)
            rows.append(
                {
                    "regime": regime,
                    "effective_rank": effective_rank(vectors),
                    "max_family_leverage": max(leverage.values()),
                    "q90_q10": float(np.quantile(values, 0.9) - np.quantile(values, 0.1)),
                }
            )
    return rows


def condition_number_simulation() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 4)
    rows: list[dict[str, Any]] = []
    for dyads in (24, 40):
        for correlation in (0.0, 0.80, 0.95, 0.99):
            condition_numbers = []
            covariance = np.full((3, 3), correlation)
            np.fill_diagonal(covariance, 1.0)
            for _ in range(5_000):
                features = rng.multivariate_normal(np.zeros(3), covariance, size=dyads)
                features -= np.mean(features, axis=0)
                features /= np.std(features, axis=0)
                singular = np.linalg.svd(features, compute_uv=False)
                condition_numbers.append(float(singular[0] / singular[-1]))
            values = np.asarray(condition_numbers)
            rows.append(
                {
                    "dyads_per_shell": dyads,
                    "predictor_correlation": correlation,
                    "repetitions": len(values),
                    "condition_number_median": float(np.median(values)),
                    "condition_number_q95": float(np.quantile(values, 0.95)),
                    "condition_number_le_30_rate": float(np.mean(values <= 30.0)),
                }
            )
    return rows


def radial_simulation() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 2)
    rows: list[dict[str, Any]] = []
    for true_effect in (0.0, 0.03, 0.05, 0.08):
        for item_count in N_GRID:
            counters: dict[str, list[float]] = defaultdict(list)
            for _ in range(1_000):
                family_effect = rng.normal(true_effect, 0.025, size=4)
                item_family = rng.normal(scale=0.20, size=(item_count, 4))
                item_direction = rng.normal(scale=0.18, size=(item_count, 8))
                contributions = np.asarray(
                    [
                        family_effect[direction // 2]
                        + item_family[:, direction // 2]
                        + item_direction[:, direction]
                        for direction in range(8)
                    ]
                ).T
                direction_values = np.mean(contributions, axis=0)
                family_values = np.asarray(
                    [np.mean(direction_values[2 * family : 2 * family + 2]) for family in range(4)]
                )
                aggregate = float(np.median(direction_values))
                standard_error = float(
                    np.std(np.median(contributions, axis=1), ddof=1) / math.sqrt(item_count)
                )
                lower = aggregate - 1.96 * standard_error
                counters["dirs_7of8"].append(float(np.sum(direction_values > 0) >= 7))
                counters["dirs_8of8"].append(float(np.all(direction_values > 0)))
                counters["families_4of4"].append(float(np.all(family_values > 0)))
                counters["bootstrap_proxy_lower_positive"].append(float(lower > 0))
                counters["recommended_radial_gate"].append(
                    float(
                        aggregate > 0
                        and np.sum(direction_values > 0) >= 7
                        and np.all(family_values > 0)
                        and lower > 0
                    )
                )
            rows.append(
                {
                    "true_effect": true_effect,
                    "N": item_count,
                    **{key: float(np.mean(value)) for key, value in counters.items()},
                }
            )
    return rows


def superiority_simulation() -> list[dict[str, Any]]:
    """Supplementary paired metric-margin precision simulation."""

    rng = np.random.default_rng(SEED + 3)
    rows = []
    covariance = np.asarray(
        [
            [1.0, 0.65, 0.65],
            [0.65, 1.0, 0.65],
            [0.65, 0.65, 1.0],
        ]
    )
    for margin in (0.0, 0.05, 0.10, 0.15):
        for item_count in N_GRID:
            item_scale = 0.10 * math.sqrt(200 / item_count)
            controller_scale = 0.07
            item_error = rng.multivariate_normal(
                np.zeros(3), item_scale**2 * covariance, size=20_000
            )
            controller_error = rng.multivariate_normal(
                np.zeros(3), controller_scale**2 * covariance, size=20_000
            )
            estimates = np.asarray([0.25, 0.25, 0.25 + margin]) + item_error + controller_error
            delta0 = estimates[:, 2] - estimates[:, 0]
            delta1 = estimates[:, 2] - estimates[:, 1]
            paired_se = math.sqrt(2.0 * (1.0 - 0.65)) * item_scale
            rows.append(
                {
                    "true_m2_margin": margin,
                    "N": item_count,
                    "point_margin_both_ge_0_10_rate": float(
                        np.mean((delta0 >= 0.10) & (delta1 >= 0.10))
                    ),
                    "paired_item_lower_both_positive_rate": float(
                        np.mean(
                            (delta0 - 1.96 * paired_se > 0.0)
                            & (delta1 - 1.96 * paired_se > 0.0)
                        )
                    ),
                    "combined_effect_and_precision_rate": float(
                        np.mean(
                            (delta0 >= 0.10)
                            & (delta1 >= 0.10)
                            & (delta0 - 1.96 * paired_se > 0.0)
                            & (delta1 - 1.96 * paired_se > 0.0)
                        )
                    ),
                }
            )
    return rows


def summarize_geometry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for regime in sorted({row["regime"] for row in rows}):
        selected = [row for row in rows if row["regime"] == regime]
        record: dict[str, Any] = {"regime": regime, "repetitions": len(selected)}
        for field in ("effective_rank", "max_family_leverage", "q90_q10"):
            values = np.asarray([row[field] for row in selected])
            record[f"{field}_median"] = float(np.median(values))
            record[f"{field}_q05"] = float(np.quantile(values, 0.05))
            record[f"{field}_q95"] = float(np.quantile(values, 0.95))
        record["effective_rank_ge_4_rate"] = float(
            np.mean([row["effective_rank"] >= 4.0 for row in selected])
        )
        record["max_leverage_le_0_375_rate"] = float(
            np.mean([row["max_family_leverage"] <= 0.375 for row in selected])
        )
        record["max_leverage_le_0_40_rate"] = float(
            np.mean([row["max_family_leverage"] <= 0.40 for row in selected])
        )
        record["angular_range_ge_0_20_rate"] = float(
            np.mean([row["q90_q10"] >= 0.20 for row in selected])
        )
        output.append(record)
    return output


def panel_manifest(selected_n: int) -> dict[str, Any]:
    ledger = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    allowed_fields = ("item_id", "provenance_class", "canonical_content")
    eligible = [
        {field: row[field] for field in allowed_fields}
        for row in ledger
        if row["provenance_class"] == "C"
    ]
    existing_primary = json.loads((AMENDMENT / "PRIMARY_PANEL_MANIFEST.json").read_text())
    inherited_ids = list(existing_primary["item_ids"])
    disjoint_manifests = (
        "SOURCE_CONSTRUCTION_MANIFEST.json",
        "SOURCE_VALIDATION_MANIFEST.json",
        "SHELL_CALIBRATION_MANIFEST.json",
        "M1_COVARIANCE_MANIFEST.json",
    )
    excluded = {
        item_id
        for filename in disjoint_manifests
        for item_id in json.loads((AMENDMENT / filename).read_text())["item_ids"]
    }
    eligible_lookup = {row["item_id"]: row for row in eligible}
    candidates = [row for row in eligible if row["item_id"] not in set(inherited_ids) | excluded]
    candidates.sort(
        key=lambda row: (
            stable_rank("Q2-V3-FOUR-FAMILY-PANEL-EXPANSION-V1", row["item_id"]),
            row["item_id"],
        )
    )
    selected_ids = inherited_ids + [
        row["item_id"] for row in candidates[: selected_n - len(inherited_ids)]
    ]
    ordered_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    universe_ids = sorted(eligible_lookup)
    universe_hash = hashlib.sha256("\n".join(universe_ids).encode()).hexdigest()
    return {
        "schema_version": "q2-v3-four-family-panel-design-v1",
        "status": "PROSPECTIVE_DESIGN_LOCK_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "evidence_class": "historical-item / prospective-controller same-domain validation",
        "dataset_repo": "cruxeval-org/cruxeval",
        "dataset_revision": "b96af0450242eb4da433032b90998f25588a5d0f",
        "eligible_class": "C",
        "eligible_count": len(eligible),
        "eligible_universe_ids": universe_ids,
        "eligible_universe_sha256": universe_hash,
        "inherited_primary_count": len(inherited_ids),
        "inherited_primary_ordered_ids_sha256": existing_primary["ordered_ids_sha256"],
        "expansion_namespace": "Q2-V3-FOUR-FAMILY-PANEL-EXPANSION-V1",
        "disjoint_allocation_files": list(disjoint_manifests),
        "disjoint_excluded_ids": sorted(excluded),
        "selected_n": selected_n,
        "selected_ids": selected_ids,
        "selected_ordered_ids_sha256": ordered_hash,
        "selected_items": [eligible_lookup[item_id] for item_id in selected_ids],
        "outcome_fields_loaded": [],
        "correctness_values_read": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    power_rows = run_power_simulation()
    geometry_raw = geometry_simulation()
    geometry_summary = summarize_geometry(geometry_raw)
    condition_rows = condition_number_simulation()
    radial_rows = radial_simulation()
    superiority_rows = superiority_simulation()
    write_csv(OUTPUT / "SIMULATION_POWER_PRECISION.csv", power_rows)
    write_csv(OUTPUT / "IDENTIFIABILITY_SIMULATION.csv", geometry_summary)
    write_csv(OUTPUT / "CONDITION_NUMBER_SIMULATION.csv", condition_rows)
    write_csv(OUTPUT / "RADIAL_INFERENCE_SIMULATION.csv", radial_rows)
    write_csv(OUTPUT / "M2_SUPERIORITY_SIMULATION.csv", superiority_rows)
    write_json(
        OUTPUT / "SIMULATION_DESIGN.json",
        {
            "schema_version": "q2-v3-four-family-dependent-simulation-v1",
            "seed": SEED,
            "repetitions_per_power_cell": REPETITIONS,
            "rho_grid": list(RHO_GRID),
            "N_grid": list(N_GRID),
            "scenarios": SCENARIOS,
            "dependence": {
                "items": "shared item latent and all controllers/shells move together",
                "controllers": "shared controller Bernoulli errors induce dyad covariance",
                "families": "shared family item effects and family signal multipliers",
                "shells": "same base direction with shell-specific logits and signal multiplier",
                "rollouts": "two independent Bernoulli draws; canonical product estimator",
            },
            "semantic_data": "SYNTHETIC_ONLY",
            "historical_correctness_read": False,
        },
    )
    write_json(OUTPUT / "FOUR_FAMILY_PRIMARY_PANEL_MANIFEST.json", panel_manifest(300))
    print(
        json.dumps(
            {
                "phase": "q2_v3_four_family_statistical_design",
                "power_cells": len(power_rows),
                "geometry_regimes": len(geometry_summary),
                "radial_cells": len(radial_rows),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
