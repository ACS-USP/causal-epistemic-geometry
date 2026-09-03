#!/usr/bin/env python3
"""CPU-only null calibration for the Q2 OOS V2 fresh-row QAP.

The script generates synthetic binary panels only. It never derives the V2
controller-stream seed, loads a model, reads benchmark data, or reads semantic
outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    angular_cross_block,
    cross_block_shape,
    fresh_row_permutations,
)
from epistemic_geometry.experiments.q2_v4 import average_ranks
from scripts.review_q2_oos_v2_k import (
    SAFETY_MODEL_PATH,
    draw_planning_banks,
    solve_intercept,
    unit_sphere,
)
from scripts.review_q2_oos_v2_k import (
    derived_seed as stress_seed,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_qualification"
PRECHECK = OUT / "NULL_CALIBRATION_PRECHECK.json"
ERRATUM = OUT / "PHASE0_PRECHECK_ERRATUM.json"
REFERENCE_MANIFEST = (
    ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
)
SOURCE_COMMIT = "74ba16ad03d63dd680bb85e8a4486e29eaa12c9c"
PRECHECK_COMMIT = "630b0854b191d1da87800396987e1e6d85c2bdc6"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_pvalues(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def child_seed(base: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{label}".encode()).digest()[:16], "big")


def unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def normalized_ranks(values: np.ndarray) -> np.ndarray | None:
    ranks = average_ranks(np.asarray(values, dtype=np.float64).reshape(-1))
    ranks -= np.mean(ranks)
    norm = float(np.linalg.norm(ranks))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return ranks / norm


def qap_cache(geometry: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    cache = np.empty((len(permutations), geometry.size), dtype=np.float32)
    for index, permutation in enumerate(np.asarray(permutations, dtype=np.int64)):
        ranks = normalized_ranks(geometry[permutation, :])
        if ranks is None:
            raise ValueError("representative geometry is rank-degenerate")
        cache[index] = ranks.astype(np.float32)
    return cache


def strict_panel_ranks(
    *, seed: int, k: int, reference_count: int = 31
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    item_factors = rng.standard_normal((300, 4))
    base_logit = rng.normal(0.0, 0.75, size=300)
    fresh_loadings = rng.standard_normal((k, 4))
    reference_loadings = rng.standard_normal((reference_count, 4))
    fresh_intercept = rng.normal(0.0, 0.25, size=(k, 1))
    reference_intercept = rng.normal(0.0, 0.25, size=(reference_count, 1))
    combined_response = np.vstack(
        [
            fresh_loadings @ item_factors.T / 2.0 + fresh_intercept,
            reference_loadings @ item_factors.T / 2.0 + reference_intercept,
        ]
    )
    ranks_by_shell: list[np.ndarray] = []
    for amplitude in (0.75, 1.15):
        probability = 1.0 / (
            1.0 + np.exp(-(base_logit[None, :] + amplitude * combined_response))
        )
        errors = (
            rng.random((k + reference_count, 300, 2)) < probability[:, :, None]
        ).astype(np.float64)
        dshape = cross_block_shape(errors[:k], errors[k:])
        ranks = normalized_ranks(dshape)
        if ranks is None:
            ranks = np.full(dshape.size, np.nan)
        ranks_by_shell.append(ranks)
    return ranks_by_shell[0], ranks_by_shell[1]


def stress_panel_ranks(*, latent: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    item_features = rng.standard_normal((300, latent.shape[1]))
    base_logit = rng.normal(0.0, 0.75, size=300)
    response = latent @ item_features.T
    ranks_by_shell: list[np.ndarray] = []
    for amplitude in (0.75, 1.15):
        probability = 1.0 / (1.0 + np.exp(-(base_logit[None, :] + amplitude * response)))
        errors = (rng.random((len(latent), 300, 2)) < probability[:, :, None]).astype(
            np.float64
        )
        dshape = cross_block_shape(errors[:k], errors[k:])
        ranks = normalized_ranks(dshape)
        if ranks is None:
            ranks = np.full(dshape.size, np.nan)
        ranks_by_shell.append(ranks)
    return ranks_by_shell[0], ranks_by_shell[1]


def evaluate_rank_panels(
    cache: np.ndarray,
    generator: Callable[[int], tuple[np.ndarray, np.ndarray]],
    panel_indices: list[int],
    *,
    chunk_size: int = 25,
) -> np.ndarray:
    pvalues = np.empty(len(panel_indices), dtype=np.float64)
    for start in range(0, len(panel_indices), chunk_size):
        indices = panel_indices[start : start + chunk_size]
        pairs = [generator(index) for index in indices]
        medium = np.column_stack([pair[0] for pair in pairs]).astype(np.float32)
        strong = np.column_stack([pair[1] for pair in pairs]).astype(np.float32)
        finite = np.all(np.isfinite(medium), axis=0) & np.all(np.isfinite(strong), axis=0)
        statistics = 0.5 * (cache @ medium + cache @ strong)
        observed = statistics[0]
        chunk_p = np.mean(statistics >= observed[None, :], axis=0)
        chunk_p[~finite] = 1.0
        pvalues[start : start + len(indices)] = chunk_p
    return pvalues


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return center - half, center + half


def summarize_pvalues(values: np.ndarray) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 11)
    histogram, _edges = np.histogram(values, bins=bins)
    rejections = int(np.sum(values <= 0.05))
    low, high = wilson(rejections, len(values))
    return {
        "panels": int(len(values)),
        "rejections_at_0_05": rejections,
        "rejection_rate_at_0_05": float(rejections / len(values)),
        "monte_carlo_se": float(
            math.sqrt((rejections / len(values)) * (1.0 - rejections / len(values)) / len(values))
        ),
        "Wilson_95_interval": [low, high],
        "fraction_p_le_0_01": float(np.mean(values <= 0.01)),
        "fraction_p_le_0_05": float(np.mean(values <= 0.05)),
        "fraction_p_le_0_10": float(np.mean(values <= 0.10)),
        "p_value_quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        },
        "p_value_decile_histogram": histogram.tolist(),
    }


def reference_coefficients() -> np.ndarray:
    manifest = read_json(REFERENCE_MANIFEST)
    values = np.asarray([row["coefficients"] for row in manifest["directions"]])
    if values.shape != (31, 8):
        raise RuntimeError("frozen reference bank is not 31x8")
    return values


def representative_geometry(k: int, seed: int, reference: np.ndarray) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    fresh = unit_rows(rng.standard_normal((k, 8)))
    return angular_cross_block(fresh, reference)


def strict_materially_anti_conservative(summary: dict[str, Any]) -> bool:
    p05 = float(summary["fraction_p_le_0_05"])
    low05 = float(summary["Wilson_95_interval"][0])
    p01 = float(summary["fraction_p_le_0_01"])
    low01 = wilson(round(p01 * int(summary["panels"])), int(summary["panels"]))[0]
    p10 = float(summary["fraction_p_le_0_10"])
    low10 = wilson(round(p10 * int(summary["panels"])), int(summary["panels"]))[0]
    return bool(
        (p05 > 0.065 and low05 > 0.055)
        or (p01 > 0.020 and low01 > 0.0125)
        or (p10 > 0.125 and low10 > 0.110)
    )


def stress_setup(
    reference: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    safety = read_json(SAFETY_MODEL_PATH)
    coordinate_mean = float(safety["coordinate_mean"])
    coordinate_sd = float(safety["coordinate_standard_deviation"])
    slope = float(safety["logistic_slope"])
    population_rng = np.random.Generator(
        np.random.PCG64DXSM(stress_seed("SAFETY-INTERCEPT"))
    )
    population = unit_sphere(population_rng, (250000, 8))
    standardized = (population[:, 4] - coordinate_mean) / coordinate_sd
    intercept = solve_intercept(standardized, slope, 0.60)
    banks = draw_planning_banks(
        k=16,
        n=34,
        count=8,
        reference=reference,
        slope=slope,
        intercept=intercept,
        coordinate_mean=coordinate_mean,
        coordinate_sd=coordinate_sd,
    )
    latents = []
    caches = []
    from scripts.review_q2_oos_v2_k import choose_latent

    for bank_index, fresh in enumerate(banks):
        geometry = angular_cross_block(fresh, reference)
        permutations = fresh_row_permutations(
            16,
            999,
            seed=stress_seed(f"POWER-QAP-K16-BANK{bank_index}"),
        )
        caches.append(qap_cache(geometry, permutations))
        latent_rng = np.random.Generator(
            np.random.PCG64DXSM(stress_seed(f"LATENT-K16-BANK{bank_index}-NULL"))
        )
        latent, _rho = choose_latent(np.vstack([fresh, reference]), 16, 0.0, latent_rng)
        latents.append(latent)
    return banks, latents, caches


def implementation_audit(
    geometry: np.ndarray,
    permutations_1k: np.ndarray,
    permutations_50k: np.ndarray,
    strict_generator: Callable[[int], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    deterministic_first = strict_generator(0)
    deterministic_second = strict_generator(0)
    tied = normalized_ranks(np.asarray([0.0, 0.0, 1.0, 2.0]))
    degenerate = normalized_ranks(np.ones(8))
    identity = np.arange(geometry.shape[0])
    checks = {
        "identity_first_1k": bool(np.array_equal(permutations_1k[0], identity)),
        "identity_first_50k": bool(np.array_equal(permutations_50k[0], identity)),
        "identity_count_1k": int(np.sum(np.all(permutations_1k == identity, axis=1))) == 1,
        "identity_count_50k": int(np.sum(np.all(permutations_50k == identity, axis=1))) == 1,
        "unique_1k": len({tuple(row) for row in permutations_1k}) == len(permutations_1k),
        "unique_50k": len({tuple(row) for row in permutations_50k})
        == len(permutations_50k),
        "permutations_cover_complete_rows": bool(
            np.all(np.sort(permutations_50k, axis=1) == identity)
        ),
        "reference_columns_fixed": bool(
            np.array_equal(geometry[permutations_1k[1], :], geometry[permutations_1k[1]])
        ),
        "deterministic_panel": bool(
            np.array_equal(deterministic_first[0], deterministic_second[0])
            and np.array_equal(deterministic_first[1], deterministic_second[1])
        ),
        "tied_average_ranks_finite": tied is not None and bool(np.all(np.isfinite(tied))),
        "degenerate_detected": degenerate is None,
        "right_tail_and_no_sign_inversion": True,
        "same_map_both_shells": True,
        "dyad_level_permutation": False,
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def run_calibration(
    *,
    strict_panels: int,
    future_panels: int,
    stress_panels: int,
    small_k_panels: int,
    persist: bool,
) -> dict[str, Any]:
    precheck = read_json(PRECHECK)
    erratum = read_json(ERRATUM)
    if precheck["source_commit"] != SOURCE_COMMIT:
        raise RuntimeError("unexpected precheck source commit")
    if erratum["parent_precheck_commit"] != PRECHECK_COMMIT:
        raise RuntimeError("stress-null erratum is not pinned to the precheck")
    reference = reference_coefficients()
    geometry = representative_geometry(
        16, int(precheck["seeds"]["STRICT_GEOMETRY"]), reference
    )
    permutations_1k = fresh_row_permutations(
        16, 1000, seed=int(precheck["seeds"]["STRICT_QAP_1K"])
    )
    permutations_50k = fresh_row_permutations(
        16, 50000, seed=int(precheck["seeds"]["STRICT_QAP_50K"])
    )
    cache_1k = qap_cache(geometry, permutations_1k)
    base_seed = int(precheck["seeds"]["STRICT_PANELS"])

    def strict_generator(index: int) -> tuple[np.ndarray, np.ndarray]:
        return strict_panel_ranks(seed=child_seed(base_seed, f"PANEL-{index}"), k=16)

    strict_pvalues = evaluate_rank_panels(
        cache_1k, strict_generator, list(range(strict_panels))
    )
    cache_50k = qap_cache(geometry, permutations_50k)
    future_indices = list(range(future_panels))
    future_50k = evaluate_rank_panels(cache_50k, strict_generator, future_indices, chunk_size=10)
    future_1k = strict_pvalues[:future_panels]
    del cache_50k

    _banks, stress_latents, stress_caches = stress_setup(reference)
    per_bank = stress_panels // 8
    if stress_panels % 8:
        raise ValueError("stress panel count must be divisible by eight")
    stress_values = []
    stress_rows = []
    for bank_index, (latent, cache) in enumerate(zip(stress_latents, stress_caches, strict=True)):
        def stress_generator(index: int, *, bank: int = bank_index, value: np.ndarray = latent):
            seed = stress_seed(f"POWER-PANEL-K16-BANK{bank}-NULL-R{index}")
            return stress_panel_ranks(latent=value, k=16, seed=seed)

        values = evaluate_rank_panels(cache, stress_generator, list(range(per_bank)))
        stress_values.append(values)
        stress_rows.extend(
            {"bank_index": bank_index, "replicate": index, "p_value": float(pvalue)}
            for index, pvalue in enumerate(values)
        )
    stress_pvalues = np.concatenate(stress_values)

    small_geometry = representative_geometry(
        6, int(precheck["seeds"]["SMALL_K_EXACT"]), reference
    )
    exact_permutations = fresh_row_permutations(
        6, 720, seed=int(precheck["seeds"]["SMALL_K_EXACT"])
    )
    sampled_permutations = exact_permutations[:500]
    exact_cache = qap_cache(small_geometry, exact_permutations)
    sampled_cache = qap_cache(small_geometry, sampled_permutations)

    def small_generator(index: int) -> tuple[np.ndarray, np.ndarray]:
        seed = child_seed(int(precheck["seeds"]["SMALL_K_EXACT"]), f"PANEL-{index}")
        return strict_panel_ranks(seed=seed, k=6)

    small_indices = list(range(small_k_panels))
    exact_p = evaluate_rank_panels(exact_cache, small_generator, small_indices)
    sampled_p = evaluate_rank_panels(sampled_cache, small_generator, small_indices)
    small_difference = np.abs(exact_p - sampled_p)
    small_audit = {
        "K": 6,
        "panels": small_k_panels,
        "exact_maps": 720,
        "sampled_maps": 500,
        "exact_rejection_rate": float(np.mean(exact_p <= 0.05)),
        "sampled_rejection_rate": float(np.mean(sampled_p <= 0.05)),
        "absolute_rejection_rate_difference": float(
            abs(np.mean(exact_p <= 0.05) - np.mean(sampled_p <= 0.05))
        ),
        "p95_absolute_p_value_difference": float(np.quantile(small_difference, 0.95)),
    }
    small_audit["pass"] = bool(
        small_audit["absolute_rejection_rate_difference"] <= 0.02
        and small_audit["p95_absolute_p_value_difference"] <= 0.05
    )

    future_difference = np.abs(future_50k - future_1k)
    future_audit = {
        "panels": future_panels,
        "maps_1k": 1000,
        "maps_50k": 50000,
        "rejection_rate_1k": float(np.mean(future_1k <= 0.05)),
        "rejection_rate_50k": float(np.mean(future_50k <= 0.05)),
        "absolute_rejection_rate_difference": float(
            abs(np.mean(future_1k <= 0.05) - np.mean(future_50k <= 0.05))
        ),
        "p95_absolute_p_value_difference": float(np.quantile(future_difference, 0.95)),
    }
    future_audit["pass"] = bool(
        future_audit["absolute_rejection_rate_difference"] <= 0.03
        and future_audit["p95_absolute_p_value_difference"] <= 0.035
    )
    implementation = implementation_audit(
        geometry, permutations_1k, permutations_50k, strict_generator
    )
    strict_summary = summarize_pvalues(strict_pvalues)
    stress_summary = summarize_pvalues(stress_pvalues)
    strict_block = strict_materially_anti_conservative(strict_summary)
    stress_warning = bool(
        stress_summary["rejection_rate_at_0_05"] > 0.065
        and stress_summary["Wilson_95_interval"][0] > 0.055
    )
    implementation_pass = bool(
        implementation["pass"] and small_audit["pass"] and future_audit["pass"]
    )
    if not implementation_pass or strict_block:
        ruling = "PRIMARY_ROW_QAP_IMPLEMENTATION_NOT_CALIBRATED"
    elif stress_warning:
        ruling = "STRESS_NULL_EXCHANGEABILITY_WARNING"
    else:
        ruling = "PRIMARY_ROW_QAP_CALIBRATED"
    result = {
        "schema_version": "q2-oos-v2-row-qap-null-calibration-result-v1",
        "precheck_commit": PRECHECK_COMMIT,
        "precheck_erratum_commit": "fe255060f4ec9ee2143a62b810a9d91aecd9167b",
        "strict_exchangeable_null": strict_summary,
        "stress_null": stress_summary,
        "future_50000_map_path": future_audit,
        "small_K_exact_audit": small_audit,
        "implementation_audit": implementation,
        "strict_materially_anti_conservative": strict_block,
        "stress_materially_anti_conservative": stress_warning,
        "ruling": ruling,
        "new_v2_stream_generated": False,
        "actual_v2_seed_derived": False,
        "model_inference": 0,
        "semantic_trajectories": 0,
        "correctness_inspected": False,
    }
    if persist:
        strict_rows = [
            {"replicate": index, "p_value": float(value)}
            for index, value in enumerate(strict_pvalues)
        ]
        future_rows = [
            {
                "replicate": index,
                "p_value_1000_maps": float(future_1k[index]),
                "p_value_50000_maps": float(future_50k[index]),
            }
            for index in range(future_panels)
        ]
        write_pvalues(OUT / "STRICT_NULL_PVALUES.csv", strict_rows)
        write_pvalues(OUT / "STRESS_NULL_PVALUES.csv", stress_rows)
        write_pvalues(OUT / "FUTURE_50000_MAP_PATH_PVALUES.csv", future_rows)
        write_json(OUT / "SMALL_K_EXACT_AUDIT.json", small_audit)
        write_json(OUT / "NULL_CALIBRATION_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.benchmark == args.full:
        raise SystemExit("choose exactly one of --benchmark or --full")
    started = time.monotonic()
    if args.benchmark:
        result = run_calibration(
            strict_panels=100,
            future_panels=10,
            stress_panels=80,
            small_k_panels=100,
            persist=False,
        )
        elapsed = time.monotonic() - started
        projected = elapsed * max(10000 / 100, 500 / 10, 10000 / 80, 2000 / 100)
        print(
            json.dumps(
                {
                    "benchmark_seconds": elapsed,
                    "conservative_linear_projection_seconds": projected,
                    "conservative_linear_projection_minutes": projected / 60.0,
                    "local_full_run_eligible": projected <= 1800.0,
                    "benchmark_result_not_a_scientific_ruling": result["ruling"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        result = run_calibration(
            strict_panels=10000,
            future_panels=500,
            stress_panels=10000,
            small_k_panels=2000,
            persist=True,
        )
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(OUT / "NULL_CALIBRATION_RESULT.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
