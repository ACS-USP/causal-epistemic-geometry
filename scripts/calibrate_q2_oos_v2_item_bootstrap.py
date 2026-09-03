#!/usr/bin/env python3
"""Model-free calibration of Q2 OOS V2 item-level uncertainty methods.

The simulation preserves N=300, R=2, 16 fresh controllers, 31 fixed
references, and two coupled shells.  It uses only tracked coefficient geometry
and synthetic Bernoulli outcomes.  No model, benchmark text, or semantic
output is loaded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRECHECK = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_semantic_execution"
    / "Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_PRECHECK.json"
)
ANALYSIS_SOURCE = ROOT / "scripts/analyze_q2_oos_v2_semantic.py"
FRESH_MANIFEST = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
    / "V2_CANDIDATE_BANK_MANIFEST.json"
)
REFERENCE_MANIFEST = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json"

SCENARIOS: dict[str, dict[str, float]] = {
    "NULL": {
        "signal": 0.0,
        "nuisance": 0.65,
        "difficulty_sd": 1.0,
        "sparse_probability": 1.0,
        "controller_scale_sd": 0.0,
    },
    "WEAK_POSITIVE": {
        "signal": 1.0,
        "nuisance": 0.75,
        "difficulty_sd": 1.0,
        "sparse_probability": 1.0,
        "controller_scale_sd": 0.0,
    },
    "MODERATE_POSITIVE": {
        "signal": 1.8,
        "nuisance": 0.55,
        "difficulty_sd": 1.0,
        "sparse_probability": 1.0,
        "controller_scale_sd": 0.0,
    },
    "OBSERVED_LIKE": {
        "signal": 2.8,
        "nuisance": 0.35,
        "difficulty_sd": 1.0,
        "sparse_probability": 1.0,
        "controller_scale_sd": 0.0,
    },
    "HETEROGENEOUS_CONTROLLER_EFFECTS": {
        "signal": 3.0,
        "nuisance": 0.40,
        "difficulty_sd": 1.0,
        "sparse_probability": 1.0,
        "controller_scale_sd": 0.45,
    },
    "HETEROGENEOUS_ITEM_DIFFICULTY": {
        "signal": 3.0,
        "nuisance": 0.40,
        "difficulty_sd": 2.0,
        "sparse_probability": 1.0,
        "controller_scale_sd": 0.0,
    },
    "SPARSE_BLIND_SPOT_DIFFERENCES": {
        "signal": 10.0,
        "nuisance": 0.35,
        "difficulty_sd": 1.0,
        "sparse_probability": 0.15,
        "controller_scale_sd": 0.0,
    },
    "HEAVY_TIES_NEAR_DEGENERATE": {
        "signal": 0.15,
        "nuisance": 0.15,
        "difficulty_sd": 0.35,
        "sparse_probability": 0.35,
        "controller_scale_sd": 0.0,
    },
}
SHELL_MULTIPLIERS = {"MEDIUM": 0.80, "STRONG": 1.20}
STATISTICS = ("global_equal_shell_mean", "median_row_association")
NORMAL_975 = 1.959963984540054
SYNTHETIC_PARAMETERIZATION = {
    "status": "FROZEN_BEFORE_FINAL_CALIBRATION",
    "selection_basis": (
        "synthetic development panels only; target named weak, moderate, and "
        "observed-like full-sample signal regimes before method coverage was run"
    ),
    "real_result_role": (
        "the sealed A0 global rho 0.6430547122 is used only as the explicit "
        "planning target for OBSERVED_LIKE"
    ),
    "method_outputs_used_for_parameterization": False,
    "development_panels": 8,
    "development_resamples_per_method": 1,
    "development_full_sample_global_means": {
        "WEAK_POSITIVE": 0.134,
        "MODERATE_POSITIVE": 0.364,
        "OBSERVED_LIKE": 0.628,
    },
}
METHOD_TARGETS = {
    "A_ordinary_item_bootstrap": (
        "item-population uncertainty for the compound R=2 Dshape-to-Spearman estimator; "
        "historical percentile implementation"
    ),
    "B_bayesian_multiplier": (
        "item-population uncertainty under continuous exchangeable item weights with full support"
    ),
    "C_subsampling": (
        "item-population uncertainty estimated from without-replacement "
        "subsets and root-m calibration"
    ),
    "D_delete_d": (
        "local item-deletion sensitivity and approximate item-population standard error"
    ),
    "E_controller_cluster": (
        "fresh-controller-population uncertainty conditional on the observed "
        "item panel; not item uncertainty"
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_analysis_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "q2_oos_v2_analysis_for_calibration", ANALYSIS_SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import frozen analysis source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_coefficients(analysis: Any) -> tuple[np.ndarray, np.ndarray]:
    fresh_payload = read_json(FRESH_MANIFEST)
    reference_payload = read_json(REFERENCE_MANIFEST)
    fresh_by_id = {
        str(row["candidate_id"]): np.asarray(row["coefficients"], dtype=np.float64)
        for row in fresh_payload["candidates"]
    }
    reference_by_id = {
        str(row["candidate_id"]): np.asarray(row["coefficients"], dtype=np.float64)
        for row in reference_payload["candidates"]
    }
    fresh = np.stack([fresh_by_id[name] for name in analysis.FRESH_IDS])
    reference = np.stack([reference_by_id[name] for name in analysis.REFERENCE_IDS])
    fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    return fresh, reference


def logistic(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def shape_from_probabilities(
    fresh_probabilities: np.ndarray, reference_probabilities: np.ndarray
) -> np.ndarray:
    differences = fresh_probabilities[:, None, :] - reference_probabilities[None, :, :]
    mean = np.mean(differences, axis=2)
    return np.mean(np.square(differences), axis=2) - np.square(mean)


def weighted_shape(
    fresh: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / np.sum(normalized)
    correction_denominator = 1.0 - float(np.sum(np.square(normalized)))
    if correction_denominator <= 0.0:
        return np.full((fresh.shape[0], reference.shape[0]), np.nan)
    d0 = fresh[:, None, :, 0] - reference[None, :, :, 0]
    d1 = fresh[:, None, :, 1] - reference[None, :, :, 1]
    panel = np.einsum("frn,n->fr", d0 * d1, normalized, optimize=True)
    mean0 = np.einsum("frn,n->fr", d0, normalized, optimize=True)
    mean1 = np.einsum("frn,n->fr", d1, normalized, optimize=True)
    return (panel - mean0 * mean1) / correction_denominator


def statistics(
    analysis: Any,
    geometry: dict[str, np.ndarray],
    shapes: dict[str, np.ndarray],
) -> dict[str, float]:
    rows = analysis.row_associations(geometry, shapes)
    shell_values = [
        analysis.spearman_flat(geometry[shell], shapes[shell]) for shell in analysis.SHELLS
    ]
    return {
        "global_equal_shell_mean": float(np.mean(shell_values)),
        "median_row_association": float(np.median(rows)),
    }


def resample_statistics(
    analysis: Any,
    geometry: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    weight_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {name: np.empty(len(weight_matrix), dtype=np.float64) for name in STATISTICS}
    for index, weights in enumerate(weight_matrix):
        shapes = {
            shell: weighted_shape(fresh[shell], reference[shell], weights)
            for shell in analysis.SHELLS
        }
        values = statistics(analysis, geometry, shapes)
        for name in STATISTICS:
            output[name][index] = values[name]
    return output


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def subsampling_interval(
    values: np.ndarray, full: float, fraction: float
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    scale = math.sqrt(fraction)
    return float(full - scale * (high - full)), float(full - scale * (low - full))


def jackknife_interval(
    values: np.ndarray, full: float, items: int, deleted: int
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan"), float("nan")
    variance = (items - deleted) / (deleted * len(finite)) * float(
        np.sum(np.square(finite - np.mean(finite)))
    )
    standard_error = math.sqrt(max(0.0, variance))
    return (
        float(full - NORMAL_975 * standard_error),
        float(full + NORMAL_975 * standard_error),
    )


def controller_cluster_values(
    analysis: Any,
    geometry: dict[str, np.ndarray],
    shapes: dict[str, np.ndarray],
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, np.ndarray]:
    output = {name: np.empty(resamples, dtype=np.float64) for name in STATISTICS}
    row_values = analysis.row_associations(geometry, shapes)
    for index in range(resamples):
        sample = rng.integers(0, len(row_values), size=len(row_values))
        shell_values = [
            analysis.spearman_flat(geometry[shell][sample], shapes[shell][sample])
            for shell in analysis.SHELLS
        ]
        output["global_equal_shell_mean"][index] = float(np.mean(shell_values))
        output["median_row_association"][index] = float(np.median(row_values[sample]))
    return output


def synthetic_probabilities(
    coefficients: np.ndarray,
    nuisance: np.ndarray,
    item_features: np.ndarray,
    difficulty: np.ndarray,
    controller_scales: np.ndarray,
    parameters: dict[str, float],
    shell: str,
    sparse_mask: np.ndarray,
) -> np.ndarray:
    loadings = (
        parameters["signal"] * coefficients * controller_scales[:, None]
        + parameters["nuisance"] * nuisance
    )
    interaction = loadings @ item_features.T / math.sqrt(coefficients.shape[1])
    interaction *= sparse_mask[None, :]
    logits = difficulty[None, :] + SHELL_MULTIPLIERS[shell] * interaction
    return logistic(logits)


def simulate_panel(task: dict[str, Any]) -> dict[str, Any]:
    analysis = load_analysis_module()
    fresh_coefficients = np.asarray(task["fresh_coefficients"], dtype=np.float64)
    reference_coefficients = np.asarray(task["reference_coefficients"], dtype=np.float64)
    geometry = {
        shell: np.asarray(task["geometry"][shell], dtype=np.float64)
        for shell in analysis.SHELLS
    }
    scenario = str(task["scenario"])
    parameters = SCENARIOS[scenario]
    panel_index = int(task["panel_index"])
    resamples = int(task["resamples"])
    panel_seed = int(task["panel_seed"])
    resampling_seed = int(task["resampling_seed"])
    rng_panel = np.random.Generator(np.random.PCG64DXSM(panel_seed))
    rng_resampling = np.random.Generator(np.random.PCG64DXSM(resampling_seed))
    all_coefficients = np.concatenate([fresh_coefficients, reference_coefficients], axis=0)
    nuisance = rng_panel.normal(size=all_coefficients.shape)
    nuisance /= np.linalg.norm(nuisance, axis=1, keepdims=True)
    controller_scales = np.exp(
        rng_panel.normal(
            scale=parameters["controller_scale_sd"], size=len(all_coefficients)
        )
    )

    truth_items = 6000
    truth_features = rng_panel.normal(size=(truth_items, all_coefficients.shape[1]))
    truth_difficulty = rng_panel.normal(
        scale=parameters["difficulty_sd"], size=truth_items
    )
    truth_mask = (
        rng_panel.random(truth_items) < parameters["sparse_probability"]
    ).astype(np.float64)
    truth_shapes: dict[str, np.ndarray] = {}
    for shell in analysis.SHELLS:
        probabilities = synthetic_probabilities(
            all_coefficients,
            nuisance,
            truth_features,
            truth_difficulty,
            controller_scales,
            parameters,
            shell,
            truth_mask,
        )
        truth_shapes[shell] = shape_from_probabilities(
            probabilities[: len(fresh_coefficients)],
            probabilities[len(fresh_coefficients) :],
        )
    truth = statistics(analysis, geometry, truth_shapes)

    items = int(task["items"])
    item_features = rng_panel.normal(size=(items, all_coefficients.shape[1]))
    difficulty = rng_panel.normal(scale=parameters["difficulty_sd"], size=items)
    sparse_mask = (
        rng_panel.random(items) < parameters["sparse_probability"]
    ).astype(np.float64)
    fresh_errors: dict[str, np.ndarray] = {}
    reference_errors: dict[str, np.ndarray] = {}
    for shell in analysis.SHELLS:
        probabilities = synthetic_probabilities(
            all_coefficients,
            nuisance,
            item_features,
            difficulty,
            controller_scales,
            parameters,
            shell,
            sparse_mask,
        )
        draws = (
            rng_panel.random((len(all_coefficients), items, 2))
            < probabilities[:, :, None]
        )
        fresh_errors[shell] = draws[: len(fresh_coefficients)].astype(np.float64)
        reference_errors[shell] = draws[len(fresh_coefficients) :].astype(np.float64)
    full_shapes = {
        shell: weighted_shape(
            fresh_errors[shell], reference_errors[shell], np.ones(items, dtype=np.float64)
        )
        for shell in analysis.SHELLS
    }
    full = statistics(analysis, geometry, full_shapes)

    methods: dict[str, dict[str, Any]] = {}
    ordinary_counts = np.zeros((resamples, items), dtype=np.float64)
    for index in range(resamples):
        ordinary_counts[index] = np.bincount(
            rng_resampling.integers(0, items, size=items), minlength=items
        )
    ordinary = resample_statistics(
        analysis, geometry, fresh_errors, reference_errors, ordinary_counts
    )
    methods["A_ordinary_item_bootstrap"] = {
        "values": ordinary,
        "interval_type": "percentile",
    }

    multiplier = rng_resampling.exponential(size=(resamples, items))
    methods["B_bayesian_multiplier"] = {
        "values": resample_statistics(
            analysis, geometry, fresh_errors, reference_errors, multiplier
        ),
        "interval_type": "percentile",
    }

    for fraction in (0.5, 0.632, 0.75, 0.9):
        size = int(round(items * fraction))
        weights = np.zeros((resamples, items), dtype=np.float64)
        for index in range(resamples):
            weights[
                index,
                rng_resampling.choice(items, size=size, replace=False),
            ] = 1.0
        methods[f"C_subsampling_{fraction:.3f}"] = {
            "values": resample_statistics(
                analysis, geometry, fresh_errors, reference_errors, weights
            ),
            "interval_type": "subsampling",
            "fraction": fraction,
        }

    delete_groups = int(task["delete_groups"])
    for fraction in (0.1, 0.25):
        deleted = int(round(items * fraction))
        weights = np.ones((delete_groups, items), dtype=np.float64)
        for index in range(delete_groups):
            weights[
                index,
                rng_resampling.choice(items, size=deleted, replace=False),
            ] = 0.0
        methods[f"D_delete_d_{fraction:.2f}"] = {
            "values": resample_statistics(
                analysis, geometry, fresh_errors, reference_errors, weights
            ),
            "interval_type": "jackknife",
            "deleted": deleted,
        }

    methods["E_controller_cluster"] = {
        "values": controller_cluster_values(
            analysis, geometry, full_shapes, rng_resampling, resamples
        ),
        "interval_type": "percentile",
    }

    output_methods: dict[str, dict[str, Any]] = {}
    for method, specification in methods.items():
        output_methods[method] = {}
        for statistic_name in STATISTICS:
            values = np.asarray(specification["values"][statistic_name], dtype=np.float64)
            if specification["interval_type"] == "subsampling":
                interval = subsampling_interval(
                    values, full[statistic_name], float(specification["fraction"])
                )
            elif specification["interval_type"] == "jackknife":
                interval = jackknife_interval(
                    values,
                    full[statistic_name],
                    items,
                    int(specification["deleted"]),
                )
            else:
                interval = percentile_interval(values)
            finite = values[np.isfinite(values)]
            output_methods[method][statistic_name] = {
                "resampling_median_minus_full": (
                    float(np.median(finite) - full[statistic_name])
                    if len(finite)
                    else float("nan")
                ),
                "resampling_median_minus_truth": (
                    float(np.median(finite) - truth[statistic_name])
                    if len(finite)
                    else float("nan")
                ),
                "interval_low": interval[0],
                "interval_high": interval[1],
                "interval_width": float(interval[1] - interval[0]),
                "covers_truth": bool(
                    np.isfinite(interval[0])
                    and interval[0] <= truth[statistic_name] <= interval[1]
                ),
                "degenerate": bool(len(finite) != len(values)),
            }
    tie_fraction = float(
        np.mean(
            [
                1.0 - len(np.unique(full_shapes[shell])) / full_shapes[shell].size
                for shell in analysis.SHELLS
            ]
        )
    )
    return {
        "scenario": scenario,
        "panel_index": panel_index,
        "panel_seed": panel_seed,
        "resampling_seed": resampling_seed,
        "truth": truth,
        "full": full,
        "full_minus_truth": {
            name: float(full[name] - truth[name]) for name in STATISTICS
        },
        "full_Dshape_tie_fraction": tie_fraction,
        "methods": output_methods,
    }


def wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1.0 + NORMAL_975**2 / trials
    center = (proportion + NORMAL_975**2 / (2.0 * trials)) / denominator
    half = NORMAL_975 * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + NORMAL_975**2 / (4.0 * trials**2)
    ) / denominator
    return float(center - half), float(center + half)


def aggregate(results: list[dict[str, Any]], precheck: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scenario in SCENARIOS:
        panels = [row for row in results if row["scenario"] == scenario]
        methods: dict[str, Any] = {}
        method_names = sorted(panels[0]["methods"])
        for method in method_names:
            methods[method] = {}
            for statistic_name in STATISTICS:
                entries = [row["methods"][method][statistic_name] for row in panels]
                coverage_count = sum(bool(entry["covers_truth"]) for entry in entries)
                coverage = coverage_count / len(entries)
                coverage_interval = wilson(coverage_count, len(entries))
                centering = np.asarray(
                    [entry["resampling_median_minus_full"] for entry in entries],
                    dtype=np.float64,
                )
                truth_bias = np.asarray(
                    [entry["resampling_median_minus_truth"] for entry in entries],
                    dtype=np.float64,
                )
                widths = np.asarray(
                    [entry["interval_width"] for entry in entries], dtype=np.float64
                )
                degeneracy = float(np.mean([entry["degenerate"] for entry in entries]))
                screen = bool(
                    0.90 <= coverage <= 0.99
                    and abs(float(np.nanmean(centering))) <= 0.05
                    and degeneracy <= 0.01
                )
                methods[method][statistic_name] = {
                    "panels": len(entries),
                    "coverage": coverage,
                    "coverage_wilson_95": list(coverage_interval),
                    "mean_resampling_median_minus_full": float(np.nanmean(centering)),
                    "median_resampling_median_minus_full": float(np.nanmedian(centering)),
                    "mean_resampling_median_minus_truth": float(
                        np.nanmean(truth_bias)
                    ),
                    "mean_interval_width": float(np.nanmean(widths)),
                    "degeneracy_fraction": degeneracy,
                    "calibration_screen_pass": screen,
                }
            if method.startswith("C_subsampling"):
                target_key = "C_subsampling"
            elif method.startswith("D_delete_d"):
                target_key = "D_delete_d"
            else:
                target_key = method
            methods[method]["uncertainty_population"] = METHOD_TARGETS[target_key]
        output[scenario] = {
            "parameters": SCENARIOS[scenario],
            "truth_mean": {
                name: float(np.mean([row["truth"][name] for row in panels]))
                for name in STATISTICS
            },
            "full_estimate_mean": {
                name: float(np.mean([row["full"][name] for row in panels]))
                for name in STATISTICS
            },
            "full_estimator_bias": {
                name: float(np.mean([row["full_minus_truth"][name] for row in panels]))
                for name in STATISTICS
            },
            "mean_full_Dshape_tie_fraction": float(
                np.mean([row["full_Dshape_tie_fraction"] for row in panels])
            ),
            "methods": methods,
        }

    eligible_methods = [
        "A_ordinary_item_bootstrap",
        "B_bayesian_multiplier",
        "C_subsampling_0.500",
        "C_subsampling_0.632",
        "C_subsampling_0.750",
        "C_subsampling_0.900",
        "D_delete_d_0.10",
        "D_delete_d_0.25",
    ]
    pass_counts = {}
    for method in eligible_methods:
        pass_counts[method] = sum(
            bool(output[scenario]["methods"][method][statistic]["calibration_screen_pass"])
            for scenario in SCENARIOS
            for statistic in STATISTICS
        )
    full_required = len(SCENARIOS) * len(STATISTICS)
    alternatives = [method for method, count in pass_counts.items() if count == full_required]
    ordinary_failed_scenarios = [
        scenario
        for scenario in SCENARIOS
        if not all(
            output[scenario]["methods"]["A_ordinary_item_bootstrap"][statistic][
                "calibration_screen_pass"
            ]
            for statistic in STATISTICS
        )
    ]
    ruling = (
        "Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED"
        if len(ordinary_failed_scenarios) >= 2
        else "Q2_OOS_V2_ITEM_BOOTSTRAP_IMPLEMENTATION_CLEAN_INTERPRETATION_RECALIBRATED"
    )
    return {
        "schema_version": "q2-oos-v2-item-bootstrap-synthetic-calibration-v1",
        "label": "POST_HOC_DIAGNOSTIC_ONLY",
        "status": "SYNTHETIC_CALIBRATION_COMPLETE",
        "dimensions": precheck["structural_dimensions"],
        "simulation_protocol": precheck["simulation_protocol"],
        "synthetic_parameterization": SYNTHETIC_PARAMETERIZATION,
        "scenarios": output,
        "method_screen_pass_counts": pass_counts,
        "method_screen_total": full_required,
        "fully_screened_alternatives": alternatives,
        "controller_cluster_role": (
            "SEPARATE_CONTROLLER_POPULATION_UNCERTAINTY_NOT_ITEM_UNCERTAINTY_REPLACEMENT"
        ),
        "ordinary_item_bootstrap_failed_screens": (
            full_required - pass_counts["A_ordinary_item_bootstrap"]
        ),
        "ordinary_item_bootstrap_failed_scenarios": ordinary_failed_scenarios,
        "diagnostic_ruling_from_frozen_rule": ruling,
        "frozen_primary_classification": "Q2_OOS_V2_A0_PASS",
        "new_semantic_trajectories": 0,
        "Qwen_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--development-panels", type=int)
    parser.add_argument("--development-resamples", type=int)
    args = parser.parse_args()
    precheck = read_json(PRECHECK)
    if precheck.get("status") != "FROZEN_PRE_DIAGNOSTIC":
        raise RuntimeError("diagnostic precheck is not frozen")
    analysis = load_analysis_module()
    fresh_coefficients, reference_coefficients = load_coefficients(analysis)
    matrices, _metadata = analysis.load_prediction_matrices()
    geometry = {
        shell: matrices[f"A0_{shell}_FRESH_REFERENCE"] for shell in analysis.SHELLS
    }
    recomputed = 1.0 - fresh_coefficients @ reference_coefficients.T
    geometry_difference = max(
        float(np.max(np.abs(recomputed - geometry[shell]))) for shell in analysis.SHELLS
    )
    if geometry_difference > 1e-12:
        raise RuntimeError("coefficient geometry does not reproduce frozen A0")
    frozen_protocol = precheck["simulation_protocol"]
    panels = int(
        args.development_panels
        if args.development_panels is not None
        else frozen_protocol["outer_panels_per_scenario"]
    )
    resamples = int(
        args.development_resamples
        if args.development_resamples is not None
        else frozen_protocol["resamples_per_method_per_panel"]
    )
    development = args.development_panels is not None or args.development_resamples is not None
    tasks = []
    panel_seed = int(frozen_protocol["seeds"]["synthetic_panel_generation"])
    resampling_seed = int(frozen_protocol["seeds"]["synthetic_resampling"])
    for scenario_index, scenario in enumerate(SCENARIOS):
        for panel_index in range(panels):
            tasks.append(
                {
                    "scenario": scenario,
                    "panel_index": panel_index,
                    "panel_seed": (
                        panel_seed + scenario_index * 1_000_000 + panel_index
                    ),
                    "resampling_seed": (
                        resampling_seed + scenario_index * 1_000_000 + panel_index
                    ),
                    "items": int(precheck["structural_dimensions"]["items"]),
                    "resamples": resamples,
                    "delete_groups": int(frozen_protocol["delete_d_groups_per_fraction"]),
                    "fresh_coefficients": fresh_coefficients.tolist(),
                    "reference_coefficients": reference_coefficients.tolist(),
                    "geometry": {shell: geometry[shell].tolist() for shell in analysis.SHELLS},
                }
            )
    if args.workers <= 1:
        results = [simulate_panel(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(simulate_panel, tasks, chunksize=1))
    result = aggregate(results, precheck)
    result["development_mode"] = development
    result["tie_and_degeneracy_audit"] = {
        "method": (
            "deterministic exact tie counts over every simulated panel; "
            "no jitter or stochastic tie breaking"
        ),
        "frozen_seed": frozen_protocol["seeds"]["method_tie_and_degeneracy_audit"],
        "seed_consumed": False,
        "reason_seed_not_consumed": (
            "the frozen audit is exhaustive and deterministic, so random "
            "subsampling is unnecessary"
        ),
    }
    result["observed_coefficient_A0_max_abs_difference"] = geometry_difference
    result["source_hashes"] = {
        "precheck": sha256_file(PRECHECK),
        "analysis_source": sha256_file(ANALYSIS_SOURCE),
        "fresh_coefficient_manifest": sha256_file(FRESH_MANIFEST),
        "reference_coefficient_manifest": sha256_file(REFERENCE_MANIFEST),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "development_mode": development,
                "ruling": result["diagnostic_ruling_from_frozen_rule"],
                "fully_screened_alternatives": result["fully_screened_alternatives"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
