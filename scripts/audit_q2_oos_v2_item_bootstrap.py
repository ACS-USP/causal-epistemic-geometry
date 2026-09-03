#!/usr/bin/env python3
"""Post-hoc diagnostic audit of the Q2 OOS V2 item bootstrap.

This script is intentionally outcome-read-only.  It operates on the sealed
scored JSONL and numeric arrays, never on raw generations, and cannot modify
the frozen primary result.  Its first responsibility is to reproduce the
prospectively secondary 50,000-resample item bootstrap exactly while recording
support, multiplicity, tie, and degeneracy diagnostics that the original
closeout did not retain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SOURCE = ROOT / "scripts/analyze_q2_oos_v2_semantic.py"
PRECHECK = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_semantic_execution"
    / "Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_PRECHECK.json"
)
RELEASE_ANALYSIS = (
    ROOT
    / "review/q2_oos_fresh_controller_design/v2_semantic_execution"
    / "Q2_OOS_V2_SEMANTIC_ANALYSIS.json"
)

EXPECTED_PRECHECK_STATUS = "FROZEN_PRE_DIAGNOSTIC"
EXPECTED_LABEL = "POST_HOC_DIAGNOSTIC_ONLY"
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "3a0c861aaec49dfa615db0b87f04f22c8a97f82a4182f85d7cf27a366174ddba"
)
EXPECTED_RELEASE_ANALYSIS_SHA256 = (
    "97913256d32dcbdfd30fb247bdf925ed0ad0d6a8d39da29a0195bfd7845987c5"
)
EXPECTED_FRESH_SCORES_SHA256 = (
    "9f03d96d40839e228d6cfb55408ea056e262fbf7e9aef2e863080e035e4b721b"
)
EXPECTED_ERROR_ARRAYS_SHA256 = (
    "6c0e555f7cccba0c41415c7605d161e84f99ba705b6430eaac53d2527b05086c"
)
EXPECTED_DSHAPE_SHA256 = (
    "a6a6b4889e2c86df04ce42c4415281dde82af0d2deb1347b8083015e95089ea5"
)
EXPECTED_DTOTAL_SHA256 = (
    "354db2363a845c654eafa00a3865b28ee04158978dc035a8128d95cf58a05ed9"
)
EXPECTED_HISTORICAL_SCORES_SHA256 = (
    "a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_analysis_module() -> Any:
    if sha256_file(ANALYSIS_SOURCE) != EXPECTED_ANALYSIS_SOURCE_SHA256:
        raise RuntimeError("frozen analysis source hash mismatch")
    specification = importlib.util.spec_from_file_location(
        "q2_oos_v2_frozen_analysis", ANALYSIS_SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import frozen analysis source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_precheck() -> dict[str, Any]:
    payload = read_json(PRECHECK)
    if payload.get("status") != EXPECTED_PRECHECK_STATUS:
        raise RuntimeError("diagnostic precheck is not frozen")
    if payload.get("label") != EXPECTED_LABEL:
        raise RuntimeError("diagnostic label mismatch")
    if payload["frozen_primary"]["classification"] != "Q2_OOS_V2_A0_PASS":
        raise RuntimeError("frozen primary classification changed")
    if payload["frozen_primary"]["mutable"] is not False:
        raise RuntimeError("frozen primary unexpectedly mutable")
    return payload


def load_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                # Fail closed if a scored artifact unexpectedly contains raw content.
                if "raw_output" in row or "generated_token_ids" in row:
                    raise RuntimeError(
                        "scored artifact unexpectedly contains raw generation content"
                    )
                rows.append(row)
    return rows


def rebuild_fresh_errors(
    analysis: Any,
    fresh_scores_path: Path,
    item_ids: list[str],
) -> dict[str, np.ndarray]:
    rows = load_scores(fresh_scores_path)
    expected = len(analysis.FRESH_IDS) * len(analysis.SHELLS) * len(item_ids) * 2
    if len(rows) != expected:
        raise RuntimeError("fresh scored-row count mismatch")
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        if key in scores:
            raise RuntimeError("duplicate fresh scored key")
        scores[key] = row
    errors, _valid, _evaluable, _correct = analysis.build_error_arrays(item_ids, scores)
    return errors


def stack_fresh(analysis: Any, errors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        shell: np.stack(
            [errors[f"{controller}_{shell}"] for controller in analysis.FRESH_IDS], axis=0
        )
        for shell in analysis.SHELLS
    }


def maximum_archive_difference(
    archive: np.lib.npyio.NpzFile,
    expected: dict[str, np.ndarray],
    prefix: str,
) -> float:
    values = []
    for key, array in expected.items():
        archive_key = f"{prefix}__{key}"
        if archive_key not in archive.files:
            raise RuntimeError(f"archive key missing: {archive_key}")
        values.append(float(np.max(np.abs(np.asarray(archive[archive_key]) - array))))
    return max(values, default=0.0)


def occupancy_expectation(items: int, draws: int) -> dict[str, float]:
    empty_one = (1.0 - 1.0 / items) ** draws
    occupied_probability = 1.0 - empty_one
    both_occupied = 1.0 - 2.0 * empty_one + (1.0 - 2.0 / items) ** draws
    variance = items * occupied_probability * (1.0 - occupied_probability)
    variance += items * (items - 1.0) * (
        both_occupied - occupied_probability * occupied_probability
    )
    return {
        "mean": float(items * occupied_probability),
        "standard_deviation": float(math.sqrt(max(0.0, variance))),
    }


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q975": float(np.quantile(array, 0.975)),
        "maximum": float(np.max(array)),
    }


def tie_fraction(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(1.0 - len(np.unique(flat)) / len(flat))


def structural_index_audit(
    *, items: int, resamples: int, seed: int
) -> dict[str, dict[str, float | int] | int]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    unique_items = np.empty(resamples, dtype=np.float64)
    effective_support = np.empty(resamples, dtype=np.float64)
    maximum_multiplicity = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        counts = np.bincount(rng.integers(0, items, size=items), minlength=items)
        unique_items[index] = np.count_nonzero(counts)
        effective_support[index] = float(items**2 / np.sum(np.square(counts)))
        maximum_multiplicity[index] = np.max(counts)
    return {
        "seed": seed,
        "resamples": resamples,
        "unique_items": summarize(unique_items),
        "multiplicity_effective_support": summarize(effective_support),
        "maximum_item_multiplicity": summarize(maximum_multiplicity),
    }


def reproduce_item_bootstrap(
    analysis: Any,
    fresh: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    global_values = np.empty(resamples, dtype=np.float64)
    median_values = np.empty(resamples, dtype=np.float64)
    unique_items = np.empty(resamples, dtype=np.float64)
    effective_support = np.empty(resamples, dtype=np.float64)
    maximum_multiplicity = np.empty(resamples, dtype=np.float64)
    global_ties = np.empty((resamples, len(analysis.SHELLS)), dtype=np.float64)
    mean_row_ties = np.empty_like(global_ties)
    rank_degenerate = np.zeros(resamples, dtype=bool)
    differences: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for shell in analysis.SHELLS:
        d0 = fresh[shell][:, None, :, 0] - reference[shell][None, :, :, 0]
        d1 = fresh[shell][:, None, :, 1] - reference[shell][None, :, :, 1]
        differences[shell] = (d0, d1, d0 * d1)

    batch_size = 16
    written = 0
    while written < resamples:
        count = min(batch_size, resamples - written)
        item_indices = rng.integers(
            0, analysis.EXPECTED_ITEM_COUNT, size=(count, analysis.EXPECTED_ITEM_COUNT)
        )
        shape_batch: dict[str, np.ndarray] = {}
        for shell in analysis.SHELLS:
            d0, d1, product = differences[shell]
            sampled_d0 = np.take(d0, item_indices, axis=2).transpose(2, 0, 1, 3)
            sampled_d1 = np.take(d1, item_indices, axis=2).transpose(2, 0, 1, 3)
            sampled_product = np.take(product, item_indices, axis=2).transpose(2, 0, 1, 3)
            panel = sampled_product.mean(axis=-1)
            mean_product = sampled_d0.mean(axis=-1) * sampled_d1.mean(axis=-1)
            shape_batch[shell] = (panel - mean_product) * (
                analysis.EXPECTED_ITEM_COUNT / (analysis.EXPECTED_ITEM_COUNT - 1.0)
            )
        for offset in range(count):
            index = written + offset
            counts = np.bincount(
                item_indices[offset], minlength=analysis.EXPECTED_ITEM_COUNT
            )
            positive = counts[counts > 0]
            unique_items[index] = len(positive)
            effective_support[index] = float(
                analysis.EXPECTED_ITEM_COUNT**2 / np.sum(np.square(positive))
            )
            maximum_multiplicity[index] = float(np.max(positive))
            shape = {shell: shape_batch[shell][offset] for shell in analysis.SHELLS}
            rows = analysis.row_associations(geometry, shape)
            shell_values = [
                analysis.spearman_flat(geometry[shell], shape[shell])
                for shell in analysis.SHELLS
            ]
            global_values[index] = float(np.mean(shell_values))
            median_values[index] = float(np.median(rows))
            rank_degenerate[index] = not (
                np.all(np.isfinite(rows)) and np.all(np.isfinite(shell_values))
            )
            for shell_index, shell in enumerate(analysis.SHELLS):
                global_ties[index, shell_index] = tie_fraction(shape[shell])
                mean_row_ties[index, shell_index] = float(
                    np.mean([tie_fraction(row) for row in shape[shell]])
                )
        written += count

    payload = {
        "resamples": int(resamples),
        "seed": str(seed),
        "global_equal_shell_mean": {
            "q025": float(np.quantile(global_values, 0.025)),
            "q50": float(np.quantile(global_values, 0.50)),
            "q975": float(np.quantile(global_values, 0.975)),
        },
        "median_row_association": {
            "q025": float(np.quantile(median_values, 0.025)),
            "q50": float(np.quantile(median_values, 0.50)),
            "q975": float(np.quantile(median_values, 0.975)),
        },
        "unique_items": summarize(unique_items),
        "expected_unique_items": occupancy_expectation(
            analysis.EXPECTED_ITEM_COUNT, analysis.EXPECTED_ITEM_COUNT
        ),
        "multiplicity_effective_support": summarize(effective_support),
        "maximum_item_multiplicity": summarize(maximum_multiplicity),
        "global_Dshape_tie_fraction": {
            shell: summarize(global_ties[:, shell_index])
            for shell_index, shell in enumerate(analysis.SHELLS)
        },
        "mean_row_Dshape_tie_fraction": {
            shell: summarize(mean_row_ties[:, shell_index])
            for shell_index, shell in enumerate(analysis.SHELLS)
        },
        "rank_degeneracy_count": int(np.sum(rank_degenerate)),
        "rank_degeneracy_fraction": float(np.mean(rank_degenerate)),
    }
    arrays = {
        "global_equal_shell_mean": global_values,
        "median_row_association": median_values,
        "unique_items": unique_items,
        "effective_support": effective_support,
    }
    return payload, arrays


def maximum_nested_quantile_difference(
    observed: dict[str, Any], expected: dict[str, Any]
) -> float:
    differences = []
    for statistic in ("global_equal_shell_mean", "median_row_association"):
        for quantile in ("q025", "q50", "q975"):
            differences.append(
                abs(float(observed[statistic][quantile]) - float(expected[statistic][quantile]))
            )
    return max(differences)


def write_histograms(path: Path, arrays: dict[str, np.ndarray], bins: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["statistic", "bin_left", "bin_right", "count", "density"],
        )
        writer.writeheader()
        for name, values in arrays.items():
            if name not in {"global_equal_shell_mean", "median_row_association"}:
                continue
            counts, edges = np.histogram(values, bins=bins, density=False)
            widths = np.diff(edges)
            density = counts / (np.sum(counts) * widths)
            for index, count in enumerate(counts):
                writer.writerow(
                    {
                        "statistic": name,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                        "density": float(density[index]),
                    }
                )


def audit_real(args: argparse.Namespace) -> dict[str, Any]:
    precheck = validate_precheck()
    if sha256_file(RELEASE_ANALYSIS) != EXPECTED_RELEASE_ANALYSIS_SHA256:
        raise RuntimeError("release analysis hash mismatch")
    expected_hashes = {
        args.fresh_scores: EXPECTED_FRESH_SCORES_SHA256,
        args.error_arrays: EXPECTED_ERROR_ARRAYS_SHA256,
        args.dshape: EXPECTED_DSHAPE_SHA256,
        args.dtotal: EXPECTED_DTOTAL_SHA256,
        args.historical_scores: EXPECTED_HISTORICAL_SCORES_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"private input hash mismatch: {path.name}")

    analysis = load_analysis_module()
    item_ids, _items = analysis.load_panel()
    fresh_errors = rebuild_fresh_errors(analysis, args.fresh_scores, item_ids)
    stored_errors = np.load(args.error_arrays, allow_pickle=False)
    error_max = max(
        float(
            np.max(
                np.abs(
                    np.asarray(stored_errors[f"error__{condition}"])
                    - fresh_errors[condition]
                )
            )
        )
        for condition in fresh_errors
    )
    reference_errors, reference_metadata = analysis.load_reference_errors(
        args.historical_scores, item_ids
    )
    shape_ref, total_ref, shape_ff, total_ff = analysis.compute_distances(
        fresh_errors, reference_errors
    )
    stored_shape = np.load(args.dshape, allow_pickle=False)
    stored_total = np.load(args.dtotal, allow_pickle=False)
    dshape_max = max(
        maximum_archive_difference(stored_shape, shape_ref, "fresh_reference"),
        maximum_archive_difference(stored_shape, shape_ff, "fresh_fresh"),
    )
    dtotal_max = max(
        maximum_archive_difference(stored_total, total_ref, "fresh_reference"),
        maximum_archive_difference(stored_total, total_ff, "fresh_fresh"),
    )
    matrices, _matrix_metadata = analysis.load_prediction_matrices()
    geometry = {
        shell: matrices[f"A0_{shell}_FRESH_REFERENCE"] for shell in analysis.SHELLS
    }
    primary_r = analysis.row_associations(geometry, shape_ref)
    release = read_json(RELEASE_ANALYSIS)
    release_r = np.asarray(release["primary"]["r_i"], dtype=np.float64)
    primary_r_max = float(np.max(np.abs(primary_r - release_r)))
    fresh = stack_fresh(analysis, fresh_errors)
    reproduction, arrays = reproduce_item_bootstrap(
        analysis,
        fresh,
        reference_errors,
        geometry,
        resamples=int(precheck["current_reproduction"]["resamples"]),
        seed=int(precheck["current_reproduction"]["seed"]),
    )
    archived_bootstrap = release["secondary"]["item_bootstrap"]
    reproduction_max = maximum_nested_quantile_difference(reproduction, archived_bootstrap)
    full_global = float(
        release["secondary"]["global_fresh_reference"]["A0"]["global"][
            "equal_shell_mean"
        ]
    )
    full_median = float(release["primary"]["median"])
    full_ties = {
        "global_Dshape_tie_fraction": {
            shell: tie_fraction(shape_ref[shell]) for shell in analysis.SHELLS
        },
        "mean_row_Dshape_tie_fraction": {
            shell: float(np.mean([tie_fraction(row) for row in shape_ref[shell]]))
            for shell in analysis.SHELLS
        },
    }
    tolerance = float(
        precheck["evaluation_criteria"]["exact_real_data_reproduction_max_abs_difference"]
    )
    primary_integrity = bool(
        error_max == 0.0
        and dshape_max == 0.0
        and dtotal_max == 0.0
        and primary_r_max <= tolerance
    )
    implementation_exact = bool(reproduction_max <= tolerance)
    result = {
        "schema_version": "q2-oos-v2-item-bootstrap-implementation-audit-v1",
        "label": EXPECTED_LABEL,
        "status": "IMPLEMENTATION_AUDIT_COMPLETE",
        "frozen_primary_classification": "Q2_OOS_V2_A0_PASS",
        "primary_integrity": {
            "pass": primary_integrity,
            "fresh_scored_rows": len(load_scores(args.fresh_scores)),
            "historical_reference_rows": int(reference_metadata["rows"]),
            "error_array_max_abs_difference": error_max,
            "Dshape_max_abs_difference": dshape_max,
            "Dtotal_max_abs_difference": dtotal_max,
            "primary_r_i_max_abs_difference": primary_r_max,
            "fresh_historical_item_order_exact": True,
        },
        "implementation_reconstruction": {
            "pass": implementation_exact,
            "maximum_archived_quantile_difference": reproduction_max,
            "resampling_unit": "item",
            "draws_per_resample": analysis.EXPECTED_ITEM_COUNT,
            "replacement": True,
            "all_32_fresh_conditions_coupled": True,
            "both_rollouts_coupled": True,
            "all_31_historical_references_coupled": True,
            "shells_use_same_item_indices": True,
            "Dshape_recomputed_within_resample": True,
            "row_and_global_spearman_recomputed_within_resample": True,
            "negative_distances_unclipped": True,
            "complete_case_filtering": False,
            "normalization_uses_draw_count_not_unique_count": True,
            "quantile_implementation": "numpy.quantile linear default",
        },
        "full_sample": {
            "global_A0_fresh_reference_rho": full_global,
            "median_fresh_controller_r_i": full_median,
            **full_ties,
        },
        "reproduced_bootstrap": reproduction,
        "independent_structural_index_audit": structural_index_audit(
            items=analysis.EXPECTED_ITEM_COUNT,
            resamples=int(
                precheck["simulation_protocol"]["real_data_structural_index_resamples"]
            ),
            seed=int(
                precheck["simulation_protocol"]["seeds"][
                    "real_data_structural_indices"
                ]
            ),
        ),
        "centering_displacement": {
            "global_q50_minus_full": float(
                reproduction["global_equal_shell_mean"]["q50"] - full_global
            ),
            "median_row_q50_minus_full": float(
                reproduction["median_row_association"]["q50"] - full_median
            ),
        },
        "private_input_hashes_verified": {
            path.name: expected for path, expected in expected_hashes.items()
        },
        "source_hashes": {
            "diagnostic_precheck": sha256_file(PRECHECK),
            "diagnostic_implementation": sha256_file(Path(__file__)),
            "frozen_analysis_source": sha256_file(ANALYSIS_SOURCE),
            "release_analysis": sha256_file(RELEASE_ANALYSIS),
        },
        "raw_text_manually_inspected": False,
        "new_semantic_trajectories": 0,
        "Qwen_loaded": False,
    }
    write_json(args.output_dir / "ITEM_BOOTSTRAP_IMPLEMENTATION_AUDIT.json", result)
    write_histograms(
        args.output_dir / "ITEM_BOOTSTRAP_REPRODUCTION_HISTOGRAM.csv", arrays
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-scores", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--error-arrays", type=Path, required=True)
    parser.add_argument("--dshape", type=Path, required=True)
    parser.add_argument("--dtotal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_real(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_integrity": result["primary_integrity"]["pass"],
                "implementation_exact": result["implementation_reconstruction"]["pass"],
                "maximum_archived_quantile_difference": result[
                    "implementation_reconstruction"
                ]["maximum_archived_quantile_difference"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
