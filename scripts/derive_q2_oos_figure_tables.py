#!/usr/bin/env python3
"""Derive release-safe Q2 OOS figure tables from sealed aggregate artifacts.

The private Dshape archive is read only to derive controller-level and pairwise
aggregate distances. Raw generations, prompts, benchmark text, and item-level
correctness are neither read nor emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_oos_fresh_controller import spearman_flat  # noqa: E402

ANALYSIS = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
    "Q2_OOS_V2_SEMANTIC_ANALYSIS.json"
)
DIAGNOSTIC = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
    "item_bootstrap_diagnostic/Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_RESULT.json"
)
MATRICES = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout/PREDICTION_MATRICES.npz"
)
MATRIX_METADATA = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout/"
    "PREDICTION_MATRIX_METADATA.json"
)
RUNTIME_LOCK = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout/RUNTIME_MONITOR_LOCK.json"
)
OUTPUT = ROOT / "manuscript/data/paper1_q2_oos/derived_figure_tables"
EXPECTED_DSHAPE_SHA256 = "a6a6b4889e2c86df04ce42c4415281dde82af0d2deb1347b8083015e95089ea5"
EXPECTED_MATRIX_SHA256 = "b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703"
SHELLS = ("MEDIUM", "STRONG")
METRICS = ("A0", "A1", "A2", "D2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=tolerance):
        raise RuntimeError(f"scientific reconciliation failed: {left} != {right}")


def derive(dshape_path: Path, output_dir: Path) -> dict[str, Path]:
    if sha256(dshape_path) != EXPECTED_DSHAPE_SHA256:
        raise RuntimeError("private Dshape hash mismatch")
    if sha256(MATRICES) != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("prediction-matrix hash mismatch")

    analysis = read_json(ANALYSIS)
    diagnostic = read_json(DIAGNOSTIC)
    metadata = read_json(MATRIX_METADATA)
    runtime_lock = read_json(RUNTIME_LOCK)
    prediction = np.load(MATRICES, allow_pickle=False)
    dshape = np.load(dshape_path, allow_pickle=False)
    fresh_ids = list(metadata["fresh_controller_order"])
    reference_ids = list(metadata["reference_controller_order"])
    if len(fresh_ids) != 16 or len(reference_ids) != 31:
        raise RuntimeError("unexpected controller dimensions")

    table_paths: dict[str, Path] = {}

    controller_rows: list[dict[str, Any]] = []
    archived_primary = analysis["primary"]["r_i"]
    for index, controller_id in enumerate(fresh_ids):
        shell_r = {
            shell: float(
                spearman_flat(
                    prediction[f"A0_{shell}_FRESH_REFERENCE"][index],
                    dshape[f"fresh_reference__{shell}"][index],
                )
            )
            for shell in SHELLS
        }
        average = float(np.mean(list(shell_r.values())))
        _close(average, float(archived_primary[index]))
        controller_rows.append(
            {
                "controller_order": index + 1,
                "controller_id": controller_id,
                "medium_rho": shell_r["MEDIUM"],
                "strong_rho": shell_r["STRONG"],
                "equal_shell_r_i": average,
                "primary_positive": str(average > 0.0).lower(),
            }
        )
    controller_path = output_dir / "controller_associations.csv"
    write_csv(
        controller_path,
        [
            "controller_order",
            "controller_id",
            "medium_rho",
            "strong_rho",
            "equal_shell_r_i",
            "primary_positive",
        ],
        controller_rows,
    )
    table_paths["controller_associations"] = controller_path

    global_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        archived = analysis["secondary"]["global_fresh_reference"][metric]
        shell_values: dict[str, float] = {}
        for shell in SHELLS:
            value = float(
                spearman_flat(
                    prediction[f"{metric}_{shell}_FRESH_REFERENCE"],
                    dshape[f"fresh_reference__{shell}"],
                )
            )
            _close(value, float(archived["global"]["shell"][shell]))
            shell_values[shell] = value
        equal_shell = float(np.mean(list(shell_values.values())))
        _close(equal_shell, float(archived["global"]["equal_shell_mean"]))
        global_rows.append(
            {
                "metric": metric,
                "role": "PRIMARY_GEOMETRY" if metric == "A0" else "SECONDARY_ONLY",
                "medium_rho": shell_values["MEDIUM"],
                "strong_rho": shell_values["STRONG"],
                "equal_shell_rho": equal_shell,
            }
        )
    global_path = output_dir / "global_associations.csv"
    write_csv(
        global_path,
        ["metric", "role", "medium_rho", "strong_rho", "equal_shell_rho"],
        global_rows,
    )
    table_paths["global_associations"] = global_path

    pair_rows: list[dict[str, Any]] = []
    upper = np.triu_indices(len(fresh_ids), 1)
    for i, j in zip(*upper, strict=True):
        row: dict[str, Any] = {
            "controller_i_order": int(i + 1),
            "controller_j_order": int(j + 1),
            "controller_i": fresh_ids[i],
            "controller_j": fresh_ids[j],
        }
        for shell in SHELLS:
            row[f"A0_{shell.lower()}"] = float(prediction[f"A0_{shell}_FRESH_FRESH"][i, j])
            row[f"Dshape_{shell.lower()}"] = float(dshape[f"fresh_fresh__{shell}"][i, j])
        pair_rows.append(row)
    pair_path = output_dir / "fresh_fresh_pairs.csv"
    write_csv(
        pair_path,
        [
            "controller_i_order",
            "controller_j_order",
            "controller_i",
            "controller_j",
            "A0_medium",
            "Dshape_medium",
            "A0_strong",
            "Dshape_strong",
        ],
        pair_rows,
    )
    table_paths["fresh_fresh_pairs"] = pair_path

    lofo_path = output_dir / "lofo.csv"
    write_csv(
        lofo_path,
        [
            "omitted_controller",
            "mean",
            "median",
            "positive_count",
            "p_value",
            "positive_sign_pass",
        ],
        analysis["secondary"]["lofo"],
    )
    table_paths["lofo"] = lofo_path

    fresh_fresh = analysis["secondary"]["fresh_fresh_node_jackknife"]
    summary_path = output_dir / "fresh_fresh_summary.csv"
    write_csv(
        summary_path,
        ["role", "association", "jackknife_standard_error", "t", "p_value"],
        [
            {
                "role": fresh_fresh["role"],
                "association": fresh_fresh["full_association"],
                "jackknife_standard_error": fresh_fresh["jackknife_standard_error"],
                "t": fresh_fresh["t"],
                "p_value": fresh_fresh["p_value"],
            }
        ],
    )
    table_paths["fresh_fresh_summary"] = summary_path

    cluster = analysis["secondary"]["controller_cluster_bootstrap"]
    item = analysis["secondary"]["item_bootstrap"]
    diagnostic_row = diagnostic["historical_bootstrap_implementation"]
    bootstrap_path = output_dir / "bootstrap_diagnostic.csv"
    write_csv(
        bootstrap_path,
        ["object", "estimate", "q025", "q50", "q975", "interpretation"],
        [
            {
                "object": "global_A0_item_resampling",
                "estimate": analysis["secondary"]["global_fresh_reference"]["A0"]["global"][
                    "equal_shell_mean"
                ],
                "q025": item["global_equal_shell_mean"]["q025"],
                "q50": item["global_equal_shell_mean"]["q50"],
                "q975": item["global_equal_shell_mean"]["q975"],
                "interpretation": "PANEL_PERTURBATION_SENSITIVITY_NOT_CONVENTIONAL_CI",
            },
            {
                "object": "median_r_i_item_resampling",
                "estimate": analysis["primary"]["median"],
                "q025": item["median_row_association"]["q025"],
                "q50": item["median_row_association"]["q50"],
                "q975": item["median_row_association"]["q975"],
                "interpretation": "PANEL_PERTURBATION_SENSITIVITY_NOT_CONVENTIONAL_CI",
            },
            {
                "object": "global_A0_controller_cluster",
                "estimate": cluster["estimate"],
                "q025": cluster["percentile_95"][0],
                "q50": "",
                "q975": cluster["percentile_95"][1],
                "interpretation": "FRESH_CONTROLLER_POPULATION_CONDITIONAL_ON_ITEM_PANEL",
            },
            {
                "object": "ordinary_bootstrap_effective_support",
                "estimate": 300,
                "q025": "",
                "q50": diagnostic_row["observed_mean_multiplicity_effective_support"],
                "q975": "",
                "interpretation": "MEAN_KISH_EFFECTIVE_ITEM_SUPPORT",
            },
        ],
    )
    table_paths["bootstrap_diagnostic"] = bootstrap_path

    forecast = runtime_lock["normal_seconds"]
    runtime_path = output_dir / "runtime_summary.csv"
    write_csv(
        runtime_path,
        ["measure", "hours", "role"],
        [
            {
                "measure": "observed",
                "hours": analysis["efficient_termination"]["wall_seconds"] / 3600.0,
                "role": "OBSERVED",
            },
            {
                "measure": "forecast_P50",
                "hours": forecast["P50"] / 3600.0,
                "role": "FROZEN_FORECAST",
            },
            {
                "measure": "forecast_P80",
                "hours": forecast["P80"] / 3600.0,
                "role": "FROZEN_FORECAST",
            },
            {
                "measure": "forecast_P95",
                "hours": forecast["P95"] / 3600.0,
                "role": "FROZEN_FORECAST",
            },
        ],
    )
    table_paths["runtime_summary"] = runtime_path

    return table_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-dshape", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = derive(args.private_dshape.resolve(), args.output_dir.resolve())
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
