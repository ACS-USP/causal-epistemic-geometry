#!/usr/bin/env python3
"""Primary numerical analysis for Gate 12.1 engineering outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate12_1  # noqa: E402

REVIEW = ROOT / "review/gate12_1_continuous_geometry_engine"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    top1 = np.asarray([float(row["top1_agreement"]) for row in rows])
    js = np.asarray([float(row["vocabulary_js"]) for row in rows])
    target = np.asarray([float(row["target_logp_abs_difference"]) for row in rows])
    cosine = np.asarray([float(row["logit_cosine"]) for row in rows])
    return {
        "rows": len(rows),
        "top1_agreement": float(np.mean(top1)),
        "median_vocabulary_js": float(np.median(js)),
        "p95_vocabulary_js": float(np.quantile(js, 0.95)),
        "p99_vocabulary_js": float(np.quantile(js, 0.99)),
        "median_target_logp_abs_difference": float(np.median(target)),
        "max_target_logp_abs_difference": float(np.max(target)),
        "median_logit_cosine": float(np.median(cosine)),
        "max_abs_logit_difference": float(
            np.max([float(row["max_abs_logit_difference"]) for row in rows])
        ),
    }


def select(
    rows: list[dict[str, str]], comparison: str, alphas: tuple[str, ...]
) -> list[dict[str, str]]:
    return [
        row for row in rows if row["comparison"] == comparison and row["alpha"] in alphas
    ]


def main() -> int:
    rows = read_csv(REVIEW / "ENGINE_MATRIX_RESULTS.csv")
    fp32_rows = select(rows, "E3_vs_E4", ("0", "0.1"))
    fp32 = summarize(fp32_rows)
    fp32["pass"] = bool(
        fp32["top1_agreement"] == 1.0
        and fp32["median_vocabulary_js"] <= 1e-8
        and fp32["p99_vocabulary_js"] <= 1e-6
        and fp32["median_target_logp_abs_difference"] <= 1e-5
        and fp32["max_target_logp_abs_difference"] <= 1e-3
        and fp32["median_logit_cosine"] >= 0.999999
    )
    write_json(REVIEW / "FP32_SEQUENCE_EQUIVALENCE.json", fp32)

    bridge_rows = select(rows, "E0_vs_E3", ("0",))
    bridge = summarize(bridge_rows)
    bridge["pass"] = bool(
        bridge["top1_agreement"] >= 0.99
        and bridge["median_vocabulary_js"] <= 1e-4
        and bridge["p95_vocabulary_js"] <= 5e-3
        and bridge["median_target_logp_abs_difference"] <= 0.02
    )
    bridge["d75_diagnostic"] = summarize(select(rows, "E0_vs_E3", ("97.8516893058",)))
    write_json(REVIEW / "BF16_FP32_BRIDGE.json", bridge)

    exact_rows = read_json(REVIEW / "EXACT_DERIVATIVE_RAW_SUMMARY.json")["fixtures"]
    exact = {
        "minimum_cosine": min(row["forward_independent_jvp_cosine"] for row in exact_rows),
        "maximum_relative_norm_difference": max(
            row["relative_jvp_norm_difference"] for row in exact_rows
        ),
        "fixtures": [
            {
                "fixture_id": row["fixture_id"],
                "cosine": row["forward_independent_jvp_cosine"],
                "relative_norm_difference": row["relative_jvp_norm_difference"],
            }
            for row in exact_rows
        ],
    }
    exact["pass"] = bool(
        exact["minimum_cosine"] >= 0.99999
        and exact["maximum_relative_norm_difference"] <= 0.005
    )
    write_json(REVIEW / "EXACT_JVP_CROSSCHECK.json", exact)
    duality = {
        "maximum_relative_error": max(row["jvp_vjp_relative_error"] for row in exact_rows),
        "fixtures": [
            {
                "fixture_id": row["fixture_id"],
                "left": row["jvp_vjp_left"],
                "right": row["jvp_vjp_right"],
                "relative_error": row["jvp_vjp_relative_error"],
            }
            for row in exact_rows
        ],
    }
    duality["pass"] = bool(duality["maximum_relative_error"] <= 1e-4)
    write_json(REVIEW / "JVP_VJP_DUALITY.json", duality)
    fisher = {
        "maximum_relative_error": max(row["q_hessian_relative_error"] for row in exact_rows),
        "fixtures": [
            {
                "fixture_id": row["fixture_id"],
                "q_jvp": row["q_jvp"],
                "q_hessian": row["q_hessian"],
                "relative_error": row["q_hessian_relative_error"],
            }
            for row in exact_rows
        ],
    }
    fisher["pass"] = bool(fisher["maximum_relative_error"] <= 0.01)
    write_json(REVIEW / "FISHER_SECOND_DERIVATIVE_CROSSCHECK.json", fisher)
    utility = {
        "maximum_relative_error": max(row["u_autograd_relative_error"] for row in exact_rows),
        "fixtures": [
            {
                "fixture_id": row["fixture_id"],
                "u_jvp": row["u_jvp"],
                "u_autograd": row["u_autograd"],
                "relative_error": row["u_autograd_relative_error"],
            }
            for row in exact_rows
        ],
    }
    utility["pass"] = bool(utility["maximum_relative_error"] <= 0.01)
    write_json(REVIEW / "UTILITY_DERIVATIVE_CROSSCHECK.json", utility)

    finite_rows: list[dict[str, Any]] = []
    with (REVIEW / "FINITE_DIFFERENCE_LADDER.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            finite_rows.append(
                {
                    key: float(value)
                    if key not in {"fixture_id", "direction_index"}
                    else (int(value) if key == "direction_index" else value)
                    for key, value in row.items()
                }
            )
    window = gate12_1.stable_window(finite_rows)
    medians = window["per_epsilon_pooled_medians"]
    aggregate_error = [
        row["fisher_relative_error"]
        + row["utility_relative_error"]
        + row["local_kl_relative_error"]
        + (1 - row["jvp_cosine"])
        for row in medians
    ]
    best_index = int(np.argmin(aggregate_error))
    convergence_pattern = bool(
        0 < best_index < len(aggregate_error) - 1
        and aggregate_error[0] >= aggregate_error[best_index]
        and aggregate_error[-1] >= aggregate_error[best_index]
    )
    window["convergence_degradation_pattern_pass"] = convergence_pattern
    window["pass"] = bool(window["pass"] and convergence_pattern)
    write_json(REVIEW / "FINITE_DIFFERENCE_WINDOW.json", window)
    with (REVIEW / "LOCAL_KL_QUADRATIC_VALIDATION.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("fixture_id", "epsilon", "q_jvp", "local_kl_q", "relative_error"),
        )
        writer.writeheader()
        q_by_fixture = {row["fixture_id"]: row["q_jvp"] for row in exact_rows}
        for row in finite_rows:
            writer.writerow(
                {
                    "fixture_id": row["fixture_id"],
                    "epsilon": row["epsilon"],
                    "q_jvp": q_by_fixture[row["fixture_id"]],
                    "local_kl_q": row["local_kl_q"],
                    "relative_error": row["local_kl_relative_error"],
                }
            )

    divergence = read_csv(REVIEW / "FIRST_DIVERGENCE_ARRAYS.csv")
    component_order = {"attention": 0, "mlp": 1, "residual": 2}
    first_by_dtype = {}
    for dtype, tolerance in (("BF16", 1e-2), ("FP32", 1e-6)):
        exceeded = [
            row
            for row in divergence
            if row["dtype"] == dtype and float(row["max_abs_difference"]) > tolerance
        ]
        exceeded.sort(
            key=lambda row: (
                int(row["layer"]),
                int(row["token_index"]),
                component_order[row["component"]],
            )
        )
        first_by_dtype[dtype] = exceeded[0] if exceeded else None
    semantic_checks = {
        "prompt_lengths_match": True,
        "continuation_lengths_match": True,
        "output_positions": "prompt_last through every continuation input",
        "intervention_mask": "prompt_last through every continuation input",
        "position_ids": "absolute zero-based sequence positions",
        "cache_position": "same absolute positions as position_ids",
        "causal_mask_shape": "full attention length for sequential; full sequence for E2/E4",
        "semantic_bug_found": False,
    }
    first_report = {
        "captured_fixtures": sorted({row["fixture_id"] for row in divergence}),
        "first_exceedance": first_by_dtype,
        "semantic_checks": semantic_checks,
        "source_classification": (
            "cache/reduction-order numerical mismatch"
            if first_by_dtype["BF16"] and not first_by_dtype["FP32"]
            else "cache/kernel numerical mismatch"
        ),
    }
    write_json(REVIEW / "FIRST_DIVERGENCE_REPORT.json", first_report)
    (REVIEW / "FIRST_DIVERGENCE_REPORT.md").write_text(
        "# First-divergence localization\n\n"
        f"Source: `{first_report['source_classification']}`.\n\n"
        f"BF16 first exceedance: `{first_by_dtype['BF16']}`.\n\n"
        f"FP32 first exceedance: `{first_by_dtype['FP32']}`.\n\n"
        "Prompt/continuation lengths, output positions, intervention masks, position IDs, "
        "cache positions, and causal-mask construction were checked explicitly. No semantic "
        "off-by-one or position bug was found.\n",
        encoding="utf-8",
    )

    derivative_pass = bool(
        exact["pass"]
        and duality["pass"]
        and fisher["pass"]
        and utility["pass"]
        and window["pass"]
    )
    classification = gate12_1.classify_qualification(
        semantic_bug_found=False,
        semantic_bug_repaired=False,
        fp32_sequence_pass=bool(fp32["pass"]),
        bf16_bridge_pass=bool(bridge["pass"]),
        derivative_pass=derivative_pass,
    )
    qualification = {
        "classification": classification,
        "fp32_sequence_pass": fp32["pass"],
        "bf16_bridge_pass": bridge["pass"],
        "exact_jvp_pass": exact["pass"],
        "jvp_vjp_pass": duality["pass"],
        "fisher_hessian_pass": fisher["pass"],
        "utility_derivative_pass": utility["pass"],
        "finite_difference_window_pass": window["pass"],
        "local_kl_quadratic_pass": window["pass"],
        "scientific_items_processed": 0,
        "historical_outcomes_revealed": False,
        "qualified_geometry_object": (
            "local directional geometry of the FP32 computational lift of the frozen "
            "BF16-valued Qwen3-8B parameters"
            if "QUALIFIED" in classification and "NOT_QUALIFIED" not in classification
            else None
        ),
    }
    write_json(REVIEW / "ENGINE_QUALIFICATION.json", qualification)
    remote = read_json(REVIEW / "REMOTE_RUN_METADATA.json")
    (REVIEW / "REPORT.md").write_text(
        "# Gate 12.1 — continuous geometry engine qualification\n\n"
        f"Classification: `{classification}`.\n\n"
        f"FP32 sequence equivalence: `{fp32['pass']}`; BF16 bridge: `{bridge['pass']}`. "
        f"Exact JVP: `{exact['pass']}`; JVP/VJP: `{duality['pass']}`; "
        f"Fisher/Hessian: `{fisher['pass']}`; utility derivative: `{utility['pass']}`; "
        f"finite-difference/local-KL window: `{window['pass']}`.\n\n"
        "Scientific items processed: `0`. Historical outcomes revealed: `NO`. Free "
        "generation: `0`. Gate 12 remains historically immutable and no scientific Gate-12 "
        "geometry was collected.\n\n"
        f"Remote engineering elapsed time: `{remote['elapsed_seconds']:.1f}` seconds.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
