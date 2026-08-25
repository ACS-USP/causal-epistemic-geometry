#!/usr/bin/env python3
"""Analyze Q2-V2 non-scientific engine benchmarks and freeze the cost choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
EXPECTED_ROWS = 6960
SAFETY_MARGIN = 1.25


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def keyed_rows(benchmark: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["fixture_id"]), str(row["condition"])): row for row in benchmark["rows"]
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = keyed_rows(reference)
    right = keyed_rows(candidate)
    fields = (
        "seed",
        "generated_token_count",
        "generated_token_ids",
        "generated_token_ids_sha256",
        "raw_output_sha256",
        "commitment_valid",
        "parser_failure_reason",
        "canonical_value",
    )
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            mismatches.append({"key": key, "field": "logical_key_presence"})
            continue
        for field in fields:
            if left[key].get(field) != right[key].get(field):
                mismatches.append({"key": key, "field": field})
    seeds = [int(row["seed"]) for row in candidate["rows"]]
    return {
        "reference_rows": len(left),
        "candidate_rows": len(right),
        "logical_keys_identical": set(left) == set(right),
        "logical_seeds_unique": len(seeds) == len(set(seeds)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "exact_discrete_equivalence": len(mismatches) == 0 and set(left) == set(right),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in (
            ("p50", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
        )
    } | {"max": float(max(values))}


def tail_analysis(reference: dict[str, Any]) -> dict[str, Any]:
    rows = list(reference["rows"])
    total_seconds = sum(float(row["elapsed_seconds"]) for row in rows)
    total_tokens = sum(int(row["generated_token_count"]) for row in rows)
    thresholds: dict[str, Any] = {}
    for threshold in (512, 1024, 2048, 3072, 4096):
        selected = [row for row in rows if int(row["generated_token_count"]) >= threshold]
        thresholds[str(threshold)] = {
            "rows": len(selected),
            "row_fraction": len(selected) / len(rows),
            "runtime_fraction": sum(float(row["elapsed_seconds"]) for row in selected)
            / total_seconds,
            "token_fraction": sum(int(row["generated_token_count"]) for row in selected)
            / total_tokens,
        }
    ordered_runtime = sorted((float(row["elapsed_seconds"]) for row in rows), reverse=True)
    top_shares = {}
    for fraction in (0.01, 0.05, 0.10, 0.25):
        count = max(1, int(np.ceil(len(rows) * fraction)))
        top_shares[f"top_{int(fraction * 100)}pct"] = {
            "rows": count,
            "runtime_fraction": sum(ordered_runtime[:count]) / total_seconds,
        }
    return {
        "fixture_rows": len(rows),
        "runtime_seconds": quantiles([float(row["elapsed_seconds"]) for row in rows]),
        "generated_tokens": quantiles(
            [float(row["generated_token_count"]) for row in rows]
        ),
        "long_generation_thresholds": thresholds,
        "runtime_concentration": top_shares,
        "max_new_tokens_unchanged": 4096,
        "rows_excluded": 0,
    }


def projection(benchmark: dict[str, Any], hourly_rate: float) -> dict[str, float]:
    mean_seconds = float(benchmark["mean_seconds_per_row"])
    hours = mean_seconds * EXPECTED_ROWS * SAFETY_MARGIN / 3600.0
    return {
        "hourly_rate_usd": hourly_rate,
        "mean_seconds_per_row": mean_seconds,
        "steady_state_tokens_per_second": float(benchmark["steady_state_tokens_per_second"]),
        "projected_common_hours_with_25pct_margin": hours,
        "projected_common_cost_usd": hours * hourly_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-cumulative-usd", type=float, required=True)
    parser.add_argument("--a40-startup-seconds", type=float, required=True)
    parser.add_argument("--rtx-startup-seconds", type=float, required=True)
    parser.add_argument("--rtx-engineering-pass", action="store_true")
    parser.add_argument("--wallet-balance-usd", type=float, required=True)
    parser.add_argument("--wallet-buffer-fraction", type=float, default=0.10)
    args = parser.parse_args()

    a40 = read_json(REVIEW / "V2_ENGINE_BENCHMARK_A40_SECURE_REFERENCE.json")
    rtx = read_json(REVIEW / "V2_ENGINE_BENCHMARK_RTX6000_ADA_SECURE.json")
    equivalence = compare(a40, rtx)
    equivalence.update(
        {
            "reference": "A40_SECURE_REFERENCE",
            "candidate": "RTX6000_ADA_SECURE",
            "candidate_engineering_hook_audit_pass": args.rtx_engineering_pass,
            "candidate_qualified": equivalence["exact_discrete_equivalence"]
            and args.rtx_engineering_pass,
            "batching": "SERIAL_REFERENCE_NO_BATCHING",
            "cross_row_rng_contamination": False,
        }
    )
    write_json(REVIEW / "ENGINE_EQUIVALENCE.json", equivalence)
    write_json(REVIEW / "EXECUTION_COST_TAIL.json", tail_analysis(a40))

    candidates = {
        "A40_SECURE_REFERENCE": {
            **projection(a40, 0.44),
            "startup_and_model_prep_seconds": args.a40_startup_seconds,
            "model_load_seconds": float(a40["model_load_seconds"]),
            "qualified": True,
            "qualification_reason": "frozen serial reference engine",
        },
        "RTX6000_ADA_SECURE": {
            **projection(rtx, 0.84),
            "startup_and_model_prep_seconds": args.rtx_startup_seconds,
            "model_load_seconds": float(rtx["model_load_seconds"]),
            "qualified": equivalence["candidate_qualified"],
            "qualification_reason": (
                "exact discrete equivalence and hook audit"
                if equivalence["candidate_qualified"]
                else "rejected by exact discrete equivalence or hook audit"
            ),
        },
        "H100_SXM_SECURE": {
            "hourly_rate_usd": 3.29,
            "qualified": False,
            "benchmarked": False,
            "qualification_reason": (
                "economically implausible: requires more than 7.47x A40 throughput "
                "to beat the reference dollar cost"
            ),
        },
    }
    for value in candidates.values():
        if "projected_common_cost_usd" in value:
            value["projected_cumulative_cost_usd"] = (
                args.observed_cumulative_usd + value["projected_common_cost_usd"]
            )
    qualified = [
        (name, value)
        for name, value in candidates.items()
        if value.get("qualified") and "projected_common_cost_usd" in value
    ]
    selected_name, selected = min(
        qualified,
        key=lambda pair: (
            pair[1]["projected_common_cost_usd"],
            pair[1]["projected_common_hours_with_25pct_margin"],
        ),
    )
    wallet_required_usd = max(
        0.0, selected["projected_common_cost_usd"] - args.wallet_balance_usd
    )
    operational_buffer_usd = (
        selected["projected_common_cost_usd"] * args.wallet_buffer_fraction
    )
    recommended_top_up_usd = wallet_required_usd + operational_buffer_usd
    bakeoff = {
        "selection_rule": (
            "lowest projected complete common-panel dollar cost among qualified engines; "
            "tie-break lower projected wall-clock"
        ),
        "scientific_outputs_used": False,
        "common_panel_rows_existing": 0,
        "observed_cumulative_q2_v2_cost_usd": args.observed_cumulative_usd,
        "candidates": candidates,
        "selected": selected_name,
    }
    write_json(REVIEW / "GPU_BAKEOFF.json", bakeoff)
    lock = {
        "status": "FROZEN_PRE_COMMON_PANEL",
        "selected_execution_platform": selected_name,
        "engine": "SERIAL_REFERENCE",
        "batching_policy": "NONE",
        "hourly_rate_usd": selected["hourly_rate_usd"],
        "projected_common_hours_with_25pct_margin": selected[
            "projected_common_hours_with_25pct_margin"
        ],
        "projected_common_cost_usd": selected["projected_common_cost_usd"],
        "projected_cumulative_cost_usd": selected["projected_cumulative_cost_usd"],
        "preferred_cumulative_cost_usd": 30.0,
        "hard_cumulative_cost_ceiling_usd": 45.0,
        "preferred_cost_pass": selected["projected_cumulative_cost_usd"] <= 30.0,
        "hard_cost_pass": selected["projected_cumulative_cost_usd"] <= 45.0,
        "wallet_gate": "PENDING_ACCOUNT_BALANCE_VERIFICATION",
        "wallet_balance_usd_principal_reported": args.wallet_balance_usd,
        "minimum_additional_wallet_usd": wallet_required_usd,
        "operational_buffer_usd": operational_buffer_usd,
        "recommended_top_up_usd": recommended_top_up_usd,
        "equivalence_artifact": "ENGINE_EQUIVALENCE.json",
        "bakeoff_artifact": "GPU_BAKEOFF.json",
        "tail_artifact": "EXECUTION_COST_TAIL.json",
        "scientific_lock": "V2_FINAL_PROTOCOL_LOCK.json",
        "scientific_design_changed": False,
        "common_panel_rows_at_lock": 0,
        "qualification_source_commit": "e73a3ef",
    }
    lock["wallet_gate"] = (
        "PASS"
        if args.wallet_balance_usd >= selected["projected_common_cost_usd"]
        else "FAIL_INSUFFICIENT_WALLET"
    )
    write_json(REVIEW / "EXECUTION_ENGINE_LOCK.json", lock)
    print(json.dumps({"selected": selected_name, **selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
