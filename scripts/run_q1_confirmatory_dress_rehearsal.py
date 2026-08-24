#!/usr/bin/env python3
"""Run the Q1 confirmatory pipeline on deterministic synthetic fixtures only."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_q1_confirmatory import analyze_model, bootstrap_primary  # noqa: E402
from audit_q1_confirmatory import _independent_point  # noqa: E402

from epistemic_geometry.experiments import q1_confirmatory as q1  # noqa: E402
from epistemic_geometry.reproducibility import stable_seed  # noqa: E402

REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"


def _synthetic_rows(schedule: list[dict[str, Any]], model_role: str) -> list[dict[str, Any]]:
    rows = []
    conditions = {
        "BASELINE": 0.43,
        "TEXTUAL_CAREFUL": 0.20,
        "MEANINGFUL_FIXED": 0.27,
        "RANDOM_R0": 0.41,
        "RANDOM_R1": 0.44,
        "RANDOM_R2": 0.40,
        "RANDOM_R3": 0.42,
    }
    for planned in schedule:
        probability = conditions[str(planned["condition"])]
        rng = np.random.default_rng(
            stable_seed(
                q1.EXPERIMENT_ID,
                "DRESS_REHEARSAL_OUTCOME",
                model_role,
                planned["item_id"],
                planned["condition"],
                planned["rollout_index"],
            )
        )
        wrong = bool(rng.random() < probability)
        rows.append(
            {
                **planned,
                "correct": not wrong,
                "commitment_valid": True,
                "semantic_evaluable": True,
                "generated_token_count": int(10 + rng.integers(0, 40)),
                "parser_version": "external-semantic-v3",
                "confirmatory_source_commit": "SYNTHETIC_DRESS_REHEARSAL",
            }
        )
    return rows


def main() -> int:
    item_ids = [f"synthetic_{index:03d}" for index in range(q1.N_ITEMS)]
    schedules = {
        role: q1.build_schedule(item_ids, model_role=role) for role in ("Qwen", "Ministral")
    }
    all_seeds = [int(row["seed"]) for schedule in schedules.values() for row in schedule]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("dress rehearsal found a cross-model seed collision")

    meaningful = np.arange(1, 33, dtype=np.float64)
    meaningful /= np.linalg.norm(meaningful)
    paired = np.random.default_rng(20260911).normal(size=(24, 32))
    first_bank, first_meta = q1.build_null_bank(
        meaningful, paired, model_role="DRESS_REHEARSAL"
    )
    second_bank, second_meta = q1.build_null_bank(
        meaningful, paired, model_role="DRESS_REHEARSAL"
    )
    null_stable = first_meta == second_meta and all(
        np.array_equal(first_bank[name], second_bank[name]) for name in q1.RANDOM_NAMES
    )

    model_results: dict[str, Any] = {}
    metric_differences = []
    resume_checks = []
    bootstrap_checks = []
    for model_role, schedule in schedules.items():
        rows = _synthetic_rows(schedule, model_role)
        first_completed = q1.completed_keys(
            rows[:100], source_commit="SYNTHETIC_DRESS_REHEARSAL"
        )
        complete = q1.completed_keys(rows, source_commit="SYNTHETIC_DRESS_REHEARSAL")
        resume_checks.append(len(first_completed) == 100 and len(complete - first_completed) == 698)
        result = analyze_model(
            rows,
            schedule,
            model_role,
            bootstrap_resamples=2_000,
        )
        item_order = sorted({str(row["item_id"]) for row in rows})
        arrays = q1.error_arrays(rows, item_order)
        for condition in q1.CONDITIONS[1:]:
            independent = _independent_point(arrays["BASELINE"], arrays[condition])
            for metric in ("G", "C", "D", "rescue", "damage"):
                metric_differences.append(
                    abs(independent[metric] - result["estimands"][condition][metric])
                )
        first_bootstrap = bootstrap_primary(
            arrays,
            seed=q1.BOOTSTRAP_SEEDS[model_role],
            resamples=2_000,
        )
        second_bootstrap = bootstrap_primary(
            arrays,
            seed=q1.BOOTSTRAP_SEEDS[model_role],
            resamples=2_000,
        )
        bootstrap_checks.append(first_bootstrap == second_bootstrap)
        model_results[model_role] = {
            "rows": len(rows),
            "unique_keys": len(complete),
            "classification_deterministic": result["model_pass"]
            == q1.classify_model(
                summaries=result["summaries"],
                estimands=result["estimands"],
                intervals=result["intervals"],
            )["pass"],
        }

    checks = {
        "exact_expected_row_count": all(row["rows"] == 798 for row in model_results.values()),
        "zero_duplicate_logical_keys": all(
            row["unique_keys"] == 798 for row in model_results.values()
        ),
        "independent_seed_collision_check": len(all_seeds) == len(set(all_seeds)),
        "C_G_D_independent_recomputation": max(metric_differences, default=0.0) <= 1e-12,
        "bootstrap_determinism": all(bootstrap_checks),
        "null_hashes_stable": null_stable,
        "classification_deterministic": all(
            row["classification_deterministic"] for row in model_results.values()
        ),
        "resume_semantics": all(resume_checks),
        "no_midrun_scientific_metric_access_required": True,
        "holdout_content_accessed": False,
        "model_inference": False,
    }
    pass_checks = (
        "exact_expected_row_count",
        "zero_duplicate_logical_keys",
        "independent_seed_collision_check",
        "C_G_D_independent_recomputation",
        "bootstrap_determinism",
        "null_hashes_stable",
        "classification_deterministic",
        "resume_semantics",
        "no_midrun_scientific_metric_access_required",
    )
    passed = all(bool(checks[name]) for name in pass_checks) and not bool(
        checks["holdout_content_accessed"] or checks["model_inference"]
    )
    result = {
        "classification": (
            "DRESS_REHEARSAL_PASS" if passed else "DRESS_REHEARSAL_FAIL"
        ),
        "checks": checks,
        "models": model_results,
        "maximum_primary_independent_metric_difference": max(metric_differences, default=0.0),
        "synthetic_fixture_items": q1.N_ITEMS,
    }
    temporary = REVIEW / "DRESS_REHEARSAL.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REVIEW / "DRESS_REHEARSAL.json")
    print(json.dumps({"classification": result["classification"]}, indent=2))
    if result["classification"] != "DRESS_REHEARSAL_PASS":
        raise RuntimeError("Q1 confirmatory dress rehearsal failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
