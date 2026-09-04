#!/usr/bin/env python3
"""Freeze the Q3.1 prompt-representation development precheck."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_route_a_prompt_representation"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
Q3_PRECHECK = (
    ROOT / "review/q3_realizable_utility_design/Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK.json"
)
Q3_FOLDS = ROOT / "review/q3_realizable_utility_design/CROSS_FIT_RESULTS.json"
CAPTURE_RUNNER = ROOT / "scripts/run_q3_prompt_representation_capture.py"
ANALYSIS_RUNNER = ROOT / "scripts/analyze_q3_prompt_representation.py"
MODEL_MANIFEST = ROOT / "review/q2_v4_spark1_presemantic/EXACT_MODEL_MANIFEST.json"
PROMPT_BUILDER = ROOT / "src/epistemic_geometry/benchmarks/prompts.py"
HF_BACKEND = ROOT / "src/epistemic_geometry/backends/huggingface.py"
BASE_COMMIT = "da90220311ad710794233745677430067bf30d75"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-prompt-manifest", type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    if len(panel.get("items", [])) != 300:
        raise RuntimeError("Q3.1 panel count changed")
    items = [
        {
            "item_id": str(row["item_id"]),
            "official_index": int(row["official_index"]),
            "order": int(row["order"]),
            "prompt": str(row["prompt"]),
            "prompt_sha256": str(row["prompt_sha256"]),
        }
        for row in panel["items"]
    ]
    prompt_manifest = {
        "schema_version": "q3-route-a-private-prompt-manifest-v1",
        "source_panel_sha256": sha256_file(PANEL),
        "item_count": 300,
        "item_ids": [row["item_id"] for row in items],
        "reference_answers_included": False,
        "correctness_included": False,
        "items": items,
    }
    write_json(args.private_prompt_manifest, prompt_manifest)
    repeat_ids = sorted(
        prompt_manifest["item_ids"],
        key=lambda item: hashlib.sha256(f"Q3.1_CAPTURE_REPEAT_V1|{item}".encode()).hexdigest(),
    )[:16]
    folds_payload = json.loads(Q3_FOLDS.read_text(encoding="utf-8"))
    fold_assignment = folds_payload["fold_assignment"]
    fold_sha = hashlib.sha256(
        json.dumps(fold_assignment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    precheck = {
        "schema_version": "q3-route-a-prompt-representation-precheck-v1",
        "status": "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK_FROZEN",
        "created_utc": datetime.now(UTC).isoformat(),
        "evidence_class": ["DEVELOPMENT_ONLY", "POST_CLOSED_RESULT_PLANNING"],
        "lineage": {
            "q3_0_closed_commit": BASE_COMMIT,
            "q3_0_ruling": "Q3_FRESH_HOLDOUT_INSUFFICIENT",
            "branch": "research/q3-route-a-prompt-representation",
            "q3_scientific_state": "NOT_RUN",
        },
        "scientific_question": (
            "Does an unsteered label-free prompt representation make one-call policy "
            "selectability stable, and does true controller geometry add incremental value "
            "over capacity-matched geometry-blind policy identity?"
        ),
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "environment_fingerprint": (
                "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
            ),
            "python": "3.12.3",
            "torch": "2.13.0+cu130",
            "torch_cuda": "13.0",
            "transformers": "4.57.6",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "model_manifest_sha256": sha256_file(MODEL_MANIFEST),
        },
        "capture": {
            "source_panel": str(PANEL.relative_to(ROOT)),
            "source_panel_sha256": sha256_file(PANEL),
            "development_family_count": 300,
            "item_ids": prompt_manifest["item_ids"],
            "private_prompt_manifest_sha256": sha256_file(args.private_prompt_manifest),
            "prompt_mode": "chat",
            "enable_thinking": False,
            "prompt_builder_sha256": sha256_file(PROMPT_BUILDER),
            "huggingface_backend_sha256": sha256_file(HF_BACKEND),
            "representation_site": "layer_27_block_input_final_nonpadding_prompt_token",
            "equivalence_candidate": "layer_26_block_output_final_nonpadding_prompt_token",
            "expected_width": 4096,
            "ordinary_unsteered_forward": True,
            "semantic_generation": 0,
            "reference_or_correctness_loading": False,
            "repeat_subset_selection": "16 smallest SHA256(Q3.1_CAPTURE_REPEAT_V1|item_id)",
            "repeat_subset_item_ids": repeat_ids,
            "repeat_captures_per_subset_item": 2,
            "full_capture_repeats": 1,
            "expected_prompt_only_forward_count": 332,
            "repeat_max_abs_tolerance": 1e-5,
            "site_equivalence_max_abs_tolerance": 0.0,
        },
        "single_forward_mechanism": {
            "capture": "layer-27 block forward-pre-hook reads current final prompt token",
            "selection": "frozen CPU router selects once during prompt prefill",
            "steering": (
                "a pre-registered layer-27 forward hook applies the selected frozen delta to "
                "the current-token block output during the same prefill and every decode step"
            ),
            "historical_token_or_kv_retroactive_change": False,
            "must_be_verified_with_synthetic_and_prompt-only_tests": True,
        },
        "preprocessing": {
            "representation_dimensions": [8, 16, 32],
            "method": "outer/inner-training-only centered SVD PCA",
            "normalization": "training-only component standardization",
            "global_pca_forbidden": True,
            "outcome_free_fixed_projection_control": {
                "dimension": 32,
                "distribution": "Rademacher/sqrt(4096)",
                "seed": 2026090507,
            },
        },
        "policy_banks": {
            "primary": {"method": "A0_MAXIMIN", "K": 8, "include_baseline": False},
            "secondary": [
                {"method": "A1_MAXIMIN", "K": 8, "include_baseline": False},
                {"method": "A2_MAXIMIN", "K": 8, "include_baseline": False},
            ],
            "selection": "exact Q3.0 outer-training-only shell and maximin rules",
            "secondary_cannot_rescue_primary": True,
        },
        "models": {
            "primary": "LOW_RANK_LOGISTIC_TRUE_CONTROLLER_GEOMETRY",
            "controls": [
                "LOW_RANK_LOGISTIC_LEARNED_POLICY_IDENTITY",
                "LOW_RANK_LOGISTIC_FIXED_PERMUTED_CONTROLLER_COORDINATES",
                "Q3_0_DETERMINISTIC_PROMPT_STRUCTURE",
                "FROZEN_GLOBAL_CHAMPION",
                "FROZEN_POLICY_PRIOR_RANDOM_ROUTER",
            ],
            "interaction_ranks": [1, 2, 4],
            "l2_grid": [0.1, 1.0, 10.0],
            "optimizer": {
                "name": "deterministic_full_batch_adam",
                "learning_rate": 0.03,
                "steps": 400,
                "initialization_seed": 2026090511,
            },
            "capacity_match": (
                "K=8 and controller width=8 make geometry and learned-policy-identity "
                "factorizations parameter matched at d*r + 8*r + 8 biases"
            ),
            "coordinate_permutation_seed": 2026090509,
            "inner_selection_tie_break": (
                "maximum inner routed accuracy, then smaller representation dimension, "
                "smaller interaction rank, then larger L2"
            ),
            "secondary_bank_hyperparameters": (
                "reuse the corresponding primary A0 per-fold/model hyperparameters; "
                "no secondary-bank retuning"
            ),
        },
        "nested_cross_fitting": {
            "outer_folds": 5,
            "inner_folds": 4,
            "family_partition": "exact Q3.0 balanced hash folds",
            "fold_assignment_sha256": fold_sha,
            "split_unit": "CRUXEval item/family",
            "all_policies_shells_rollouts_features_and_labels_coupled": True,
            "all_preprocessing_and_hyperparameters_training_only": True,
        },
        "feasibility_gates": {
            "absolute_routed_gain_min": 0.03,
            "oracle_headroom_fraction_min": 0.25,
            "positive_outer_folds_min": 4,
            "worst_fold_gain_min": -0.02,
            "commitment_validity_harm_max": 0.02,
            "semantic_evaluability_harm_max": 0.02,
            "incremental_true_geometry_over_blind_gain_min": 0.01,
            "incremental_true_geometry_over_permuted_gain_min": 0.01,
            "incremental_nonnegative_outer_folds_min": 4,
            "maximum_overall_single_policy_selection_share": 0.60,
            "minimum_distinct_selected_policies": 3,
            "rationale": (
                "One point is one third of the minimum primary utility gain and prevents a "
                "negligible geometry attribution; fold and concentration gates require the "
                "increment to be distributed rather than driven by one split or policy."
            ),
        },
        "reporting": [
            "routed and champion accuracy",
            "oracle fraction realized",
            "five fold gains and dispersion",
            "true-minus-blind and true-minus-permuted gains",
            "Brier score, log loss and 10-bin expected calibration error",
            "policy selection frequencies",
            "prompt-only forward and prefill compute",
        ],
        "terminal_rulings": [
            "Q3_ROUTE_A_REPRESENTATION_GEOMETRY_READY_FOR_HOLDOUT_DESIGN",
            "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL",
            "Q3_ROUTE_A_REPRESENTATION_UNSTABLE",
            "Q3_ROUTE_A_SINGLE_FORWARD_DEPLOYMENT_INFEASIBLE",
            "Q3_ROUTE_A_REPRESENTATION_CAPTURE_INVALID",
        ],
        "ruling_precedence": [
            "capture invalid",
            "single-forward infeasible",
            "realization gates fail -> unstable",
            (
                "realization passes but incremental geometry fails -> "
                "selectable but not geometry incremental"
            ),
            "all realization and incremental geometry gates pass -> ready for holdout design",
        ],
        "future_holdout_firewall": {
            "allocated": False,
            "outcomes_inspected": False,
            "prompt_capture": 0,
            "candidate_policy_generation": 0,
        },
        "implementation": {
            "capture_runner": str(CAPTURE_RUNNER.relative_to(ROOT)),
            "capture_runner_sha256": sha256_file(CAPTURE_RUNNER),
            "analysis_runner": str(ANALYSIS_RUNNER.relative_to(ROOT)),
            "analysis_runner_sha256": sha256_file(ANALYSIS_RUNNER),
            "precheck_builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "q3_0_precheck_sha256": sha256_file(Q3_PRECHECK),
        },
        "automatic_followup": False,
    }
    write_json(REVIEW / "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK.json", precheck)
    print(
        json.dumps(
            {
                "status": precheck["status"],
                "private_prompt_manifest_sha256": precheck["capture"][
                    "private_prompt_manifest_sha256"
                ],
                "repeat_subset_count": len(repeat_ids),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
