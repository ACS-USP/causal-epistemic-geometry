#!/usr/bin/env python3
"""Freeze the Q3.2 closed-data geometry-role decomposition precheck."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_geometry_role_decomposition"
OUTPUT = REVIEW / "Q3_GEOMETRY_ROLE_DECOMPOSITION_PRECHECK.json"
ANALYSIS = ROOT / "scripts/analyze_q3_geometry_role_decomposition.py"
PANEL = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"
Q31_SUMMARY = (
    ROOT
    / "review/q3_route_a_prompt_representation"
    / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json"
)
Q31_PRECHECK = (
    ROOT
    / "review/q3_route_a_prompt_representation"
    / "Q3_ROUTE_A_PROMPT_REPRESENTATION_PRECHECK.json"
)

EXPECTED = {
    "representation_matrix": "3612a645e3739e3cf7bf4d32f1f808034b15604a1e7f99e784c45e04b49d81ac",
    "historical_scores": "a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f",
    "fresh_scores": "9f03d96d40839e228d6cfb55408ea056e262fbf7e9aef2e863080e035e4b721b",
    "panel": "c127cf3594e8ea849dbd038492606b3afaaac406feb4146188769c04d6691187",
}


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--fresh-scores", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "representation_matrix": args.representations,
        "historical_scores": args.historical_scores,
        "fresh_scores": args.fresh_scores,
        "panel": PANEL,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"Q3.2 source hash mismatch: {observed}")
    q31 = read_json(Q31_SUMMARY)
    if q31.get("status") != "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL":
        raise RuntimeError("immutable Q3.1 classification mismatch")
    q31_precheck = read_json(Q31_PRECHECK)
    payload = {
        "schema_version": "q3-geometry-role-decomposition-precheck-v1",
        "status": "Q3_GEOMETRY_ROLE_DECOMPOSITION_PRECHECK_FROZEN",
        "evidence_class": ["DEVELOPMENT_ONLY", "POST_CLOSED_RESULT_PLANNING"],
        "lineage": {
            "branch": "research/q3-geometry-role-decomposition",
            "parent_branch": "research/q3-route-a-prompt-representation",
            "parent_commit": "9c8c3fd5f34fcc8262be01d5250b5db7a1517323",
        },
        "immutable_q3_1": {
            "classification": "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL",
            "primary_routed_gain": 0.05333333333333334,
            "true_minus_geometry_blind_gain": 0.003333333333333334,
            "positive_outer_folds": 5,
            "single_forward_deployment_feasible": True,
            "classification_mutable": False,
        },
        "sources": {
            "representation_matrix_sha256": observed["representation_matrix"],
            "historical_scores_sha256": observed["historical_scores"],
            "fresh_scores_sha256": observed["fresh_scores"],
            "semantic_panel_sha256": observed["panel"],
            "q3_1_summary_sha256": sha256_file(Q31_SUMMARY),
            "q3_1_precheck_sha256": sha256_file(Q31_PRECHECK),
            "q3_1_family_fold_sha256": q31_precheck["nested_cross_fitting"][
                "fold_assignment_sha256"
            ],
            "raw_generated_text_loaded": False,
        },
        "shared_design": {
            "scientific_unit": "CRUXEval item/family",
            "development_families": 300,
            "outer_folds": 5,
            "inner_folds": 4,
            "folds": "exact Q3.0/Q3.1 balanced hash folds",
            "both_rollouts_and_all_policies_coupled_by_family": True,
            "all_preprocessing_and_tuning_outer_training_only": True,
            "baseline_included": False,
        },
        "router": {
            "family": "regularized low-rank logistic prompt-policy interaction",
            "part_a_model": "Q3.1 capacity-matched learned policy identity",
            "part_a_hyperparameters": "reuse exact Q3.1 A0 geometry-blind fold choices",
            "representation_dimensions": [8, 16],
            "interaction_ranks": [1, 2],
            "l2_grid": [0.1, 1.0, 10.0],
            "transfer_tuning_objective": (
                "historical-controller inner-fold log loss; then Brier, lower "
                "dimension/rank, stronger regularization"
            ),
            "optimizer": "deterministic full-batch Adam",
            "optimizer_steps": 400,
            "transfer_optimizer_steps": 200,
            "learning_rate": 0.03,
        },
        "part_a": {
            "question": (
                "Does A0-maximin K=8 improve realizable routing through bank construction alone?"
            ),
            "controller_population": "31 historical plus 16 closed OOS fresh controllers",
            "policy_eligibility": "one outer-training-selected MEDIUM/STRONG shell per controller",
            "bank_size": 8,
            "fixed_banks": [
                "A0_MAXIMIN",
                "A1_MAXIMIN",
                "A2_MAXIMIN",
                "OUTCOME_OPTIMIZED_TRAINING_ONLY_UPPER_BOUND",
            ],
            "random_distributions": [
                "DETERMINISTIC_RANDOM",
                "COMPETENCE_MATCHED_RANDOM",
                "LOW_A0_DIVERSITY",
            ],
            "candidate_random_bank_pool_per_fold": 4096,
            "evaluated_banks_per_distribution": 512,
            "low_diversity_match_pool": 2048,
            "random_bank_seed": 2026090601,
            "router_initialization_seed": q31_precheck["models"]["optimizer"][
                "initialization_seed"
            ],
            "matching": {
                "variables": [
                    "outer-training mean accuracy",
                    "commitment validity",
                    "semantic evaluability",
                    "mean generated tokens",
                ],
                "metric": "Euclidean distance after candidate-pool SD scaling",
                "selection": "512 closest unique controller-subset banks per fold",
                "shell_choice": "outer-training-only Q3.0 rule",
            },
            "negative_control": (
                "lowest mean pairwise A0 diversity among the 2048 closest "
                "competence-matched candidates"
            ),
            "primary_statistic": (
                "empirical percentile/rank of A0 cross-fitted routed gain in "
                "competence-matched random banks"
            ),
            "opportunity_statistic": (
                "empirical percentile/rank of A0 oracle headroom in competence-matched random banks"
            ),
            "paired_diagnostic": (
                "plus-one upper-tail random-bank comparison; shared item folds "
                "are not treated as IID banks"
            ),
            "minimum_percentile": 0.95,
            "randomization_alpha": 0.05,
            "minimum_gain_over_matched_median": 0.01,
            "minimum_nonnegative_folds": 4,
            "realization_gain_min": 0.03,
            "ruling": {
                "all_gates_pass": "GEOMETRY_BANK_SELECTION_SUPPORTED",
                "otherwise": "GEOMETRY_BANK_SELECTION_NOT_SUPPORTED",
            },
        },
        "part_b": {
            "question": (
                "Can true coordinates transfer prompt-policy routing from 31 "
                "historical to 16 held-out fresh controllers?"
            ),
            "training_controllers": "exact historical 31",
            "held_out_controllers": "exact prospectively sampled OOS 16",
            "controller_split_reshuffling": False,
            "fresh_outcomes_allowed_for_fit_or_tuning": False,
            "policies": "all direction-by-shell policies; 62 train and 32 held out",
            "true_descriptor": (
                "[shell_amplitude * unit_rank8_coefficient_coordinates, shell_amplitude]"
            ),
            "shell_amplitudes": {"MEDIUM": 0.25, "STRONG": 0.50},
            "descriptor_scaling": (
                "fit on historical training policies and applied to fresh policies"
            ),
            "modes": ["TRUE", "PERMUTED", "RANDOM", "AGNOSTIC"],
            "coordinate_permutation": "one frozen bijection over all 47 identities; no outcome use",
            "coordinate_permutation_seed": 2026090603,
            "random_coordinates": "independent normalized Gaussian 8-vectors for all 47 identities",
            "random_coordinate_seed": 2026090605,
            "model_initialization_seed": 2026090607,
            "controller_agnostic_descriptor": (
                "shell amplitude only; controller identities tied mechanically"
            ),
            "geometry_prior": (
                "Gaussian-kernel historical competence transfer with median "
                "historical descriptor distance bandwidth"
            ),
            "global_policy_prior": (
                "uniform fresh-controller expectation under historically "
                "preferred shell; descriptive"
            ),
            "primary_tuning": "historical-controller outcomes and inner item folds only",
            "predictive_metrics": [
                "held-out-controller log loss",
                "Brier score",
                "ECE-10",
                "mean itemwise policy-ranking Spearman",
                "top-1/top-3/top-5 best-policy recall",
            ],
            "routing_metrics": [
                "selected fresh-policy accuracy",
                "uniform random fresh-policy accuracy",
                "fresh-bank oracle accuracy",
                "oracle fraction realized",
                "fold stability",
                "per-fresh-controller predictive residuals",
            ],
            "realization_gain_min": 0.03,
            "minimum_positive_folds": 4,
            "worst_fold_gain_min": -0.02,
            "minimum_routing_gain_over_control": 0.01,
            "minimum_log_loss_improvement": 0.01,
            "minimum_nonnegative_folds": 4,
            "ruling": {
                "true_passes_all_realization_and_attribution_gates": (
                    "CONTROLLER_OOS_GEOMETRY_TRANSFER_SUPPORTED"
                ),
                "a_model_realizes_gain_but_true_attribution_fails": (
                    "CONTROLLER_OOS_SELECTABILITY_WITHOUT_GEOMETRY"
                ),
                "no_model_realizes_gain": "CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED",
            },
        },
        "high_level_ruling": {
            "part_a_supported_and_part_b_supported": (
                "Q3_GEOMETRY_BRIDGE_SUPPORTED_READY_FOR_FRESH_INSTRUMENT_DESIGN"
            ),
            "part_a_supported_part_b_not_supported": "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING",
            "part_a_not_supported_part_b_supported": "Q3_GEOMETRY_ROLE_UNRESOLVED",
            "part_a_not_supported_part_b_not_geometry_specific": (
                "Q3_CAUSAL_POLICY_SELECTABILITY_WITHOUT_GEOMETRY_ATTRIBUTION"
            ),
            "invalid_or_unresolved_execution": "Q3_GEOMETRY_ROLE_UNRESOLVED",
        },
        "future_instrument_roadmap": {
            "design_only": True,
            "minimum_family_count": 800,
            "categories": [
                "new executable CRUXEval-like program-tracing families",
                "family-disjoint exact-evaluator public benchmark",
                "separately generated deterministic program-execution benchmark",
            ],
            "items_generated": 0,
            "holdout_allocated": False,
        },
        "implementation": {
            "analysis_path": str(ANALYSIS.relative_to(ROOT)),
            "analysis_sha256": sha256_file(ANALYSIS),
            "precheck_builder_path": str(Path(__file__).resolve().relative_to(ROOT)),
        },
        "firewall": {
            "new_semantic_trajectories": 0,
            "new_qwen_forwards": 0,
            "new_prompt_capture": 0,
            "fresh_future_correctness_inspected": False,
            "new_controllers": 0,
            "new_random_subspaces": 0,
            "q3_confirmatory_experiment": "NOT_RUN",
            "spark1_gpu": False,
            "spark2": False,
            "runpod": False,
            "q1_q2_q3_1_reclassification": False,
            "paper_workspace_modified": False,
            "personal_handbook_modified": False,
            "automatic_followup": False,
        },
    }
    write_json(OUTPUT, payload)
    print(sha256_file(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
