#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze the model-free Q3.3 development-closure and supply-audit plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_final_system_and_evaluation_supply"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sources = {
        "q3_2_release_summary": "review/q3_geometry_role_decomposition/Q3_GEOMETRY_ROLE_DECOMPOSITION_RELEASE_SUMMARY.json",
        "q3_1_release_summary": "review/q3_route_a_prompt_representation/Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json",
        "q3_0_holdout_feasibility": "review/q3_realizable_utility_design/FRESH_HOLDOUT_FEASIBILITY.json",
        "q3_0_exposure_ledger": "review/q3_realizable_utility_design/ITEM_EXPOSURE_LEDGER.json",
        "cruxeval_provenance_ledger": "review/q2_m3_qualification_cruxeval_provenance/CRUXEVAL_PROVENANCE_LEDGER.jsonl",
        "q3_0_analysis": "scripts/design_q3_realizable_utility.py",
        "q3_1_analysis": "scripts/analyze_q3_prompt_representation.py",
    }
    source_records = {
        name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sources.items()
    }
    precheck = {
        "schema_version": "q3-final-system-evaluation-supply-precheck-v1",
        "status": "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_FROZEN",
        "evidence_class": ["DEVELOPMENT_ONLY", "CLOSED_DATA_PLANNING_ONLY"],
        "lineage": {
            "branch": "research/q3-final-system-and-evaluation-supply",
            "parent_branch": "research/q3-geometry-role-decomposition",
            "parent_commit": "cc799a79e786044c5d4d63f79ea38c1e03095362",
        },
        "immutable_parent": {
            "q3_2": "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING",
            "q3_2_mutable": False,
            "q3_status": "NOT_RUN",
            "development_panel_families": 300,
            "development_panel_closed_after_materialization": True,
            "a1_a2_pooling": False,
        },
        "sources": source_records,
        "final_system_materialization": {
            "outcomes_allowed": "closed 300-family development outcomes only",
            "bank": {
                "method": "A0_MAXIMIN",
                "size": 8,
                "controller_population": "exact frozen historical-31 plus fresh-16 identities",
                "shell_choice": "per-controller full-development Q3.0 lexicographic rule: accuracy, evaluability, lower mean tokens, MEDIUM tie preference",
                "maximin_ties": "lexicographic controller identity under the frozen implementation",
                "baseline_included": False,
            },
            "router": {
                "representation": "frozen L27 prompt representation",
                "mode": "GEOMETRY_BLIND_POLICY_ID",
                "controller_coordinates_used": False,
                "family": "regularized low-rank logistic prompt-policy interaction",
                "pca_dimension": 8,
                "interaction_rank": 2,
                "l2": 1.0,
                "hyperparameter_rule": "componentwise median/mode of the five Q3.1 outer-fold GEOMETRY_BLIND_POLICY_ID choices; no full-panel performance lookup",
                "optimizer": "deterministic full-batch Adam",
                "steps": 400,
                "learning_rate": 0.03,
                "initialization_seed": 2026090511,
                "targets": "mean correctness across the two closed rollouts for each bank policy",
                "pca": "fit once on all 300 closed development representations; persist mean, components, and score scales",
            },
            "champion": {
                "population": "all exact 94 direction-by-shell policies",
                "rule": "highest pooled correctness; then evaluability; then lower mean generated tokens; then lexicographic policy identity",
            },
            "status_after_materialization": "DEVELOPMENT_SELECTED_NOT_EVALUATED",
            "future_changes_forbidden_without_new_prelock": [
                "bank identities or shells",
                "router features or fitted parameters",
                "champion identity",
                "parser/evaluator",
                "primary endpoint",
            ],
        },
        "upper_bound_clarification": {
            "q3_2_outcome_optimized_bank": "upper bound on bank opportunity/construction under outer-training-only selection",
            "not_an_upper_bound_on": "cross-fitted routed accuracy",
            "scientific_result_changed": False,
        },
        "tier_b_audit": {
            "population": "exact 500 Q3.0 Tier-B CRUXEval families",
            "allocation_permitted": False,
            "future_correctness_permitted": False,
            "raw_content_inspection": False,
            "stratum_precedence": ["F", "E", "D", "C", "B", "A"],
            "strata": {
                "A": "stable ID/hash/manifest exposure only",
                "B": "prompt touched only by model-free parser/provenance tooling; no human raw-content inspection and no generation",
                "C": "model generation under unrelated conditions, with correctness never scored or inspected",
                "D": "correctness or semantic outcome scored/inspected under unrelated conditions",
                "E": "known manual raw item/output inspection",
                "F": "possible influence on final bank, router, parser, metric, threshold, protocol, or exact candidate-policy outcome",
            },
            "confirmatory_eligibility": "A only, with no unresolved provenance",
            "bounded_internal_validation_eligibility": "A or B only; no exact final-policy outcome, router access, manual inspection, or design influence",
            "unknown_or_conflicting_provenance": "fail closed as ineligible",
            "release_safe_fields_only": True,
        },
        "power_precision": {
            "n_families": [23, 100, 250, 400, 500, 800, 1000, 1200],
            "rollouts": [1, 2, 4, 6, 8],
            "accuracy_gains": [0.02, 0.03, 0.04, 0.05],
            "champion_accuracy": [0.35, 0.50, 0.65],
            "discordance": [0.10, 0.20, 0.35, 0.50],
            "family_latent_icc": [0.0, 0.15, 0.35],
            "seed_scenarios": ["independent policy seeds", "paired/common seeds sensitivity"],
            "endpoint": "family-weighted mean of rollout-mean routed-minus-champion correctness",
            "invalid_or_unevaluable": "incorrect",
            "missing": "blocks completion",
            "candidate_tests": [
                "one-sided studentized paired-family mean test",
                "one-sided studentized family sign-flip randomization test",
                "family-cluster bootstrap-t interval",
                "exact discordant-pair binomial test for R=1 diagnostic",
            ],
            "calibration_replicates": 20000,
            "power_replicates": 10000,
            "randomization_draws": 9999,
            "bootstrap_draws": 10000,
            "simulation_seed": 2026090701,
            "method_selection": [
                "FPR upper Wilson limit <= 0.065 in all prespecified null scenarios",
                "95% interval coverage >= 0.93 in all regular scenarios",
                "family is the independent inference unit",
                "simple and auditable",
                "power >= 0.80 for +0.03 under the designated conservative scenario",
                "smaller interval width only after calibration",
            ],
            "paired_seed_interpretation": "same numeric seed is not assumed to induce valid common-random-number coupling for divergent autoregressive policies",
            "deployment_calls": 1,
            "evaluation_rollouts_are_replication_not_deployment_calls": True,
        },
        "tier_b_route_gate": {
            "candidate_rollouts": [4, 6, 8],
            "maximum_families": 500,
            "minimum_power_for_gain_0_03": 0.80,
            "requires_frozen_system_before_allocation": True,
            "requires_no_final_policy_prior_outcomes": True,
            "requires_no_leakage": True,
            "purpose": "bounded internal validation only; never fresh confirmatory evidence",
        },
        "fresh_instrument_audit": {
            "minimum_target_families": 800,
            "routes": [
                "new executable CRUXEval-like program-tracing families",
                "family-disjoint public exact-evaluator benchmark",
                "separately generated deterministic program-execution benchmark",
                "hybrid public source programs with prospectively generated new inputs",
            ],
            "criteria": [
                "task and estimand match",
                "exact deterministic evaluator",
                "independent family definition",
                "contamination and adaptation risk",
                "licensing and redistribution",
                "template/source diversity",
                "development/evaluation separation",
                "difficulty calibration without final-evaluation outcomes",
                "reviewer credibility",
                "runtime",
            ],
            "restricted_content_download": False,
            "final_item_generation": 0,
            "holdout_allocation": 0,
        },
        "route_decision": {
            "I": "Tier B only: eligible and adequately powered bounded internal validation",
            "II": "fully fresh: defensible fresh exact-evaluator instrument and Tier B inadequate or too exposed",
            "III": "Tier B de-risking followed by unchanged fresh evaluation when both are independently justified",
            "if_adaptation_after_tier_b": "newly freeze the later instrument; Tier B remains development-only",
        },
        "future_primary": {
            "comparison": "frozen routed A0 portfolio versus frozen best single champion",
            "endpoint": "Delta_route: mean family-and-rollout correctness difference",
            "scientific_unit": "question family",
            "paired_by": "family and rollout index, without assuming RNG coupling from equal numeric seeds",
            "secondary_geometry_attribution": "matched-random-bank analysis only; cannot rescue primary utility",
        },
        "terminal_rulings": [
            "Q3_TIER_B_INTERNAL_VALIDATION_READY_FOR_PRELOCK",
            "Q3_FRESH_INSTRUMENT_DESIGN_READY_FOR_PRELOCK",
            "Q3_TWO_STAGE_VALIDATION_RECOMMENDED",
            "Q3_EVALUATION_SUPPLY_REQUIRES_FURTHER_THEORY",
            "Q3_NO_DEFENSIBLE_EVALUATION_POPULATION",
        ],
        "firewall": {
            "new_semantic_trajectories": 0,
            "new_qwen_forwards": 0,
            "new_prompt_capture": 0,
            "new_controllers": 0,
            "new_random_subspaces": 0,
            "fresh_correctness_inspected": False,
            "spark1_gpu": False,
            "spark2": False,
            "runpod": False,
            "q3_confirmatory": "NOT_RUN",
            "paper_workspace_modified": False,
            "personal_handbook_modified": False,
        },
    }
    REVIEW.mkdir(parents=True, exist_ok=True)
    path = REVIEW / "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json"
    path.write_text(json.dumps(precheck, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)} {sha256_file(path)}")


if __name__ == "__main__":
    main()
