#!/usr/bin/env python3
"""Validate public scientific claims without reading private/open outcomes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
Q2_DIR = ROOT / "review" / "q2_v4_1_semantic_execution"
Q1_LCB_DIR = ROOT / "review" / "q1_second_task_spark2_design" / "amendment1_hierarchical_unit"
Q2_OOS_DIR = ROOT / "review" / "q2_oos_fresh_controller_design" / "v2_semantic_execution"
Q3_DESIGN_DIR = ROOT / "review" / "q3_realizable_utility_design"
Q3_ROUTE_A_DIR = ROOT / "review" / "q3_route_a_prompt_representation"
Q3_ROLE_DIR = ROOT / "review" / "q3_geometry_role_decomposition"
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "START_HERE.md",
    ROOT / "docs" / "SCIENTIFIC_RESULTS.md",
    ROOT / "docs" / "CURRENT_STATUS.md",
    ROOT / "docs" / "CLAIM_EVIDENCE_MATRIX.md",
    ROOT / "docs" / "EXPERIMENT_INDEX.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []
    state = yaml.safe_load((ROOT / "project_state.yaml").read_text(encoding="utf-8"))
    closeout = json.loads((Q2_DIR / "Q2_V4_1_SEMANTIC_CLOSEOUT.json").read_text(encoding="utf-8"))
    ledger = json.loads((Q2_DIR / "ARTIFACT_HASHES.json").read_text(encoding="utf-8"))
    q1_lcb = json.loads(
        (Q1_LCB_DIR / "stage_b_closeout" / "STAGE_B_CLOSEOUT.json").read_text(encoding="utf-8")
    )
    q1_lcb_audit = json.loads(
        (
            Q1_LCB_DIR
            / "stage_b_forensic_resolution"
            / "INDEPENDENT_STAGE_B_FORENSIC_AUDIT_RESOLVED.json"
        ).read_text(encoding="utf-8")
    )
    q2_oos = json.loads(
        (Q2_OOS_DIR / "Q2_OOS_V2_PRIMARY_RESULT_SEAL.json").read_text(encoding="utf-8")
    )
    q2_oos_audit = json.loads(
        (Q2_OOS_DIR / "Q2_OOS_V2_FORENSIC_AUDIT.json").read_text(encoding="utf-8")
    )
    q3_protocol = json.loads(
        (Q3_DESIGN_DIR / "RECOMMENDED_PROTOCOL_DRAFT.json").read_text(encoding="utf-8")
    )
    q3_holdout = json.loads(
        (Q3_DESIGN_DIR / "FRESH_HOLDOUT_FEASIBILITY.json").read_text(encoding="utf-8")
    )
    q3_hashes = json.loads((Q3_DESIGN_DIR / "ARTIFACT_HASHES.json").read_text(encoding="utf-8"))
    q3_route_a = json.loads(
        (Q3_ROUTE_A_DIR / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    q3_route_a_forensics = json.loads(
        (Q3_ROUTE_A_DIR / "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSICS.json").read_text(
            encoding="utf-8"
        )
    )
    q3_route_a_hashes = json.loads(
        (Q3_ROUTE_A_DIR / "Q3_ROUTE_A_PROMPT_REPRESENTATION_ARTIFACT_HASHES.json").read_text(
            encoding="utf-8"
        )
    )
    q3_role = json.loads(
        (Q3_ROLE_DIR / "Q3_GEOMETRY_ROLE_DECOMPOSITION_RELEASE_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    q3_role_safety = json.loads(
        (Q3_ROLE_DIR / "Q3_GEOMETRY_ROLE_RELEASE_SAFETY.json").read_text(encoding="utf-8")
    )
    q3_role_hashes = json.loads(
        (Q3_ROLE_DIR / "Q3_GEOMETRY_ROLE_ARTIFACT_HASHES.json").read_text(encoding="utf-8")
    )

    allowed_q2_states = {"Q2_V4_1_G2", "Q2_V4_1_G2__Q2_OOS_V2_A0_PASS"}
    if state["scientific_firewall"]["geometry_q2"] not in allowed_q2_states:
        failures.append("project_state Q2 classification is not a recognized closed state")
    if state["scientific_firewall"]["committee_q3"] != "NOT_RUN":
        failures.append("project_state incorrectly opens or classifies Q3")
    if closeout["classification"] != "Q2_V4_1_G2":
        failures.append("Q2 closeout classification changed")
    if closeout["radial"]["shape"]["classification"] != "RS+":
        failures.append("Q2 radial shape classification changed")
    if closeout["radial"]["total"]["classification"] != "RT+":
        failures.append("Q2 radial total classification changed")
    if closeout["forensic"]["classification"] != "Q2_V4_1_SEMANTIC_FORENSIC_CLEAN":
        failures.append("Q2 forensic classification changed")
    if q1_lcb["classification"] != "Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY":
        failures.append("Q1 LiveCodeBench closeout classification changed")
    if (
        q1_lcb_audit["forensic_classification"]
        != "Q1_SECOND_TASK_STAGE_B_FORENSIC_RESOLVED_PRIMARY_CONFIRMED"
    ):
        failures.append("Q1 LiveCodeBench resolved forensic state changed")
    if q2_oos["primary_classification"] != "Q2_OOS_V2_A0_PASS":
        failures.append("Q2 OOS primary classification changed")
    if q2_oos_audit["status"] != "Q2_OOS_V2_FORENSIC_CLEAN":
        failures.append("Q2 OOS forensic classification changed")

    required_tokens = (
        "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL",
        "Q2_V4_1_G2",
        "RS+",
        "RT+",
        "Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY",
        "Q2_OOS_V2_A0_PASS",
        "Q2_OOS_V2_FORENSIC_CLEAN",
        "Q3_FRESH_HOLDOUT_INSUFFICIENT",
        "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL",
        "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING",
        "Q3",
    )
    forbidden_current_phrases = (
        "no scientific q2 outcome exists",
        "q2 geometry: not run",
        "q2 is not run",
        "q2 remains untested",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
    for token in required_tokens:
        if token not in combined:
            failures.append(f"public documentation omits {token}")
    lowered = combined.lower()
    for phrase in forbidden_current_phrases:
        if phrase in lowered:
            failures.append(f"stale public claim: {phrase}")

    active = state["current"].get("active_experiment", {})
    if active.get("status") == "OPEN_RUNNING_BLIND_COLLECTION":
        if active.get("partial_scientific_outcomes_available_to_docs") is not False:
            failures.append("open-experiment firewall is not explicit")
    elif active.get("status") == "NOT_RUN":
        if state["scientific_firewall"].get("free_generation") != "NONE_AUTHORIZED":
            failures.append("closed-state firewall does not prohibit new free generation")
        if state["current"].get("gpu_work_authorized") is not False:
            failures.append("closed-state project status unexpectedly authorizes GPU work")
    elif active.get("status") == "Q3_FRESH_HOLDOUT_INSUFFICIENT":
        if active.get("name") != "Q3_0_REALIZABLE_COLLECTIVE_UTILITY_DESIGN":
            failures.append("Q3 design-only state has an unexpected active-experiment identity")
        if active.get("evidence_level") != "DESIGN_ONLY":
            failures.append("Q3 design-only state is not explicitly labeled DESIGN_ONLY")
        if active.get("branch") != "research/q3-realizable-utility-design":
            failures.append("Q3 design-only state has an unexpected source branch")
        if state["scientific_firewall"].get("free_generation") != "NONE_AUTHORIZED":
            failures.append("Q3 design-only state does not prohibit free generation")
        if state["current"].get("gpu_work_authorized") is not False:
            failures.append("Q3 design-only state unexpectedly authorizes GPU work")
        if q3_protocol.get("final_design_ruling") != "Q3_FRESH_HOLDOUT_INSUFFICIENT":
            failures.append("Q3 protocol draft does not match the closed design ruling")
        if q3_protocol.get("execution_authorized") is not False:
            failures.append("Q3 protocol draft unexpectedly authorizes execution")
        if q3_holdout.get("future_holdout_permanently_allocated") is not False:
            failures.append("Q3 design-only state unexpectedly allocates a future holdout")
        if q3_holdout.get("future_outcomes_inspected") is not False:
            failures.append("Q3 design-only state reports inspection of future outcomes")
        if q3_hashes.get("classification") != "Q3_FRESH_HOLDOUT_INSUFFICIENT":
            failures.append("Q3 artifact manifest does not match the closed design ruling")
        if q3_hashes.get("q3_semantic_trajectories") != 0:
            failures.append("Q3 artifact manifest reports semantic trajectories")
        if q3_hashes.get("raw_text_included") is not False:
            failures.append("Q3 design package reports raw text in release artifacts")
        for relative, expected in q3_hashes.get("artifacts", {}).items():
            if not relative.startswith("review/q3_realizable_utility_design/"):
                continue
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                failures.append(f"Q3 design artifact hash mismatch: {relative}")
    elif (
        active.get("status") == "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL"
    ):
        if active.get("name") != "Q3_1_LABEL_FREE_PROMPT_REPRESENTATION_SELECTABILITY":
            failures.append("Q3.1 closed state has an unexpected experiment identity")
        if active.get("evidence_level") != "DEVELOPMENT_ONLY":
            failures.append("Q3.1 closed state is not explicitly DEVELOPMENT_ONLY")
        if active.get("branch") != "research/q3-route-a-prompt-representation":
            failures.append("Q3.1 closed state has an unexpected source branch")
        if state["scientific_firewall"].get("free_generation") != "NONE_AUTHORIZED":
            failures.append("Q3.1 closed state does not prohibit free generation")
        if state["current"].get("gpu_work_authorized") is not False:
            failures.append("Q3.1 closed state unexpectedly authorizes GPU work")
        if q3_protocol.get("final_design_ruling") != "Q3_FRESH_HOLDOUT_INSUFFICIENT":
            failures.append("Q3.1 parent Q3.0 ruling changed")
        expected_ruling = "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL"
        if q3_route_a.get("status") != expected_ruling:
            failures.append("Q3.1 release summary ruling changed")
        if q3_route_a.get("scientific_state") != "Q3_NOT_RUN_DEVELOPMENT_ONLY":
            failures.append("Q3.1 release summary upgrades Q3")
        firewall = q3_route_a.get("firewall", {})
        if firewall.get("new_semantic_trajectories") != 0:
            failures.append("Q3.1 release summary reports semantic trajectories")
        if firewall.get("fresh_evaluation_outcomes_inspected") is not False:
            failures.append("Q3.1 release summary reports fresh-outcome inspection")
        capture = q3_route_a.get("capture", {})
        if capture.get("prompt_only_forward_count") != 332:
            failures.append("Q3.1 prompt-only forward count changed")
        if capture.get("single_forward_deployment_feasible") is not True:
            failures.append("Q3.1 single-forward feasibility state changed")
        incremental = q3_route_a.get("gate_results", {}).get("incremental_geometry", {})
        if incremental.get("true_over_blind_threshold") is not False:
            failures.append("Q3.1 geometry-blind attribution gate changed")
        if incremental.get("fold_consistency") is not False:
            failures.append("Q3.1 incremental fold gate changed")
        if (
            q3_route_a_forensics.get("status")
            != "Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSIC_CLEAN"
        ):
            failures.append("Q3.1 capture forensic status changed")
        if q3_route_a_forensics.get("semantic_generation") != 0:
            failures.append("Q3.1 capture forensics reports semantic generation")
        if q3_route_a_forensics.get("reference_or_correctness_loaded") is not False:
            failures.append("Q3.1 capture forensics reports reference/correctness loading")
        for relative, expected in q3_route_a_hashes.get("artifacts", {}).items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                failures.append(f"Q3.1 artifact hash mismatch: {relative}")
    elif active.get("status") == "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING":
        if active.get("name") != "Q3_2_GEOMETRY_ROLE_DECOMPOSITION":
            failures.append("Q3.2 closed state has an unexpected experiment identity")
        if active.get("evidence_level") != "DEVELOPMENT_ONLY":
            failures.append("Q3.2 closed state is not explicitly DEVELOPMENT_ONLY")
        if active.get("branch") != "research/q3-geometry-role-decomposition":
            failures.append("Q3.2 closed state has an unexpected source branch")
        if state["scientific_firewall"].get("free_generation") != "NONE_AUTHORIZED":
            failures.append("Q3.2 closed state does not prohibit free generation")
        if state["current"].get("gpu_work_authorized") is not False:
            failures.append("Q3.2 closed state unexpectedly authorizes GPU work")
        if q3_role.get("status") != "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING":
            failures.append("Q3.2 release summary ruling changed")
        if q3_role.get("scientific_state") != "Q3_NOT_RUN_DEVELOPMENT_ONLY":
            failures.append("Q3.2 release summary upgrades Q3")
        if q3_role.get("immutable_q3_1", {}).get("classification") != (
            "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL"
        ):
            failures.append("Q3.2 release summary changes Q3.1")
        if q3_role.get("part_a", {}).get("ruling") != "GEOMETRY_BANK_SELECTION_SUPPORTED":
            failures.append("Q3.2 Part-A ruling changed")
        if q3_role.get("part_b", {}).get("ruling") != "CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED":
            failures.append("Q3.2 Part-B ruling changed")
        firewall = q3_role.get("firewall", {})
        if firewall.get("new_semantic_trajectories") != 0:
            failures.append("Q3.2 release summary reports new semantic trajectories")
        if firewall.get("new_qwen_forwards") != 0:
            failures.append("Q3.2 release summary reports new Qwen forwards")
        if firewall.get("fresh_evaluation_outcomes_inspected") is not False:
            failures.append("Q3.2 release summary reports future outcome inspection")
        future = q3_role.get("future_instrument", {})
        if future.get("holdout_allocated") is not False:
            failures.append("Q3.2 unexpectedly allocates a future holdout")
        if future.get("future_outcomes_inspected") is not False:
            failures.append("Q3.2 reports inspection of future outcomes")
        if q3_role_safety.get("status") != "Q3_GEOMETRY_ROLE_RELEASE_SAFETY_PASS":
            failures.append("Q3.2 release-safety status changed")
        for key in (
            "raw_benchmark_text_included",
            "raw_model_outputs_included",
            "prompt_representation_values_included",
            "private_itemwise_outcomes_included",
            "credentials_or_infrastructure_included",
        ):
            if q3_role_safety.get(key) is not False:
                failures.append(f"Q3.2 release-safety violation: {key}")
        for relative, expected in q3_role_hashes.get("artifacts", {}).items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                failures.append(f"Q3.2 artifact hash mismatch: {relative}")
    else:
        failures.append("active scientific state is not a recognized fail-closed lifecycle state")

    for name, expected in ledger["tracked_aggregate_artifacts"].items():
        path = Q2_DIR / name
        if not path.is_file() or _sha256(path) != expected:
            failures.append(f"Q2 tracked aggregate hash mismatch: {name}")
    for name, expected in ledger["closeout_artifacts"].items():
        path = Q2_DIR / name
        if not path.is_file() or _sha256(path) != expected:
            failures.append(f"Q2 closeout hash mismatch: {name}")
    for relative, expected in ledger["posthoc_diagnostic_implementation"].items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            failures.append(f"Q2 implementation hash mismatch: {relative}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("external readiness: public claims and tracked Q2/Q3 hashes valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
