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
Q2_MATCHED_RANDOM_DIR = ROOT / "review" / "q2_matched_random_rank8_control_design"
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


def _validate_active_state(state: dict[str, object], failures: list[str]) -> None:
    """Validate open and closed-design states without weakening either firewall."""
    current = state["current"]
    firewall = state["scientific_firewall"]
    assert isinstance(current, dict)
    assert isinstance(firewall, dict)
    active = current.get("active_experiment", {})
    assert isinstance(active, dict)
    status = active.get("status")

    if status == "OPEN_RUNNING_BLIND_COLLECTION":
        if active.get("partial_scientific_outcomes_available_to_docs") is not False:
            failures.append("open-experiment firewall is not explicit")
        return

    if status == "NOT_RUN":
        if firewall.get("free_generation") != "NONE_AUTHORIZED":
            failures.append("closed-state firewall does not prohibit new free generation")
        if current.get("gpu_work_authorized") is not False:
            failures.append("closed-state project status unexpectedly authorizes GPU work")
        return

    if status == "DESIGN_REVIEW_COMPLETE_NOT_RUN":
        if firewall.get("free_generation") != "NONE_AUTHORIZED":
            failures.append("design-review state does not prohibit new free generation")
        if current.get("gpu_work_authorized") is not False:
            failures.append("design-review state unexpectedly authorizes GPU work")
        if current.get("new_scientific_experiment_authorized") != (
            "MODEL_FREE_THEORY_ONLY_NO_EXECUTION"
        ):
            failures.append("design-review state does not explicitly prohibit execution")
        if active.get("name") != "MATCHED_RANDOM_RANK8_CONTROL_DESIGN":
            failures.append("design-review state names an unexpected workstream")
        if active.get("evidence_level") != "DESIGN_ONLY":
            failures.append("design-review state is not explicitly design-only")
        ruling_path = Q2_MATCHED_RANDOM_DIR / "DESIGN_RULING.json"
        safety_path = Q2_MATCHED_RANDOM_DIR / "RELEASE_SAFETY_AUDIT.json"
        if not ruling_path.is_file() or not safety_path.is_file():
            failures.append("design-review closeout artifacts are missing")
            return
        ruling = json.loads(ruling_path.read_text(encoding="utf-8"))
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        if ruling.get("status") != current.get("stage"):
            failures.append("design-review ruling and project state disagree")
        zero_state = {
            "closed_results_changed": False,
            "final_random_bases_generated": 0,
            "experimental_seeds_generated": 0,
            "safety_inference": 0,
            "semantic_trajectories": 0,
            "qwen_loaded": False,
            "gpu_used": False,
            "q3_run": False,
        }
        for key, expected in zero_state.items():
            if ruling.get(key) != expected:
                failures.append(f"design-review firewall mismatch: {key}")
        if safety.get("status") != "PASS":
            failures.append("design-review release-safety audit did not pass")
        return

    failures.append("active scientific state is not a recognized fail-closed state")


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

    _validate_active_state(state, failures)

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
    print("external readiness: public claims and tracked Q2 hashes valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
