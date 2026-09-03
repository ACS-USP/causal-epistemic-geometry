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

    active = state["current"].get("active_experiment", {})
    if active.get("status") == "OPEN_RUNNING_BLIND_COLLECTION":
        if active.get("partial_scientific_outcomes_available_to_docs") is not False:
            failures.append("open-experiment firewall is not explicit")
    elif active.get("status") == "NOT_RUN":
        if state["scientific_firewall"].get("free_generation") != "NONE_AUTHORIZED":
            failures.append("closed-state firewall does not prohibit new free generation")
        if state["current"].get("gpu_work_authorized") is not False:
            failures.append("closed-state project status unexpectedly authorizes GPU work")
    else:
        failures.append("active scientific state is neither blind/open nor closed design-only")

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
