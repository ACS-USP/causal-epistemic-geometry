#!/usr/bin/env python3
"""Close out Q2 V4.1 after label-free A1/A2 materialization.

This script validates only presemantic artifacts and writes the closeout and
future execution draft.  It cannot load a model, run generation, invoke a
semantic parser, or inspect correctness outcomes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
EXPECTED_PROFILE = "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    run_state = read_json(REVIEW / "LABEL_FREE_GEOMETRY_RUN.json")
    a1 = read_json(REVIEW / "A1_INSTRUMENT_QUALIFICATION.json")
    a2 = read_json(REVIEW / "A2_INSTRUMENT_QUALIFICATION.json")
    audit = read_json(REVIEW / "FORENSIC_AUDIT.json")
    environment = read_json(REVIEW / "ENVIRONMENT_PROVENANCE.json")
    raw = read_json(REVIEW / "A2_RAW_ARCHIVE_HASHES.json")
    reference = read_json(REVIEW / "A2_OFFLINE_REFERENCE_VALIDATION.json")
    if lock["controller_count"] != 31 or lock["semantic_outcomes"] != 0:
        raise RuntimeError("invalid V4.1 lock state")
    if run_state["semantic_outcomes"] != 0 or run_state["correctness_inspected"]:
        raise RuntimeError("semantic firewall is not closed")
    if a1["classification"] != "Q2_V4_1_A1_INSTRUMENT_QUALIFIED":
        raise RuntimeError("A1 did not qualify")
    if a2["classification"] != "Q2_V4_1_A2_INSTRUMENT_QUALIFIED":
        raise RuntimeError("A2 did not qualify")
    if audit["classification"] != "Q2_V4_1_LABEL_FREE_FORENSIC_CLEAN":
        raise RuntimeError("label-free forensic audit did not pass")
    if environment["qualified_environment_profile"] != EXPECTED_PROFILE:
        raise RuntimeError("environment fingerprint mismatch")
    if environment["model_revision"] != EXPECTED_MODEL_REVISION:
        raise RuntimeError("model revision mismatch")
    if raw["file_count"] != 24 or not reference["pass"]:
        raise RuntimeError("raw A2 reference validation is incomplete")
    artifact_names = [
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "A0_MEDIUM.npy",
        "A0_STRONG.npy",
        "A0_METADATA.json",
        "A1_COVARIANCE_MANIFEST.json",
        "A1_COVARIANCE_ACTIVATIONS.npz",
        "A1_COVARIANCE_METADATA.json",
        "A1_COVARIANCE_FIT.npz",
        "A1_INSTRUMENT_QUALIFICATION.json",
        "A2_PROBE_MANIFEST.json",
        "A2_RAW_ARCHIVE_HASHES.json",
        "A2_OFFLINE_REFERENCE_VALIDATION.json",
        "A2_INSTRUMENT_QUALIFICATION.json",
        "PREDICTION_MATRICES.npz",
        "PREDICTION_MATRIX_METADATA.json",
        "CONSOLIDATOR_TOLERANCES.json",
        "ENGINEERING_INCIDENTS.json",
        "ENGINEERING_RECOVERY.json",
        "INFRASTRUCTURE_TROUBLESHOOTING.json",
        "ENVIRONMENT_PROVENANCE.json",
        "ENVIRONMENT_PROVENANCE_SPARK1_CAPTURE.json",
        "LABEL_FREE_GEOMETRY_RUN.json",
        "FORENSIC_AUDIT.md",
        "FORENSIC_AUDIT.json",
        "METRIC_CROSSCHECK.csv",
        "FUTURE_SEMANTIC_SCHEDULE.json",
        "SEMANTIC_PANEL_MANIFEST.json",
        "QAP_CONTROLLER_PERMUTATIONS.npy",
        "QAP_SCHEDULE.json",
        "G3_POWER_CHARACTERIZATION.csv",
        "G3_POWER_CHARACTERIZATION.json",
        "LEVERAGE_RULING.md",
        "LEVERAGE_RULING.json",
        "SAFETY_HISTORY_ERRATUM.md",
        "SAFETY_HISTORY_ERRATUM.json",
        "PREPARATION_AUDIT.json",
    ]
    artifacts = {}
    for name in artifact_names:
        path = REVIEW / name
        if not path.is_file():
            raise RuntimeError(f"missing closeout artifact: {name}")
        artifacts[name] = sha256_file(path)
    digest = hashlib.sha256()
    for name, value in sorted(artifacts.items()):
        digest.update(f"{value}  {name}\n".encode())
    bundle = {
        "schema_version": "q2-v4.1-presemantic-bundle-hashes-v1",
        "created_at": "2026-08-26",
        "source_commit": git_head(),
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "artifacts": artifacts,
        "raw_a2_archive_aggregate_sha256": raw["aggregate_sha256"],
        "bundle_sha256": digest.hexdigest(),
    }
    write_json(REVIEW / "ARTIFACT_HASHES_FINAL.json", bundle)
    closeout = {
        "schema_version": "q2-v4.1-presemantic-closeout-v1",
        "classification": "Q2_V4_1_READY_FOR_PRINCIPAL_SEMANTIC_EXECUTION_REVIEW",
        "historical_v4_classification": "Q2_V4_SAFE_BANK_INSUFFICIENT",
        "historical_v4_forensic_classification": "Q2_V4_PRESEMANTIC_FORENSIC_CLEAN",
        "v4_1_design_classification": "Q2_V4_1_31_SAFE_BANK_ADEQUATE",
        "label_free_a1": a1["classification"],
        "label_free_a2": a2["classification"],
        "label_free_forensic": audit["classification"],
        "controller_count": 31,
        "future_conditions": 63,
        "future_panel_n": 300,
        "future_semantic_trajectories": 37800,
        "qap_maps": 50000,
        "environment_profile": EXPECTED_PROFILE,
        "model_revision": EXPECTED_MODEL_REVISION,
        "raw_a2_file_count": raw["file_count"],
        "raw_a2_aggregate_sha256": raw["aggregate_sha256"],
        "consolidator_commit": lock["label_free_offline_consolidation"]["consolidator_commit"],
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "q2": "UNTESTED",
        "q3": "NOT_RUN",
        "next_action": "PRINCIPAL_RESEARCHER_REVIEW",
        "runpod": "NOT_USED",
        "spark2": "NOT_USED",
        "local_dstack": "TERMINATED_NOT_PART_OF_PROTOCOL",
        "bundle_sha256": bundle["bundle_sha256"],
    }
    write_json(REVIEW / "Q2_V4_1_PRESEMANTIC_CLOSEOUT.json", closeout)
    (REVIEW / "Q2_V4_1_PRESEMANTIC_CLOSEOUT.md").write_text(
        "# Q2 V4.1 — final presemantic freeze and prediction lock closeout\n\n"
        "## Decision\n\n"
        "`Q2_V4_1_READY_FOR_PRINCIPAL_SEMANTIC_EXECUTION_REVIEW`\n\n"
        "The historical V4 result remains `Q2_V4_SAFE_BANK_INSUFFICIENT` and "
        "`Q2_V4_PRESEMANTIC_FORENSIC_CLEAN`. V4.1 retains all 31 directions "
        "that passed both original safety shells, in original order. The Q2 "
        "relational hypothesis remains untested.\n\n"
        "## Label-free qualification\n\n"
        f"A1: `{a1['classification']}`; fit hash `{a1['fit_hash']}`; effective "
        f"rank `{a1['effective_rank']}`; condition `{a1['condition_number']}`.\n\n"
        f"A2: `{a2['classification']}` for MEDIUM and STRONG. MEDIUM minimum "
        f"Gram eigenvalue `{a2['shells']['MEDIUM']['gram_min_eigenvalue']}`; "
        f"STRONG minimum Gram eigenvalue `{a2['shells']['STRONG']['gram_min_eigenvalue']}`. "
        "Both repeat archives are byte-identical to the raw archives and every "
        "frozen algebraic check passes.\n\n"
        "A2 uses natural-log full-vocabulary JS, `0.5 KL(p||m) + 0.5 KL(q||m)`, "
        "and an equal-weight mean over 48 probe/checkpoint rows. An independent "
        "reference check on one probe passed before accepting the complete A2 "
        "metrics.\n\n"
        "## Frozen future experiment\n\n"
        "The future panel is N=300 with 63 conditions (31 MEDIUM, 31 STRONG, "
        "and baseline), two rollouts, and 37,800 prospective semantic rows. "
        "The 50,000-map controller-label QAP and maxT structure are frozen. "
        "A1/A2/D2 are now materialized, but semantic execution is not authorized "
        "by this closeout.\n\n"
        "## Provenance and firewall\n\n"
        f"Spark 1 fingerprint: `{EXPECTED_PROFILE}`; model revision: "
        f"`{EXPECTED_MODEL_REVISION}`; access: direct SSH to Spark 1. The local "
        "dstack troubleshooting server was terminated and is not protocol "
        "infrastructure. Spark 2 and RunPod were not used.\n\n"
        f"Raw A2 files: `{raw['file_count']}/24`; aggregate SHA-256: "
        f"`{raw['aggregate_sha256']}`. Bundle SHA-256: `{bundle['bundle_sha256']}`.\n\n"
        "New semantic outcomes: `0`; correctness inspected: `NO`; Q2: "
        "`UNTESTED`; Q3: `NOT_RUN`.\n\n"
        "Status: `DRAFT / AWAITING PRINCIPAL SEMANTIC EXECUTION REVIEW`.\n",
        encoding="utf-8",
    )
    (REVIEW / "V4_1_SEMANTIC_EXECUTION_DRAFT.md").write_text(
        "# V4.1 semantic execution draft — not executed\n\n"
        "This document is a future execution draft only. It is not a new lock "
        "and it authorizes no collection.\n\n"
        "- Bank: all 31 immutable V4.1-safe directions, original order.\n"
        "- Conditions: baseline plus all 31 MEDIUM and 31 STRONG conditions.\n"
        "- Panel: the frozen N=300 `SEMANTIC_PANEL_MANIFEST.json`.\n"
        "- Schedule: the frozen 37,800-row `FUTURE_SEMANTIC_SCHEDULE.json`.\n"
        "- Endpoint: frozen `Dshape`, with the N/(N-1) superpopulation correction.\n"
        "- Metrics: A0, A1, A2, D2; controller-label QAP with 50,000 maps and "
        "maxT; item-cluster inference as locked.\n"
        "- Semantic execution: forbidden until a separate principal-reviewed "
        "execution freeze.\n\n"
        "No controller, item, dose, threshold, parser, geometry, or statistical "
        "rule may be changed in that future freeze.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": closeout["classification"], "bundle_sha256": bundle["bundle_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
