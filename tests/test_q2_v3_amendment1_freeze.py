from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from epistemic_geometry.experiments.q2_v3 import ordered_id_hash
from epistemic_geometry.experiments.q2_v3_prompt_provenance import (
    AMENDMENT1_PROVENANCE_SCHEMA,
    GATE7_CANONICAL_TEMPLATE_VERSION,
    canonical_q2_v3_task_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "review/q2_v3_amendment1_freeze"
ORIGINAL = ROOT / "review/q2_v3_radial_angular_freeze"
EXPECTED_PRIMARY_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source() -> dict[str, dict[str, object]]:
    path = ROOT / "review/q2_v3_provenance_reconciliation/OFFICIAL_SOURCE_RECORDS.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {str(row["id"]): row for row in rows}


def test_amendment1_freezes_one_gate7_template_for_all_336_records() -> None:
    source = _source()
    seen: set[str] = set()
    for filename in MANIFESTS:
        manifest = _json(AMENDMENT / filename)
        assert manifest["canonical_prompt_template"] == GATE7_CANONICAL_TEMPLATE_VERSION
        assert manifest["provenance_schema_version"] == AMENDMENT1_PROVENANCE_SCHEMA
        assert manifest["outcome_values_read_or_used"] is False
        for row in manifest["items"]:  # type: ignore[index]
            item_id = str(row["item_id"])
            assert item_id not in seen
            seen.add(item_id)
            assert "prompt_sha256" not in row
            provenance = row["prompt_provenance"]
            assert provenance["template_version"] == GATE7_CANONICAL_TEMPLATE_VERSION
            assert provenance["provenance_schema_version"] == AMENDMENT1_PROVENANCE_SCHEMA
            public = source[item_id]
            expected = canonical_q2_v3_task_prompt(
                str(public["code"]), str(public["input"])
            ).encode()
            locked = base64.b64decode(
                provenance["model_visible_prompt"]["utf8_base64"], validate=True
            )
            assert locked == expected
            assert provenance["model_visible_prompt"]["prompt_bytes_sha256"] == hashlib.sha256(
                expected
            ).hexdigest()
    assert seen == set(source)
    assert len(seen) == 336


def test_amendment1_preserves_every_non_prompt_manifest_field() -> None:
    for filename in MANIFESTS:
        original = _json(ORIGINAL / filename)
        amended = _json(AMENDMENT / filename)
        assert amended["item_ids"] == original["item_ids"]
        assert amended["ordered_ids_sha256"] == original["ordered_ids_sha256"]
        for before, after in zip(original["items"], amended["items"], strict=True):  # type: ignore[index]
            before_without_prompt = {
                key: value for key, value in before.items() if key != "prompt_sha256"
            }
            after_without_prompt = {
                key: value for key, value in after.items() if key != "prompt_provenance"
            }
            assert after_without_prompt == before_without_prompt


def test_amendment1_primary_panel_identity_and_history_are_immutable() -> None:
    panel = _json(AMENDMENT / "PRIMARY_PANEL_MANIFEST.json")
    assert panel["item_count"] == 200
    assert ordered_id_hash(panel["item_ids"]) == EXPECTED_PRIMARY_HASH  # type: ignore[arg-type]
    lock = _json(AMENDMENT / "PROTOCOL_LOCK.json")
    assert lock["status"] == "Q2_V3_AMENDMENT1_FROZEN_NOT_RUN"
    assert lock["execution_authorized"] is False
    assert lock["history"]["original_freeze_head"] == (
        "c9292d2baecb41de786912b77c39734855ed46cb"
    )
    assert lock["history"]["failed_execution_state"] == "Q2_V3_PANEL_PROVENANCE_MISMATCH"
    assert lock["history"]["failed_execution_head"] == (
        "9b1bc16ea6893ed798575a6850f6db602532ef69"
    )


def test_amendment1_cpu_preflight_and_independent_audit_pass_without_mismatch() -> None:
    preflight = _json(AMENDMENT / "CPU_PREFLIGHT_REPORT.json")
    audit = _json(AMENDMENT / "INDEPENDENT_FREEZE_AUDIT.json")
    zero = _json(AMENDMENT / "ZERO_OUTCOME_VERIFICATION.json")
    assert preflight["classification"] == "Q2_V3_AMENDMENT1_CPU_PREFLIGHT_PASS"
    assert preflight["records_checked"] == 336
    assert preflight["mismatch_count"] == 0
    assert preflight["legacy_executable_records"] == 0
    assert audit["classification"] == "Q2_V3_AMENDMENT1_FREEZE_AUDIT_PASS"
    assert audit["prompt_byte_mismatches"] == 0
    assert audit["prompt_lock_mismatches"] == 0
    assert audit["legacy_executable_records"] == 0
    assert audit["unresolved_provenance_records"] == 0
    assert zero["pass"] is True
    assert zero["scientific_trajectories"] == 0
    assert zero["prediction_matrices"] == "NONE"


def test_original_mixed_freeze_and_clean_abort_remain_preserved() -> None:
    failure = _json(ORIGINAL / "Q2_V3_PANEL_PROVENANCE_FAILURE.json")
    forensic = _json(ORIGINAL / "Q2_V3_FORENSIC_AUDIT.json")
    assert failure["classification"] == "Q2_V3_PANEL_PROVENANCE_MISMATCH"
    assert len(failure["mismatches"]) == 9
    assert forensic["audit_classification"] == "Q2_V3_FORENSIC_CLEAN"
    assert forensic["model_inference_rows"] == 0
    assert forensic["semantic_outcomes_opened"] is False
    assert forensic["prediction_lock_created"] is False
