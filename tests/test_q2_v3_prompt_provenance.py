from __future__ import annotations

import hashlib
import json
from pathlib import Path

from epistemic_geometry.experiments.q2_v3 import ordered_id_hash
from epistemic_geometry.experiments.q2_v3_prompt_provenance import (
    AMENDMENT1_PROVENANCE_SCHEMA,
    CURRENT_TEMPLATE_VERSION,
    GATE7_CANONICAL_TEMPLATE_VERSION,
    LEGACY_HASH_SCHEMA,
    LEGACY_TEMPLATE_VERSION,
    PROPOSED_CONTRACT_SCHEMA,
    RAW_HASH_SCHEMA,
    amendment1_contract,
    canonical_contract,
    canonical_q2_v3_task_prompt,
    current_task_prompt,
    decode_contract_prompt,
    legacy_external_prompt_digest,
    legacy_task_prompt,
    raw_utf8_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "review/q2_v3_radial_angular_freeze"
RECONCILIATION = ROOT / "review/q2_v3_provenance_reconciliation"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)
EXPECTED_PRIMARY_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"


def _source() -> dict[str, dict[str, object]]:
    rows = [
        json.loads(line)
        for line in (RECONCILIATION / "OFFICIAL_SOURCE_RECORDS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    return {str(row["id"]): row for row in rows}


def _manifest_rows() -> list[dict[str, object]]:
    rows = []
    for filename in MANIFESTS:
        payload = json.loads((ORIGINAL / filename).read_text(encoding="utf-8"))
        rows.extend(payload["items"])
    return rows


def test_every_q2_v3_item_prompt_hash_has_one_explicit_historical_schema() -> None:
    source = _source()
    rows = _manifest_rows()
    counts = {"current_raw": 0, "legacy_namespaced": 0}
    assert len(rows) == len(source) == 336
    for row in rows:
        item_id = str(row["item_id"])
        public = source[item_id]
        current = current_task_prompt(str(public["code"]), str(public["input"]))
        legacy = legacy_task_prompt(str(public["code"]), str(public["input"]))
        frozen = str(row["prompt_sha256"])
        matches = {
            "current_raw": frozen == raw_utf8_sha256(current),
            "legacy_namespaced": frozen == legacy_external_prompt_digest(legacy),
        }
        assert sum(matches.values()) == 1, item_id
        counts[next(key for key, value in matches.items() if value)] += 1
        assert str(row["reference_sha256"]) == raw_utf8_sha256(str(public["output"]))
    assert counts == {"current_raw": 327, "legacy_namespaced": 9}


def test_hash_schema_versions_cannot_be_silently_equated() -> None:
    prompt = "FINAL: 1"
    assert RAW_HASH_SCHEMA != LEGACY_HASH_SCHEMA
    assert raw_utf8_sha256(prompt) != legacy_external_prompt_digest(prompt)
    assert CURRENT_TEMPLATE_VERSION != LEGACY_TEMPLATE_VERSION


def test_candidate_contract_round_trips_exact_bytes_and_purpose() -> None:
    contract = canonical_contract(
        item_id="sample_fixture",
        purpose="M2_LABEL_FREE_PROBES",
        prompt="a\r\nβ\n",
        prompt_template_version=CURRENT_TEMPLATE_VERSION,
        reference="1",
        dataset_repo="fixture/repo",
        dataset_revision="fixture-revision",
        system_prompt="system β",
    )
    user, system = decode_contract_prompt(contract)
    assert user == "a\r\nβ\n".encode()
    assert system == "system β".encode()
    assert contract["schema_version"] == PROPOSED_CONTRACT_SCHEMA
    assert contract["purpose"] == "M2_LABEL_FREE_PROBES"
    assert contract["user_prompt_bytes_sha256"] == hashlib.sha256(user).hexdigest()
    assert contract["rendering"]["rendered_prompt_bytes_sha256"] is None


def test_amendment1_contract_is_typed_and_locks_gate7_bytes() -> None:
    contract = amendment1_contract(
        item_id="sample_fixture",
        purpose="PRIMARY_SEMANTIC_PANEL",
        code="def f(x):\n    return x",
        value="1",
        reference="1",
        official_index=1,
        dataset_repo="fixture/repo",
        dataset_revision="fixture-revision",
        historical_prompt_hash="0" * 64,
        historical_prompt_schema="fixture-historical-v1",
    )
    expected = canonical_q2_v3_task_prompt("def f(x):\n    return x", "1").encode()
    assert contract["provenance_schema_version"] == AMENDMENT1_PROVENANCE_SCHEMA
    assert contract["template_version"] == GATE7_CANONICAL_TEMPLATE_VERSION
    assert contract["model_visible_prompt"]["prompt_bytes_sha256"] == hashlib.sha256(
        expected
    ).hexdigest()
    assert "prompt_sha256" not in contract


def test_q2_v3_execution_path_uses_only_authoritative_gate7_constructor() -> None:
    runner = (ROOT / "scripts/run_q2_v3.py").read_text(encoding="utf-8")
    assert "canonical_q2_v3_task_prompt" in runner
    assert "legacy_task_prompt" not in runner
    assert 'REVIEW = ROOT / "review/q2_v3_amendment1_freeze"' in runner


def test_q2_v3_prompt_purpose_coverage_and_primary_identity() -> None:
    schema = json.loads((RECONCILIATION / "PROMPT_SCHEMA_AUDIT.json").read_text())
    expected = {
        "SOURCE_CONSTRUCTION",
        "SOURCE_VALIDATION",
        "SHELL_CALIBRATION",
        "M1_COVARIANCE",
        "M2_LABEL_FREE_PROBES",
        "PRIMARY_SEMANTIC_PANEL",
    }
    assert set(schema["item_prompt_schema_template_counts_by_purpose"]) == expected
    assert schema["scope"]["technical_probe_new_prompt_count"] == 0
    panel = json.loads((ORIGINAL / "PRIMARY_PANEL_MANIFEST.json").read_text())
    assert panel["item_count"] == 200
    assert ordered_id_hash(panel["item_ids"]) == EXPECTED_PRIMARY_HASH


def test_reconciliation_decision_preserves_clean_abort_and_blocks_autorefresh() -> None:
    decision = json.loads((RECONCILIATION / "PROTOCOL_DECISION.json").read_text())
    comparison = json.loads((RECONCILIATION / "NINE_ITEM_BYTE_COMPARISON.json").read_text())
    assert decision["terminal_state"] == (
        "Q2_V3_REFREEZE_REQUIRES_PRINCIPAL_RESEARCHER_DECISION"
    )
    assert decision["amendment1_created"] is False
    assert decision["original_failed_execution"]["scientific_trajectories"] == 0
    assert comparison["classification_counts"] == {"P2": 9}
    assert all(not row["comparison"]["exact_bytes_equal"] for row in comparison["records"])
