#!/usr/bin/env python3
"""Independent CPU-only audit of Q2 V3 Freeze Amendment 1."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "review/q2_v3_radial_angular_freeze"
RECONCILIATION = ROOT / "review/q2_v3_provenance_reconciliation"
AMENDMENT = ROOT / "review/q2_v3_amendment1_freeze"
EXPECTED_PRIMARY_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"
EXPECTED_TEMPLATE = "gate7-cruxeval-semantic-task-prompt-v1"
EXPECTED_SCHEMA = "q2-v3-amendment1-prompt-provenance-v1"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_prompt(code: str, value: str) -> bytes:
    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    ).encode()


def independent_ordered_id_hash(ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def independent_source_hash(source: dict[str, Any]) -> str:
    payload = {
        "schema_version": "q2-v3-cruxeval-source-record-v1",
        "item_id": str(source["id"]),
        "code": str(source["code"]),
        "input": str(source["input"]),
        "output": str(source["output"]),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def independent_contract_hash(record: dict[str, Any]) -> str:
    payload = {
        "provenance_schema_version": record["provenance_schema_version"],
        "template_version": record["template_version"],
        "purpose": record["purpose"],
        "item_id": record["item_id"],
        "exact_model_visible_utf8_base64": record["model_visible_prompt"]["utf8_base64"],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    failures: list[str] = []
    lock = read_json(AMENDMENT / "PROTOCOL_LOCK.json")
    original_lock = read_json(ORIGINAL / "PROTOCOL_LOCK.json")
    source_rows = [
        json.loads(line)
        for line in (RECONCILIATION / "OFFICIAL_SOURCE_RECORDS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    source = {str(row["id"]): row for row in source_rows}
    if len(source_rows) != 336 or len(source) != 336:
        failures.append("source subset count")
    for name, expected in lock["artifact_hashes"].items():
        if not (AMENDMENT / name).is_file() or file_hash(AMENDMENT / name) != expected:
            failures.append(f"lock artifact hash: {name}")

    purpose_counts: Counter[str] = Counter()
    seen: set[str] = set()
    prompt_mismatches = 0
    lock_mismatches = 0
    legacy_executable = 0
    for filename in MANIFESTS:
        amended = read_json(AMENDMENT / filename)
        original = read_json(ORIGINAL / filename)
        if amended["item_ids"] != original["item_ids"]:
            failures.append(f"item IDs/order changed: {filename}")
        if amended["ordered_ids_sha256"] != original["ordered_ids_sha256"]:
            failures.append(f"ordered hash changed: {filename}")
        for amended_row, original_row in zip(amended["items"], original["items"], strict=True):
            item_id = str(amended_row["item_id"])
            if item_id in seen:
                failures.append(f"duplicate: {item_id}")
            seen.add(item_id)
            if {key: value for key, value in amended_row.items() if key != "prompt_provenance"} != {
                key: value for key, value in original_row.items() if key != "prompt_sha256"
            }:
                failures.append(f"non-prompt row diff: {item_id}")
            record = amended_row["prompt_provenance"]
            public = source[item_id]
            exact = independent_prompt(str(public["code"]), str(public["input"]))
            locked = base64.b64decode(record["model_visible_prompt"]["utf8_base64"], validate=True)
            if locked != exact:
                prompt_mismatches += 1
            checks = (
                record["template_version"] == EXPECTED_TEMPLATE,
                record["provenance_schema_version"] == EXPECTED_SCHEMA,
                record["purpose"] == amended_row["allocation"],
                record["model_visible_prompt"]["prompt_bytes_sha256"]
                == hashlib.sha256(exact).hexdigest(),
                record["reference"]["utf8_sha256"]
                == hashlib.sha256(str(public["output"]).encode("utf-8")).hexdigest(),
                record["source_artifact"]["source_record_sha256"]
                == independent_source_hash(public),
                record["provenance_digest_sha256"] == independent_contract_hash(record),
            )
            if not all(checks):
                lock_mismatches += 1
            if record["template_version"] != EXPECTED_TEMPLATE:
                legacy_executable += 1
            purpose_counts[str(amended_row["allocation"])] += 1

    primary = read_json(AMENDMENT / "PRIMARY_PANEL_MANIFEST.json")
    primary_hash = independent_ordered_id_hash([str(value) for value in primary["item_ids"]])
    if primary["item_count"] != 200 or primary_hash != EXPECTED_PRIMARY_HASH:
        failures.append("primary panel identity")
    if len(seen) != 336 or seen != set(source):
        failures.append("complete identity coverage")
    aggregate = read_json(AMENDMENT / "AMENDED_PROMPT_MANIFEST.json")
    if aggregate["record_count"] != 336 or len(aggregate["records"]) != 336:
        failures.append("aggregate count")
    if prompt_mismatches:
        failures.append(f"prompt mismatches: {prompt_mismatches}")
    if lock_mismatches:
        failures.append(f"prompt lock mismatches: {lock_mismatches}")
    if legacy_executable:
        failures.append(f"legacy executable records: {legacy_executable}")

    unchanged = [
        name
        for name in original_lock["artifact_hashes"]
        if name not in MANIFESTS
    ]
    inherited_mismatches = []
    for name in unchanged:
        relative = f"review/q2_v3_radial_angular_freeze/{name}"
        if (
            file_hash(ORIGINAL / name) != original_lock["artifact_hashes"][name]
            or lock["inherited_artifact_hashes"].get(relative)
            != original_lock["artifact_hashes"][name]
        ):
            inherited_mismatches.append(name)
    if inherited_mismatches:
        failures.append(f"inherited artifact differences: {inherited_mismatches}")
    diff = read_json(AMENDMENT / "SCIENTIFIC_DIFF_AUDIT.json")
    if diff["other_scientific_changes"]:
        failures.append("unexpected scientific diff")
    zero = read_json(AMENDMENT / "ZERO_OUTCOME_VERIFICATION.json")
    if not zero["pass"] or zero["scientific_trajectories"] != 0:
        failures.append("zero-outcome firewall")
    if lock["status"] != "Q2_V3_AMENDMENT1_FROZEN_NOT_RUN":
        failures.append("terminal state")
    if lock["execution_authorized"] is not False:
        failures.append("execution authorization")

    result = {
        "schema_version": "q2-v3-amendment1-independent-freeze-audit-v1",
        "classification": (
            "Q2_V3_AMENDMENT1_FREEZE_AUDIT_PASS"
            if not failures
            else "Q2_V3_AMENDMENT1_FREEZE_AUDIT_FAIL"
        ),
        "records_checked": len(seen),
        "prompt_byte_mismatches": prompt_mismatches,
        "prompt_lock_mismatches": lock_mismatches,
        "legacy_executable_records": legacy_executable,
        "unresolved_provenance_records": 0 if not failures else None,
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "primary_n": primary["item_count"],
        "primary_ordered_ids_sha256": primary_hash,
        "inherited_scientific_artifact_mismatches": inherited_mismatches,
        "semantic_trajectories": zero["scientific_trajectories"],
        "prediction_matrices": zero["prediction_matrices"],
        "failures": failures,
        "pass": not failures,
    }
    (AMENDMENT / "INDEPENDENT_FREEZE_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (AMENDMENT / "INDEPENDENT_FREEZE_AUDIT.md").write_text(
        "# Q2 V3 Amendment-1 independent freeze audit\n\n"
        f"Classification: `{result['classification']}`\n\n"
        f"Records independently reconstructed: {len(seen)}/336. "
        f"Prompt-byte mismatches: {prompt_mismatches}. Prompt-lock mismatches: "
        f"{lock_mismatches}. Legacy executable records: {legacy_executable}.\n\n"
        f"Primary panel: N={primary['item_count']}, ordered-ID SHA-256 `{primary_hash}`.\n\n"
        "No semantic trajectories or prediction matrices exist. Execution remains unauthorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
