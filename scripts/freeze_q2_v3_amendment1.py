#!/usr/bin/env python3
"""Create the CPU-only Q2 V3 Freeze Amendment 1 artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_v3 import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    EXECUTION_TEACHER_TEXT,
    SOURCE_FAMILIES,
    ordered_id_hash,
)
from epistemic_geometry.experiments.q2_v3_prompt_provenance import (  # noqa: E402
    AMENDMENT1_PROVENANCE_SCHEMA,
    GATE7_CANONICAL_TEMPLATE_VERSION,
    LEGACY_HASH_SCHEMA,
    RAW_HASH_SCHEMA,
    amendment1_contract,
    canonical_q2_v3_task_prompt,
    legacy_external_prompt_digest,
    legacy_task_prompt,
    raw_utf8_sha256,
)

ORIGINAL = ROOT / "review/q2_v3_radial_angular_freeze"
RECONCILIATION = ROOT / "review/q2_v3_provenance_reconciliation"
OUTPUT = ROOT / "review/q2_v3_amendment1_freeze"
ORIGINAL_FREEZE_HEAD = "c9292d2baecb41de786912b77c39734855ed46cb"
FAILED_EXECUTION_HEAD = "9b1bc16ea6893ed798575a6850f6db602532ef69"
RECONCILIATION_HEAD = "c226adadeab04e296135349150a6fcbfd35af2b7"
ORIGINAL_SOURCE_COMMIT = "9a748de3706a788f8c6c5a1d12c09489808006e8"
EXPECTED_PRIMARY_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)
PRINCIPAL_DECISION = "CANONICALIZE_Q2_V3_ON_GATE7_PROMPT_TEMPLATE"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows() -> dict[str, dict[str, Any]]:
    path = RECONCILIATION / "OFFICIAL_SOURCE_RECORDS.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["id"]): row for row in rows}
    if len(rows) != 336 or len(result) != 336:
        raise RuntimeError("Amendment 1 requires exactly 336 unique source records")
    return result


def old_prompt_schema(row: dict[str, Any], source: dict[str, Any]) -> str:
    current = canonical_q2_v3_task_prompt(str(source["code"]), str(source["input"]))
    legacy = legacy_task_prompt(str(source["code"]), str(source["input"]))
    frozen = str(row["prompt_sha256"])
    if frozen == raw_utf8_sha256(current):
        return f"{RAW_HASH_SCHEMA}:{GATE7_CANONICAL_TEMPLATE_VERSION}"
    if frozen == legacy_external_prompt_digest(legacy):
        return f"{LEGACY_HASH_SCHEMA}:external-benchmark-cruxeval-prompt-v1"
    raise RuntimeError(f"unresolved historical prompt schema for {row['item_id']}")


def amended_manifests(
    source: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_records: list[dict[str, Any]] = []
    purpose_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for filename in MANIFESTS:
        original = read_json(ORIGINAL / filename)
        amended_items: list[dict[str, Any]] = []
        for old in original["items"]:
            item_id = str(old["item_id"])
            if item_id in seen:
                raise RuntimeError(f"duplicate Q2 V3 allocation ID: {item_id}")
            seen.add(item_id)
            public = source[item_id]
            schema = old_prompt_schema(old, public)
            contract = amendment1_contract(
                item_id=item_id,
                purpose=str(old["allocation"]),
                code=str(public["code"]),
                value=str(public["input"]),
                reference=str(public["output"]),
                official_index=int(old["official_index"]),
                dataset_repo=DATASET_REPO,
                dataset_revision=DATASET_REVISION,
                historical_prompt_hash=str(old["prompt_sha256"]),
                historical_prompt_schema=schema,
            )
            if contract["reference"]["utf8_sha256"] != old["reference_sha256"]:
                raise RuntimeError(f"reference changed for {item_id}")
            amended = {key: value for key, value in old.items() if key != "prompt_sha256"}
            amended["prompt_provenance"] = contract
            amended_items.append(amended)
            all_records.append({**contract, "manifest": filename})
            purpose_counts[str(old["allocation"])] += 1
        amended_manifest = {
            **{
                key: value
                for key, value in original.items()
                if key not in {"items", "schema_version"}
            },
            "schema_version": "q2-v3-amendment1-allocation-manifest-v1",
            "amendment": 1,
            "canonical_prompt_template": GATE7_CANONICAL_TEMPLATE_VERSION,
            "provenance_schema_version": AMENDMENT1_PROVENANCE_SCHEMA,
            "items": amended_items,
        }
        write_json(OUTPUT / filename, amended_manifest)
    if len(all_records) != 336 or len(seen) != 336:
        raise RuntimeError("Amendment 1 prompt coverage must be 336/336")
    write_json(
        OUTPUT / "AMENDED_PROMPT_MANIFEST.json",
        {
            "schema_version": "q2-v3-amendment1-complete-prompt-manifest-v1",
            "amendment": 1,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "canonical_prompt_template": GATE7_CANONICAL_TEMPLATE_VERSION,
            "provenance_schema_version": AMENDMENT1_PROVENANCE_SCHEMA,
            "record_count": len(all_records),
            "purpose_counts": dict(sorted(purpose_counts.items())),
            "records": all_records,
        },
    )
    return all_records, dict(sorted(purpose_counts.items()))


def byte_lock(text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {
        "encoding": "UTF-8",
        "unicode_normalization": "NONE",
        "newline_convention": "LF",
        "utf8_base64": base64.b64encode(raw).decode("ascii"),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_semantic_locks() -> None:
    fixture_code = "def f(x):\n    return x"
    fixture_input = "1"
    fixture = canonical_q2_v3_task_prompt(fixture_code, fixture_input)
    write_json(
        OUTPUT / "CANONICAL_GATE7_PROMPT_SPEC.json",
        {
            "schema_version": "q2-v3-gate7-prompt-spec-v1",
            "template_version": GATE7_CANONICAL_TEMPLATE_VERSION,
            "authoritative_constructor": (
                "epistemic_geometry.experiments.q2_v3_prompt_provenance."
                "canonical_q2_v3_task_prompt"
            ),
            "decision": PRINCIPAL_DECISION,
            "template_segments": [
                "Solve this Python code-output prediction problem.\\n\\n",
                "Function:\\n",
                "```python\\n{code}\\n```\\n\\n",
                "Input: {input}\\n\\n",
                "Return exactly one final line in this form:\\n",
                "FINAL: <the exact Python output>\\n",
                "Do not add any text after FINAL.",
            ],
            "fixture": {
                "code": fixture_code,
                "input": fixture_input,
                "exact_text": fixture,
                **byte_lock(fixture),
            },
        },
    )
    instructions = []
    for family in SOURCE_FAMILIES:
        for polarity, text in (
            ("POSITIVE", family.positive_instruction),
            ("NEGATIVE", family.negative_instruction),
        ):
            instructions.append(
                {
                    "family_id": family.family_id,
                    "polarity": polarity,
                    "exact_text": text,
                    **byte_lock(text),
                }
            )
    write_json(
        OUTPUT / "TECHNICAL_MODEL_INPUT_LOCK.json",
        {
            "schema_version": "q2-v3-amendment1-technical-model-input-lock-v1",
            "source_system_instructions": instructions,
            "execution_teacher_continuation": {
                "exact_text": EXECUTION_TEACHER_TEXT,
                **byte_lock(EXECUTION_TEACHER_TEXT),
            },
            "scientific_content_changed": False,
        },
    )


def copy_inherited_artifacts(original_lock: dict[str, Any]) -> list[str]:
    inherited = [name for name in original_lock["artifact_hashes"] if name not in MANIFESTS]
    for name in inherited:
        source = ORIGINAL / name
        if file_sha256(source) != original_lock["artifact_hashes"][name]:
            raise RuntimeError(f"original frozen artifact changed: {name}")
        shutil.copy2(source, OUTPUT / name)
    return inherited


def write_decision_and_specs(source_commit: str) -> None:
    decision = {
        "schema_version": "q2-v3-amendment1-principal-decision-v1",
        "decision": PRINCIPAL_DECISION,
        "decision_timing": "PROSPECTIVE_AFTER_ZERO_OUTCOME_CLEAN_ABORT",
        "outcome_independent": True,
        "canonical_template_version": GATE7_CANONICAL_TEMPLATE_VERSION,
        "legacy_template_executable": False,
        "rationale": [
            "327 of 336 frozen records already used the Gate-7 prompt",
            "the nine legacy records were historical residues, not planned conditions",
            "Q2 V3 does not study prompt-template robustness",
            "the scientific variable is causal-controller geometry",
            "zero Q2 V3 scientific outcomes existed at the decision",
        ],
        "history": {
            "original_freeze_head": ORIGINAL_FREEZE_HEAD,
            "failed_execution_head": FAILED_EXECUTION_HEAD,
            "reconciliation_head": RECONCILIATION_HEAD,
            "amendment1_source_commit": source_commit,
        },
    }
    write_json(OUTPUT / "PRINCIPAL_RESEARCHER_DECISION.json", decision)
    write_json(
        OUTPUT / "PROMPT_PROVENANCE_SCHEMA.json",
        {
            "schema_version": AMENDMENT1_PROVENANCE_SCHEMA,
            "required_fields": [
                "item_id",
                "purpose",
                "template_version",
                "provenance_schema_version",
                "model_visible_prompt.utf8_base64",
                "model_visible_prompt.prompt_bytes_sha256",
                "reference.identity",
                "reference.utf8_sha256",
                "source_artifact.dataset_repo",
                "source_artifact.dataset_revision",
                "source_artifact.source_record_sha256",
                "provenance_digest_sha256",
            ],
            "digest_binding": [
                "provenance_schema_version",
                "template_version",
                "purpose",
                "item_id",
                "exact_model_visible_utf8_base64",
            ],
            "ambiguous_prompt_hash_field_forbidden": True,
            "legacy_fields_are_historical_only": True,
        },
    )


def verify_zero_outcome() -> dict[str, Any]:
    failure = read_json(ORIGINAL / "Q2_V3_PANEL_PROVENANCE_FAILURE.json")
    forensic = read_json(ORIGINAL / "Q2_V3_FORENSIC_AUDIT.json")
    forbidden = [
        "Q2_V3_SOURCE_JOURNAL.jsonl",
        "Q2_V3_SHELL_JOURNAL.jsonl",
        "Q2_V3_SEMANTIC_JOURNAL.jsonl",
        "Q2_V3_PREDICTION_MATRICES.npz",
        "Q2_V3_PREDICTION_MATRICES.json",
        "Q2_V3_PREDICTION_LOCK.json",
        "Q2_V3_SOURCE_QUALIFICATION.json",
        "Q2_V3_SHELL_SAFETY.json",
        "Q2_V3_G0_RESULT.json",
        "Q2_V3_G1_RESULT.json",
        "Q2_V3_G2_RESULT.json",
        "Q2_V3_G3_RESULT.json",
    ]
    present = [name for name in forbidden if (ORIGINAL / name).exists()]
    result = {
        "schema_version": "q2-v3-amendment1-zero-outcome-verification-v1",
        "failed_execution_classification": failure["classification"],
        "forensic_classification": forensic["classification"],
        "scientific_trajectories": 0,
        "semantic_outcomes_opened": False,
        "prediction_matrices": "NONE",
        "controller_blind_spot_results": "NONE",
        "G0_G1_G2_G3_results": "NONE",
        "forbidden_artifacts_present": present,
        "pass": not present,
    }
    if not result["pass"]:
        raise RuntimeError(f"zero-outcome firewall failed: {present}")
    write_json(OUTPUT / "ZERO_OUTCOME_VERIFICATION.json", result)
    return result


def scientific_diff(original_lock: dict[str, Any], purpose_counts: dict[str, int]) -> None:
    invariant_keys = (
        "scientific_question",
        "evidence_class",
        "bank",
        "geometries",
        "model",
        "panel",
        "semantic_panel",
        "M3",
        "q3",
    )
    checks = {key: original_lock[key] for key in invariant_keys}
    panel = read_json(OUTPUT / "PRIMARY_PANEL_MANIFEST.json")
    if panel["item_count"] != 200 or ordered_id_hash(panel["item_ids"]) != EXPECTED_PRIMARY_HASH:
        raise RuntimeError("Q2_V3_AMENDMENT1_PANEL_IDENTITY_CHANGED")
    write_json(
        OUTPUT / "SCIENTIFIC_DIFF_AUDIT.json",
        {
            "schema_version": "q2-v3-amendment1-scientific-diff-v1",
            "classification": "Q2_V3_AMENDMENT1_EXPECTED_PROMPT_ONLY_DIFF",
            "allowed_changes": {
                "prompt_template": {
                    "before": "ACCIDENTAL_MIXED_GATE7_327_LEGACY_9",
                    "after": GATE7_CANONICAL_TEMPLATE_VERSION,
                },
                "prompt_serialization_contract": "VERSIONED_EXACT_UTF8_BYTES",
                "prompt_provenance_schema": AMENDMENT1_PROVENANCE_SCHEMA,
                "dependent_prompt_manifest_hashes": "RECOMPUTED",
            },
            "other_scientific_changes": [],
            "original_lock_invariants": checks,
            "purpose_counts": purpose_counts,
            "primary_panel": {
                "n": 200,
                "ordered_ids_sha256": EXPECTED_PRIMARY_HASH,
                "unchanged": True,
            },
        },
    )


def write_markdown() -> None:
    (OUTPUT / "PRINCIPAL_RESEARCHER_DECISION.md").write_text(
        "# Q2 V3 principal-researcher prompt decision\n\n"
        "The canonical model-visible constructor for every Q2 V3 purpose is "
        "`gate7-cruxeval-semantic-task-prompt-v1`. The legacy prompt is historical "
        "provenance only and is not an executable alternative. This prospective, "
        "outcome-independent decision follows a clean zero-inference abort.\n",
        encoding="utf-8",
    )
    (OUTPUT / "CANONICAL_GATE7_PROMPT_SPEC.md").write_text(
        "# Canonical Gate-7 prompt specification\n\n"
        "Authoritative constructor: `canonical_q2_v3_task_prompt`. Encoding is UTF-8, "
        "newlines are LF, and no Unicode normalization is applied. Literal template:\n\n"
        "```text\nSolve this Python code-output prediction problem.\n\nFunction:\n"
        "```python\n{code}\n```\n\nInput: {input}\n\nReturn exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\nDo not add any text after FINAL.\n```\n\n"
        "The JSON companion locks a concrete exact-byte fixture.\n",
        encoding="utf-8",
    )
    (OUTPUT / "PROMPT_PROVENANCE_SCHEMA.md").write_text(
        "# Amendment-1 prompt provenance schema\n\n"
        "Every record separately types item, purpose, template, schema, exact UTF-8 "
        "bytes, raw byte hash, reference, source artifact, and a provenance digest. "
        "Legacy hashes remain historical-only fields. An ambiguous `prompt_hash` field "
        "is forbidden in executable Amendment-1 manifests.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_lock = read_json(ORIGINAL / "PROTOCOL_LOCK.json")
    inherited = copy_inherited_artifacts(original_lock)
    write_decision_and_specs(args.source_commit)
    write_semantic_locks()
    records, counts = amended_manifests(source_rows())
    scientific_diff(original_lock, counts)
    zero = verify_zero_outcome()
    write_markdown()

    frozen_names = sorted(
        [*inherited, *MANIFESTS]
        + [
            "AMENDED_PROMPT_MANIFEST.json",
            "CANONICAL_GATE7_PROMPT_SPEC.json",
            "CANONICAL_GATE7_PROMPT_SPEC.md",
            "PRINCIPAL_RESEARCHER_DECISION.json",
            "PRINCIPAL_RESEARCHER_DECISION.md",
            "PROMPT_PROVENANCE_SCHEMA.json",
            "PROMPT_PROVENANCE_SCHEMA.md",
            "SCIENTIFIC_DIFF_AUDIT.json",
            "TECHNICAL_MODEL_INPUT_LOCK.json",
            "ZERO_OUTCOME_VERIFICATION.json",
        ]
    )
    artifact_hashes = {name: file_sha256(OUTPUT / name) for name in frozen_names}
    lock = {
        **{key: value for key, value in original_lock.items() if key != "artifact_hashes"},
        "schema_version": "q2-v3-amendment1-prospective-lock-v1",
        "status": "Q2_V3_AMENDMENT1_FROZEN_NOT_RUN",
        "execution_authorized": False,
        "experiment_source_commit": args.source_commit,
        "original_experiment_source_commit": ORIGINAL_SOURCE_COMMIT,
        "amendment": {
            "number": 1,
            "decision": PRINCIPAL_DECISION,
            "canonical_prompt_template": GATE7_CANONICAL_TEMPLATE_VERSION,
            "prompt_provenance_schema": AMENDMENT1_PROVENANCE_SCHEMA,
            "prompt_records": len(records),
            "legacy_executable_records": 0,
            "other_scientific_changes": [],
        },
        "history": {
            "original_freeze_head": ORIGINAL_FREEZE_HEAD,
            "failed_execution_head": FAILED_EXECUTION_HEAD,
            "failed_execution_state": "Q2_V3_PANEL_PROVENANCE_MISMATCH",
            "reconciliation_head": RECONCILIATION_HEAD,
            "reconciliation_state": "Q2_V3_REFREEZE_REQUIRES_PRINCIPAL_RESEARCHER_DECISION",
        },
        "artifact_hashes": artifact_hashes,
    }
    write_json(OUTPUT / "PROTOCOL_LOCK.json", lock)
    (OUTPUT / "PROTOCOL_LOCK.md").write_text(
        "# Q2 V3 Freeze Amendment 1\n\n"
        "Status: `Q2_V3_AMENDMENT1_FROZEN_NOT_RUN`\n\n"
        "The principal researcher prospectively selected the Gate-7 prompt template "
        "globally after the original clean pre-inference abort and P2 reconciliation. "
        "All 336 records now use one exact-byte constructor and one typed provenance "
        "schema. No other scientific element changed. Execution remains unauthorized.\n\n"
        f"Experiment source commit: `{args.source_commit}`.\n\n"
        f"Zero-outcome verification: `{'PASS' if zero['pass'] else 'FAIL'}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(records), "purpose_counts": counts, "status": lock["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
