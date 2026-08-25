#!/usr/bin/env python3
"""Reconcile the aborted Q2 V3 prompt provenance without model inference."""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import subprocess
import sys
import unicodedata
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
    CURRENT_TEMPLATE_VERSION,
    LEGACY_HASH_SCHEMA,
    LEGACY_TEMPLATE_VERSION,
    RAW_HASH_SCHEMA,
    canonical_contract,
    current_task_prompt,
    legacy_external_prompt_digest,
    legacy_task_prompt,
    raw_utf8_sha256,
)

ORIGINAL = ROOT / "review/q2_v3_radial_angular_freeze"
OUTPUT = ROOT / "review/q2_v3_provenance_reconciliation"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)
NINE_IDS = (
    "sample_300",
    "sample_74",
    "sample_745",
    "sample_700",
    "sample_659",
    "sample_777",
    "sample_145",
    "sample_698",
    "sample_21",
)
EXPECTED_PRIMARY_HASH = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"
FAILED_HEAD = "9b1bc16ea6893ed798575a6850f6db602532ef69"
ORIGINAL_FREEZE_SOURCE = "9a748de3706a788f8c6c5a1d12c09489808006e8"
ORIGINAL_FREEZE_HEAD = "c9292d2baecb41de786912b77c39734855ed46cb"


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


def load_public(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["id"]): dict(row) for row in rows}
    if len(rows) != 800 or len(result) != 800:
        raise RuntimeError("official CRUXEval source must contain 800 unique rows")
    if set(result) != {f"sample_{index}" for index in range(800)}:
        raise RuntimeError("official CRUXEval identities are incomplete")
    return result


def historical_paths(item_id: str) -> list[str]:
    """List tracked containing artifacts without parsing any outcome field."""

    result = subprocess.run(
        ["git", "grep", "-l", "-F", item_id, "--", ":(exclude).git"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed for {item_id}: {result.stderr}")
    return sorted(line for line in result.stdout.splitlines() if line)


def historical_git_changes(item_id: str) -> list[dict[str, Any]]:
    """Inventory commits/files where the tracked occurrence count changed."""

    result = subprocess.run(
        [
            "git",
            "log",
            "--all",
            f"-S{item_id}",
            "--format=@@%H%x09%cs%x09%s",
            "--name-only",
            "--",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    changes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            if current is not None:
                current["paths"] = sorted(set(current["paths"]))
                changes.append(current)
            commit, date, subject = line[2:].split("\t", 2)
            current = {"commit": commit, "date": date, "subject": subject, "paths": []}
        elif line and current is not None:
            current["paths"].append(line)
    if current is not None:
        current["paths"] = sorted(set(current["paths"]))
        changes.append(current)
    return changes


def text_metadata(text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {
        "utf8_byte_count": len(raw),
        "utf8_hex": raw.hex(),
        "utf8_base64": base64.b64encode(raw).decode("ascii"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(text.splitlines()),
        "line_endings": {
            "LF": text.count("\n"),
            "CRLF": text.count("\r\n"),
            "bare_CR": text.count("\r") - text.count("\r\n"),
        },
        "leading_whitespace_bytes": len(raw) - len(raw.lstrip()),
        "trailing_whitespace_bytes": len(raw) - len(raw.rstrip()),
        "unicode_normalization": {
            form: unicodedata.normalize(form, text) == text
            for form in ("NFC", "NFD", "NFKC", "NFKD")
        },
        "prefix_80": text[:80],
        "suffix_120": text[-120:],
        "contains_code_fence": "```python" in text and "```" in text,
        "contains_reference_answer": False,
    }


def manifest_index() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for filename in MANIFESTS:
        manifest = read_json(ORIGINAL / filename)
        for row in manifest["items"]:
            record = {**row, "manifest": filename}
            if record["item_id"] in by_id:
                raise RuntimeError(f"Q2 V3 allocation overlap: {record['item_id']}")
            by_id[record["item_id"]] = record
            all_rows.append(record)
    if len(all_rows) != 336:
        raise RuntimeError("expected exactly 336 frozen prompt-bearing item records")
    return by_id, all_rows


def classify_hash(frozen: str, current: str, legacy: str) -> tuple[str, str]:
    candidates = {
        "CURRENT_TEMPLATE_RAW_SHA256": raw_utf8_sha256(current),
        "CURRENT_TEMPLATE_EXTERNAL_PROMPT": legacy_external_prompt_digest(current),
        "LEGACY_TEMPLATE_RAW_SHA256": raw_utf8_sha256(legacy),
        "LEGACY_TEMPLATE_EXTERNAL_PROMPT": legacy_external_prompt_digest(legacy),
    }
    matches = [name for name, value in candidates.items() if value == frozen]
    if len(matches) != 1:
        return "UNRESOLVED", ",".join(matches)
    return matches[0], candidates[matches[0]]


def nine_comparisons(
    public: dict[str, dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    failed = read_json(ORIGINAL / "Q2_V3_PANEL_PROVENANCE_FAILURE.json")
    failed_by_id = {row["item_id"]: row for row in failed["mismatches"]}
    records: list[dict[str, Any]] = []
    for item_id in NINE_IDS:
        source = public[item_id]
        historical = legacy_task_prompt(str(source["code"]), str(source["input"]))
        current = current_task_prompt(str(source["code"]), str(source["input"]))
        frozen = str(by_id[item_id]["prompt_sha256"])
        legacy_digest = legacy_external_prompt_digest(historical)
        current_digest = raw_utf8_sha256(current)
        if (
            frozen != legacy_digest
            or failed_by_id[item_id]["official_prompt_sha256"] != current_digest
        ):
            raise RuntimeError(f"failed record did not reproduce for {item_id}")
        records.append(
            {
                "item_id": item_id,
                "official_source_fields": {
                    "id": source["id"],
                    "code": source["code"],
                    "input": source["input"],
                    "output": source["output"],
                },
                "purpose": by_id[item_id]["allocation"],
                "manifest": by_id[item_id]["manifest"],
                "official_index": by_id[item_id]["official_index"],
                "reference_answer": str(source["output"]),
                "reference_sha256": raw_utf8_sha256(str(source["output"])),
                "reference_hash_matches_frozen": (
                    raw_utf8_sha256(str(source["output"])) == by_id[item_id]["reference_sha256"]
                ),
                "historical": {
                    "template_version": LEGACY_TEMPLATE_VERSION,
                    "raw_prompt_text": historical,
                    "bytes": text_metadata(historical),
                    "hash_schema": LEGACY_HASH_SCHEMA,
                    "hash_procedure": (
                        "SHA256(UTF8('EXTERNAL-PROMPT') || 0x1f || exact_prompt_utf8_bytes)"
                    ),
                    "hash_payload_utf8_hex": (
                        b"EXTERNAL-PROMPT\x1f" + historical.encode("utf-8")
                    ).hex(),
                    "namespaced_hash": legacy_digest,
                    "code_provenance": [
                        "scripts/prepare_external_benchmark.py::_cruxeval_prompt",
                        "src/epistemic_geometry/benchmarks/external/base.py::ExternalItem.prompt_hash",
                        "src/epistemic_geometry/reproducibility.py::stable_digest",
                    ],
                },
                "current": {
                    "template_version": CURRENT_TEMPLATE_VERSION,
                    "raw_prompt_text": current,
                    "bytes": text_metadata(current),
                    "hash_schema": RAW_HASH_SCHEMA,
                    "hash_procedure": "SHA256(exact_prompt_utf8_bytes)",
                    "raw_sha256": current_digest,
                    "code_provenance": [
                        "src/epistemic_geometry/experiments/gate7.py::task_prompt",
                        "scripts/run_q2_v3.py::_normalize_public",
                    ],
                },
                "comparison": {
                    "exact_bytes_equal": historical.encode("utf-8") == current.encode("utf-8"),
                    "same_code_field": True,
                    "same_input_field": True,
                    "same_reference_field": True,
                    "same_code_fence_payload": True,
                    "same_line_ending_kind": True,
                    "system_or_chat_wrapper_in_compared_field": False,
                    "reference_answer_present_in_either_prompt": False,
                    "unified_diff": list(
                        difflib.unified_diff(
                            historical.splitlines(),
                            current.splitlines(),
                            fromfile=LEGACY_TEMPLATE_VERSION,
                            tofile=CURRENT_TEMPLATE_VERSION,
                            lineterm="",
                        )
                    ),
                    "scientifically_plausible_wording_or_instruction_change": True,
                },
                "classification": "P2",
                "classification_reason": (
                    "Model-visible task wording and answer instructions differ; the change is "
                    "not limited to a namespace, whitespace, line ending, or inert envelope."
                ),
                "historical_artifacts_containing_item_id": historical_paths(item_id),
                "historical_git_changes_containing_item_id": historical_git_changes(item_id),
            }
        )
    return {
        "schema_version": "q2-v3-nine-item-byte-comparison-v1",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "classification_counts": dict(Counter(row["classification"] for row in records)),
        "records": records,
    }


def schema_audit(
    public: dict[str, dict[str, Any]], all_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    references_match = True
    for frozen in all_rows:
        source = public[str(frozen["item_id"])]
        current = current_task_prompt(str(source["code"]), str(source["input"]))
        legacy = legacy_task_prompt(str(source["code"]), str(source["input"]))
        schema, reproduced = classify_hash(str(frozen["prompt_sha256"]), current, legacy)
        reference_match = raw_utf8_sha256(str(source["output"])) == frozen["reference_sha256"]
        references_match = references_match and reference_match
        records.append(
            {
                "item_id": frozen["item_id"],
                "purpose": frozen["allocation"],
                "manifest": frozen["manifest"],
                "frozen_prompt_value": frozen["prompt_sha256"],
                "identified_schema_template": schema,
                "reproduced_value": reproduced,
                "reference_hash_matches": reference_match,
            }
        )
    counts = Counter(row["identified_schema_template"] for row in records)
    purpose_counts: dict[str, dict[str, int]] = {}
    for row in records:
        purpose_counts.setdefault(row["purpose"], {})
        purpose_counts[row["purpose"]][row["identified_schema_template"]] = (
            purpose_counts[row["purpose"]].get(row["identified_schema_template"], 0) + 1
        )
    system_prompts = []
    for family in SOURCE_FAMILIES:
        for polarity, text in (
            ("POSITIVE", family.positive_instruction),
            ("NEGATIVE", family.negative_instruction),
        ):
            system_prompts.append(
                {
                    "family_id": family.family_id,
                    "polarity": polarity,
                    "role": "system",
                    "exact_text": text,
                    "raw_utf8_sha256": raw_utf8_sha256(text),
                    "source": "src/epistemic_geometry/experiments/q2_v3.py::SOURCE_FAMILIES",
                }
            )
    primary = read_json(ORIGINAL / "PRIMARY_PANEL_MANIFEST.json")
    return {
        "schema_version": "q2-v3-complete-prompt-schema-audit-v1",
        "scope": {
            "frozen_item_prompt_records": len(records),
            "unique_item_ids": len({row["item_id"] for row in records}),
            "manifests": list(MANIFESTS),
            "source_system_instruction_records": len(system_prompts),
            "execution_teacher_text_records": 1,
            "technical_probe_prompt_source": (
                "first two SHELL_CALIBRATION items; aliases existing canonical item prompts"
            ),
            "technical_probe_new_prompt_count": 0,
        },
        "item_prompt_schema_template_counts": dict(counts),
        "item_prompt_schema_template_counts_by_purpose": purpose_counts,
        "all_reference_hashes_match": references_match,
        "unknown_or_collision_records": [
            row for row in records if row["identified_schema_template"] == "UNRESOLVED"
        ],
        "model_visible_envelope_counts_by_purpose": {
            "SOURCE_CONSTRUCTION": 24 * 10,
            "SOURCE_VALIDATION_AND_QUALIFICATION": 24 * 10,
            "SHELL_CALIBRATION": 12,
            "M1_COVARIANCE": 64,
            "M2_LABEL_FREE_PROBES": 12,
            "PRIMARY_SEMANTIC_PANEL": 200,
            "ENGINEERING_AND_COST_PREFLIGHT": 2,
        },
        "condition_semantics": (
            "activation-controller conditions do not alter prompt bytes; source conditions add "
            "one of ten frozen system instructions through the Qwen chat template"
        ),
        "chat_rendering": {
            "mode": "chat",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "enable_thinking": False,
            "base_manifest_field_scope": "exact user-message content bytes before chat rendering",
            "system_prompt_scope": "source construction/qualification only",
            "rendered_bytes_were_not_frozen_in_original_manifests": True,
        },
        "source_system_prompts": system_prompts,
        "execution_teacher_text": {
            "role": "teacher-forced continuation, not a user/system prompt",
            "exact_text": EXECUTION_TEACHER_TEXT,
            "raw_utf8_sha256": raw_utf8_sha256(EXECUTION_TEACHER_TEXT),
        },
        "primary_panel": {
            "n": primary["item_count"],
            "ordered_ids_sha256": primary["ordered_ids_sha256"],
            "ordered_ids_sha256_recomputed": ordered_id_hash(primary["item_ids"]),
            "expected_ordered_ids_sha256": EXPECTED_PRIMARY_HASH,
        },
        "records": records,
    }


def source_subset(public: dict[str, dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    ordered = sorted({str(row["item_id"]) for row in all_rows}, key=lambda value: int(value[7:]))
    path = OUTPUT / "OFFICIAL_SOURCE_RECORDS.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item_id in ordered:
            row = public[item_id]
            payload = {
                "id": item_id,
                "official_index": int(item_id[7:]),
                "code": row["code"],
                "input": row["input"],
                "output": row["output"],
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
            }
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def decision() -> dict[str, Any]:
    return {
        "schema_version": "q2-v3-provenance-reconciliation-decision-v1",
        "terminal_state": "Q2_V3_REFREEZE_REQUIRES_PRINCIPAL_RESEARCHER_DECISION",
        "original_failed_execution": {
            "head": FAILED_HEAD,
            "classification": "Q2_V3_PANEL_PROVENANCE_MISMATCH",
            "forensic_classification": "Q2_V3_FORENSIC_CLEAN",
            "scientific_trajectories": 0,
            "semantic_outcomes_opened": False,
            "prediction_lock_created": False,
        },
        "original_freeze": {
            "source_commit": ORIGINAL_FREEZE_SOURCE,
            "head": ORIGINAL_FREEZE_HEAD,
            "status": "Q2_V3_FROZEN_NOT_RUN_PRESERVED",
        },
        "classification_counts": {"P0": 0, "P1": 0, "P2": 9, "P3": 0},
        "why_no_automatic_amendment": (
            "The nine exact user-prompt byte strings differ in behaviorally plausible wording "
            "and instructions. The original prospective freeze did not bind a global prompt "
            "template version or exact prompt bytes for all 336 records; the executor's global "
            "Gate-7 template choice was introduced after the freeze. Selecting either template "
            "now is therefore a scientific design choice requiring principal review."
        ),
        "amendment1_created": False,
        "scientific_components_changed": [],
        "new_inference": False,
        "q3": "NOT_RUN",
    }


def candidate_contract_examples(
    public: dict[str, dict[str, Any]], all_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    examples = []
    for row in all_rows[:3]:
        source = public[str(row["item_id"])]
        prompt = current_task_prompt(str(source["code"]), str(source["input"]))
        examples.append(
            canonical_contract(
                item_id=str(row["item_id"]),
                purpose=str(row["allocation"]),
                prompt=prompt,
                prompt_template_version=CURRENT_TEMPLATE_VERSION,
                reference=str(source["output"]),
                dataset_repo=DATASET_REPO,
                dataset_revision=DATASET_REVISION,
            )
        )
    return {
        "status": "CANDIDATE_NOT_FROZEN_REQUIRES_PRINCIPAL_TEMPLATE_DECISION",
        "contract_requirements": {
            "item_identity": True,
            "source_reference_identity": True,
            "exact_user_prompt_bytes": True,
            "exact_system_prompt_bytes_when_present": True,
            "prompt_template_version": True,
            "task_namespace": True,
            "purpose": True,
            "encoding_and_normalization": True,
            "rendered_chat_prompt_hash_required_before_inference": True,
            "legacy_fields_preserved_separately": True,
        },
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    args = parser.parse_args()
    public = load_public(args.dataset_jsonl)
    by_id, all_rows = manifest_index()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_subset(public, all_rows)
    comparison = nine_comparisons(public, by_id)
    audit = schema_audit(public, all_rows)
    if comparison["classification_counts"] != {"P2": 9}:
        raise RuntimeError("nine-item P2 classification did not reproduce")
    if audit["item_prompt_schema_template_counts"] != {
        "CURRENT_TEMPLATE_RAW_SHA256": 327,
        "LEGACY_TEMPLATE_EXTERNAL_PROMPT": 9,
    }:
        raise RuntimeError("wider prompt schema counts did not reproduce")
    if not audit["all_reference_hashes_match"]:
        raise RuntimeError("reference provenance mismatch")
    if audit["primary_panel"]["ordered_ids_sha256_recomputed"] != EXPECTED_PRIMARY_HASH:
        raise RuntimeError("primary panel identity changed")
    write_json(OUTPUT / "NINE_ITEM_BYTE_COMPARISON.json", comparison)
    write_json(OUTPUT / "PROMPT_SCHEMA_AUDIT.json", audit)
    write_json(
        OUTPUT / "PROMPT_PROVENANCE_CONTRACT.json",
        candidate_contract_examples(public, all_rows),
    )
    write_json(OUTPUT / "PROTOCOL_DECISION.json", decision())
    write_json(
        OUTPUT / "SOURCE_PROVENANCE.json",
        {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "downloaded_source_sha256": file_sha256(args.dataset_jsonl),
            "downloaded_source_bytes": args.dataset_jsonl.stat().st_size,
            "downloaded_source_rows": len(public),
            "persisted_allocation_source_records": len(all_rows),
            "persisted_unique_source_records": len({row["item_id"] for row in all_rows}),
            "outcome_values_read_or_used": False,
        },
    )
    print(
        json.dumps(
            {
                "classification": decision()["terminal_state"],
                "P2": 9,
                "records": 336,
                "schemas": audit["item_prompt_schema_template_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
