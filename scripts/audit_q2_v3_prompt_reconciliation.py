#!/usr/bin/env python3
"""Independent CPU-only audit of the Q2 V3 prompt reconciliation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "review/q2_v3_radial_angular_freeze"
REVIEW = ROOT / "review/q2_v3_provenance_reconciliation"
MANIFESTS = (
    "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION_MANIFEST.json",
    "SHELL_CALIBRATION_MANIFEST.json",
    "M1_COVARIANCE_MANIFEST.json",
    "M2_PROBE_MANIFEST.json",
    "PRIMARY_PANEL_MANIFEST.json",
)
NINE = {
    "sample_300",
    "sample_74",
    "sample_745",
    "sample_700",
    "sample_659",
    "sample_777",
    "sample_145",
    "sample_698",
    "sample_21",
}
EXPECTED_PRIMARY = "969da4b5bac9c2fddd7e40db1c6a82f019ac84f0fbde4662450354ff528780cf"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_prompt(code: str, value: str) -> str:
    return (
        "Solve the following code-output prediction problem.\n\n"
        "Python function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Reason carefully, then end with exactly one line in the form "
        "FINAL: <the exact Python output>. Do not add text after FINAL."
    )


def new_prompt(code: str, value: str) -> str:
    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )


def raw_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def old_namespaced_hash(text: str) -> str:
    return hashlib.sha256(b"EXTERNAL-PROMPT\x1f" + text.encode("utf-8")).hexdigest()


def ordered_hash(values: list[str]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    source_rows = [
        json.loads(line)
        for line in (REVIEW / "OFFICIAL_SOURCE_RECORDS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    source = {str(row["id"]): row for row in source_rows}
    frozen_rows: list[dict[str, Any]] = []
    for filename in MANIFESTS:
        payload = read_json(ORIGINAL / filename)
        frozen_rows.extend({**row, "manifest": filename} for row in payload["items"])
    counts: Counter[str] = Counter()
    mismatches: set[str] = set()
    references_ok = True
    p2_ok = True
    for row in frozen_rows:
        item_id = str(row["item_id"])
        public = source[item_id]
        old = old_prompt(str(public["code"]), str(public["input"]))
        new = new_prompt(str(public["code"]), str(public["input"]))
        frozen = str(row["prompt_sha256"])
        if frozen == raw_hash(new):
            counts["CURRENT_TEMPLATE_RAW_SHA256"] += 1
        elif frozen == old_namespaced_hash(old):
            counts["LEGACY_TEMPLATE_EXTERNAL_PROMPT"] += 1
            mismatches.add(item_id)
            p2_ok = p2_ok and old.encode("utf-8") != new.encode("utf-8") and (
                "Reason carefully" in old and "Return exactly one final line" in new
            )
        else:
            counts["UNRESOLVED"] += 1
        references_ok = references_ok and str(row["reference_sha256"]) == raw_hash(
            str(public["output"])
        )
    failure = read_json(ORIGINAL / "Q2_V3_PANEL_PROVENANCE_FAILURE.json")
    original_hashes = read_json(ORIGINAL / "Q2_V3_EXECUTION_ARTIFACT_HASHES.json")
    original_immutable = all(
        file_hash(ORIGINAL / name) == expected
        for name, expected in original_hashes.items()
        if name.endswith((".json", ".md"))
    )
    forensic = read_json(ORIGINAL / "Q2_V3_FORENSIC_AUDIT.json")
    original_state_ok = (
        failure["scientific_model_inference_rows"] == 0
        and failure["semantic_outcomes_opened"] is False
        and forensic["model_inference_rows"] == 0
        and forensic["prediction_lock_created"] is False
        and forensic["audit_classification"] == "Q2_V3_FORENSIC_CLEAN"
    )
    panel = read_json(ORIGINAL / "PRIMARY_PANEL_MANIFEST.json")
    comparison = read_json(REVIEW / "NINE_ITEM_BYTE_COMPARISON.json")
    decision = read_json(REVIEW / "PROTOCOL_DECISION.json")
    checks = {
        "source_subset_336_unique": len(source_rows) == len(source) == 336,
        "six_purpose_manifests_covered": len(frozen_rows) == 336,
        "schema_counts_reproduced": dict(counts)
        == {"CURRENT_TEMPLATE_RAW_SHA256": 327, "LEGACY_TEMPLATE_EXTERNAL_PROMPT": 9},
        "nine_ids_reproduced": mismatches == NINE,
        "all_nine_are_p2_wording_changes": p2_ok
        and comparison["classification_counts"] == {"P2": 9},
        "all_reference_hashes_match": references_ok,
        "primary_n_200": panel["item_count"] == 200,
        "primary_ordered_id_hash_preserved": ordered_hash(panel["item_ids"])
        == EXPECTED_PRIMARY,
        "original_clean_abort_artifacts_immutable": original_immutable,
        "original_zero_outcome_state_preserved": original_state_ok,
        "no_amendment_silently_created": decision["amendment1_created"] is False,
        "terminal_decision_is_principal_review": decision["terminal_state"]
        == "Q2_V3_REFREEZE_REQUIRES_PRINCIPAL_RESEARCHER_DECISION",
        "q3_not_run": decision["q3"] == "NOT_RUN",
    }
    passed = all(checks.values())
    classification = (
        "Q2_V3_PROVENANCE_RECONCILIATION_AUDIT_PASS"
        if passed
        else "Q2_V3_PROVENANCE_RECONCILIATION_AUDIT_FAIL"
    )
    result = {
        "schema_version": "q2-v3-prompt-reconciliation-independent-audit-v1",
        "classification": classification,
        "checks": checks,
        "independent_schema_counts": dict(counts),
        "independent_mismatch_ids": sorted(mismatches, key=lambda value: int(value[7:])),
        "model_inference": "NONE",
        "semantic_trajectories": 0,
        "prediction_matrices": "NONE",
        "q3": "NOT_RUN",
    }
    (REVIEW / "INDEPENDENT_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REVIEW / "INDEPENDENT_AUDIT.md").write_text(
        "# Q2 V3 prompt reconciliation — independent audit\n\n"
        f"Classification: `{classification}`\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in checks.items())
        + "\n\nThis audit used no model inference or behavioral outcome value. Amendment 1 "
        "was not created because all nine discrepancies are P2 and the original "
        "freeze did not prospectively determine one global prompt template.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "checks": checks}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
