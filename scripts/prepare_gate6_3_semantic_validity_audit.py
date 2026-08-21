#!/usr/bin/env python3
"""Freeze a condition-masked Gate 6.3 semantic-V3 audit corpus and lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    canonicalize_semantic_value,
)

SOURCE = ROOT / "review/gate6_3_single_mean_semantic_evaluation"
AUDIT = ROOT / "review/gate6_3_semantic_validity_audit"
EXPECTED_JOURNAL_SHA256 = "593c89e8bf13d83d2fcfa27b2a9d7eec7d4c0b17918185f0554d37390ca601e1"
HISTORICAL_CLASSIFICATION = "GATE6_3_SINGLE_MEAN_DESTRUCTIVE"
IMMUTABLE_FILES = (
    "journal.jsonl",
    "REPORT.md",
    "CONDITION_SUMMARY.csv",
    "ESTIMANDS.json",
    "BOOTSTRAP_INTERVALS.json",
    "FORENSIC_CLOSEOUT.json",
    "EVALUATION_RESULTS.csv",
    "PROTOCOL_LOCK.md",
    "PROTOCOL_LOCK.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (SOURCE / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reference_type(reference: str) -> str:
    return str(canonicalize_semantic_value(reference)[0])


def build_masked_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    duplicate_ordinals: Counter[str] = Counter()
    for row in rows:
        raw_output = str(row.get("raw_output", ""))
        record = {
            "phase_family": (
                "PRIMARY_EVALUATION"
                if str(row.get("phase")) == "GATE6_3_PRIMARY_EVALUATION"
                else "MATCHED_CONSISTENCY"
            ),
            "raw_output": raw_output,
            "reference_type": reference_type(str(row.get("reference_answer", ""))),
            "truncated": str(row.get("status")) == "TRUNCATED"
            or int(row.get("generated_token_count", 0)) >= 4096,
            "runtime_error": str(row.get("status")) == "RUNTIME_ERROR",
            "generated_token_count": int(row.get("generated_token_count", 0)),
        }
        base = hashlib.sha256(canonical_json(record).encode()).hexdigest()
        ordinal = duplicate_ordinals[base]
        duplicate_ordinals[base] += 1
        record["audit_row_id"] = hashlib.sha256(f"{base}:{ordinal}".encode()).hexdigest()
        records.append(record)
    records.sort(key=lambda record: str(record["audit_row_id"]))
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    actual_journal = digest(SOURCE / "journal.jsonl")
    if actual_journal != EXPECTED_JOURNAL_SHA256:
        raise RuntimeError("historical Gate 6.3 journal digest mismatch")
    rows = load_rows()
    if len(rows) != 920:
        raise RuntimeError(f"expected 920 preserved Gate 6.3 rows, found {len(rows)}")
    phase_counts = Counter(str(row.get("phase")) for row in rows)
    if phase_counts != {
        "GATE6_3_MATCHED_RANDOM_SUPPLEMENT": 80,
        "GATE6_3_PRIMARY_EVALUATION": 840,
    }:
        raise RuntimeError(f"unexpected Gate 6.3 phase counts: {dict(phase_counts)}")

    immutable = {name: digest(SOURCE / name) for name in IMMUTABLE_FILES}
    corpus = build_masked_corpus(rows)
    corpus_path = output / "BLINDED_CORPUS.jsonl"
    corpus_path.write_text("".join(canonical_json(record) + "\n" for record in corpus))
    corpus_hash = digest(corpus_path)
    parser_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    tests_path = ROOT / "tests/test_semantic_v3.py"
    spec_path = output / "SEMANTIC_V3_SPEC.md"
    provenance = {
        "schema_version": 1,
        "rows": len(corpus),
        "primary_rows": 840,
        "matched_consistency_rows": 80,
        "corpus_sha256": corpus_hash,
        "source_journal_sha256": actual_journal,
        "included_fields": sorted(corpus[0]),
        "excluded_fields": [
            "condition",
            "item_id",
            "original_correctness",
            "original_parser_status",
            "G_C_D_contribution",
        ],
        "ordering": "lexical audit_row_id derived without condition or correctness",
        "model_inference": False,
        "condition_labels_restored": False,
    }
    write_json(output / "BLINDED_CORPUS_PROVENANCE.json", provenance)
    test_record = {
        "parser_version": PARSER_VERSION,
        "parser_sha256": digest(parser_path),
        "test_module_sha256": digest(tests_path),
        "synthetic_test_cases": 41,
        "historical_v2_disagreement_patterns_required": True,
        "condition_invariance_required": True,
    }
    write_json(output / "SEMANTIC_V3_TESTS.json", test_record)
    lock = {
        "schema_version": 1,
        "audit_id": "GATE6_3_SEMANTIC_VALIDITY_AUDIT",
        "stage": "OFFLINE_CLASS_C_LOCK",
        "model_inference": False,
        "gpu_authorized": False,
        "historical_classification": HISTORICAL_CLASSIFICATION,
        "historical_result_mutable": False,
        "scientific_base": "32a3cd2f0ec303fcc7951fbaf694db46265cc321",
        "audit_source_commit": git_head(),
        "parser": {
            "version": PARSER_VERSION,
            "module_sha256": digest(parser_path),
            "spec_sha256": digest(spec_path),
            "tests_sha256": digest(tests_path),
        },
        "blinded_corpus_sha256": corpus_hash,
        "source_journal_sha256": actual_journal,
        "immutable_source_files": immutable,
        "condition_blind_rule_development": True,
        "condition_labels_may_be_restored_only_after_lock": True,
        "bootstrap": {"resamples": 5000, "cluster": "item_id", "seed": 6313003},
        "diagnostic_guards": {
            "commitment_validity_minimum": 0.90,
            "commitment_validity_drop_max": 0.05,
            "semantic_evaluability_minimum": 0.90,
            "semantic_evaluability_drop_max": 0.05,
            "accuracy_drop_max": 0.10,
            "D_minimum": 0.05,
            "D_minus_random_mean_minimum": 0.05,
            "D_gt_random_max": True,
            "G_minimum": 0.03,
            "C_minimum": 0.03,
            "C_minus_random_mean_minimum": 0.05,
            "C_gt_random_max": True,
        },
        "firewall": {
            "runpod": "NOT_ACCESSED",
            "new_trajectories": 0,
            "q2": "NOT_RUN",
            "character_count": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
        },
    }
    write_json(output / "AUDIT_LOCK.json", lock)
    (output / "AUDIT_LOCK.md").write_text(
        "# Gate 6.3 Semantic-Validity Audit Lock\n\n"
        f"- Historical classification: `{HISTORICAL_CLASSIFICATION}` (immutable)\n"
        f"- Scientific base: `{lock['scientific_base']}`\n"
        f"- Audit source commit: `{lock['audit_source_commit']}`\n"
        f"- Parser: `{PARSER_VERSION}`\n"
        f"- Parser SHA-256: `{lock['parser']['module_sha256']}`\n"
        f"- Blinded corpus SHA-256: `{corpus_hash}`\n"
        "- Condition labels/correctness/G-C-D contributions were excluded while "
        "rules were frozen.\n"
        "- No model inference, RunPod access, Q2, character count, or holdout "
        "access is authorized.\n"
    )
    print(json.dumps({"lock": str(output / "AUDIT_LOCK.json"), "rows": len(corpus)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
