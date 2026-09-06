#!/usr/bin/env python3
"""Frozen post-seal scoring for the Q3.4 fresh-instrument qualification.

The scorer accepts only the exact recovered 6,000-row journal and the exact
private 300-family reference dataset.  Item-level scores remain private; the
release output contains aggregate qualification quantities only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    evaluate_external_answer_v3,
)

REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
LOCK = REVIEW / "Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json"
SCHEDULE = REVIEW / "Q3_FRESH_QUALIFICATION_SCHEDULE.json"
DATASET_SEAL = REVIEW / "Q3_FRESH_INSTRUMENT_DATASET_SEAL.json"
QUALIFICATION_MANIFEST = REVIEW / "QUALIFICATION_FAMILY_MANIFEST.json"
PARSER_SOURCE = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
EXPECTED_JOURNAL_SHA256 = "2194646bcf25ff9512c5e3aaf35d4c2d0ed922f1f86ba6480709a1958dc89431"
EXPECTED_JOURNAL_BYTES = 80_077_505
EXPECTED_RECOVERY_SEAL_SHA256 = "e7eaf43da51690bd388c191283287374f3b45b7cb8f8015e33b790ff5a6e79ba"
EXPECTED_PRIVATE_DATASET_SHA256 = "c791e38c29d36a43fbac8ce00412e4c77d533665e0b8cb9eef8fa12fb918ac1d"
EXPECTED_PARSER_SHA256 = "51ac492a6cea1284c36df6ef659520adf4a04e0595cf0e66bfd15ba172b960c3"
EXPECTED_ROWS = 6000
EXPECTED_FAMILIES = 300
KEY_FIELDS = ("family_id", "condition", "rollout_index")
CHAMPION = "V4_DIRECTION_02_MEDIUM"
ROUTER = "ONLINE_ROUTED"
RECOVERY_PROVENANCE = "REEXECUTED_MISSING_PERSISTED_KEY"
TERMINAL_FAILURES = {
    "EXTREME_MECHANICAL_REPETITION_V1": "REPETITION_STOP",
    "max_new_tokens": "HARD_CAP",
    "model_runtime_error": "RUNTIME_ERROR",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise RuntimeError("Q3_QUALIFICATION_SHORT_WRITE")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def key_of(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["family_id"]), str(row["condition"]), int(row["rollout_index"])


def load_schedule() -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    lock = read_json(LOCK)
    if sha256_file(SCHEDULE) != lock["schedule_sha256"]:
        raise RuntimeError("Q3_QUALIFICATION_SCHEDULE_HASH_MISMATCH")
    payload = read_json(SCHEDULE)
    rows = list(payload["rows"])
    by_key = {key_of(row): row for row in rows}
    if (
        payload.get("status") != "FROZEN_NOT_RUN"
        or len(rows) != EXPECTED_ROWS
        or len(by_key) != EXPECTED_ROWS
        or len({int(row["seed"]) for row in rows}) != EXPECTED_ROWS
    ):
        raise RuntimeError("Q3_QUALIFICATION_SCHEDULE_INVALID")
    return rows, by_key


def load_raw_rows(
    raw_dir: Path, schedule_by_key: dict[tuple[str, str, int], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    journal = raw_dir / "journal.recovered-candidate.jsonl"
    seal_path = raw_dir / "COLLECTION_COMPLETE_RECOVERY_SEAL.json"
    if (
        sha256_file(journal) != EXPECTED_JOURNAL_SHA256
        or journal.stat().st_size != EXPECTED_JOURNAL_BYTES
        or sha256_file(seal_path) != EXPECTED_RECOVERY_SEAL_SHA256
    ):
        raise RuntimeError("Q3_QUALIFICATION_RAW_SEAL_HASH_MISMATCH")
    seal = read_json(seal_path)
    required = {
        "status": "COLLECTION_COMPLETE_RAW_UNSCORED_AFTER_TEN_KEY_RECOVERY",
        "completed": EXPECTED_ROWS,
        "expected": EXPECTED_ROWS,
        "missing": 0,
        "unexpected": 0,
        "duplicates": 0,
        "replacements": 0,
        "original_persisted_rows": 5990,
        "reexecuted_missing_rows": 10,
        "journal_sha256": EXPECTED_JOURNAL_SHA256,
        "journal_bytes": EXPECTED_JOURNAL_BYTES,
        "correctness_inspected": False,
        "semantic_scoring": "NOT_RUN",
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
    }
    if any(seal.get(key) != value for key, value in required.items()):
        raise RuntimeError("Q3_QUALIFICATION_RAW_SEAL_INVALID")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    identity: dict[str, Any] | None = None
    identity_hash: str | None = None
    recovered = 0
    schedule_indices = {key: index for index, key in enumerate(schedule_by_key)}
    with journal.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            row = wrapper["row"]
            key = key_of(row)
            if identity is None:
                identity, identity_hash = wrapper["identity"], wrapper["identity_hash"]
            if (
                wrapper.get("version") != "research-os-jsonl-v1"
                or wrapper.get("identity") != identity
                or wrapper.get("identity_hash") != identity_hash
                or wrapper.get("key_fields") != list(KEY_FIELDS)
                or wrapper.get("key") != list(key)
                or key in seen
                or key not in schedule_by_key
            ):
                raise RuntimeError(f"Q3_QUALIFICATION_RAW_WRAPPER_INVALID_{line_number}")
            planned = schedule_by_key[key]
            if row.get("schedule_index") != schedule_indices[key] or any(
                row.get(field) != value for field, value in planned.items()
            ):
                raise RuntimeError(f"Q3_QUALIFICATION_RAW_PROVENANCE_INVALID_{line_number}")
            if row.get("persistence_provenance") == RECOVERY_PROVENANCE:
                recovered += 1
            elif "persistence_provenance" in row:
                raise RuntimeError("Q3_QUALIFICATION_UNKNOWN_RECOVERY_PROVENANCE")
            rows.append(row)
            seen.add(key)
    if len(rows) != EXPECTED_ROWS or seen != set(schedule_by_key) or recovered != 10:
        raise RuntimeError("Q3_QUALIFICATION_RAW_COVERAGE_INVALID")
    if identity is None:
        raise RuntimeError("Q3_QUALIFICATION_RAW_IDENTITY_MISSING")
    return rows, seal, identity


def load_references(path: Path) -> tuple[list[str], dict[str, str], dict[str, str]]:
    lock = read_json(LOCK)
    manifest = read_json(QUALIFICATION_MANIFEST)
    if (
        sha256_file(path) != EXPECTED_PRIVATE_DATASET_SHA256
        or manifest["private_dataset"]["sha256"] != EXPECTED_PRIVATE_DATASET_SHA256
        or sha256_file(QUALIFICATION_MANIFEST) != lock["qualification_manifest_sha256"]
        or sha256_file(DATASET_SEAL) != lock["dataset_seal_sha256"]
        or sha256_file(PARSER_SOURCE) != EXPECTED_PARSER_SHA256
    ):
        raise RuntimeError("Q3_QUALIFICATION_REFERENCE_DATASET_HASH_MISMATCH")
    ids: list[str] = []
    references: dict[str, str] = {}
    reference_types: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            family_id = str(row["family_id"])
            ids.append(family_id)
            references[family_id] = str(row["reference_repr"])
            reference_types[family_id] = str(row["reference_type"])
    if len(ids) != EXPECTED_FAMILIES or len(set(ids)) != EXPECTED_FAMILIES:
        raise RuntimeError("Q3_QUALIFICATION_REFERENCE_COVERAGE_INVALID")
    manifest_ids = [str(row["family_id"]) for row in manifest["families"]]
    if ids != manifest_ids:
        raise RuntimeError("Q3_QUALIFICATION_REFERENCE_ORDER_INVALID")
    return ids, references, reference_types


def classify_row(row: dict[str, Any], reference: str) -> dict[str, Any]:
    terminal = str(row.get("terminal_reason", ""))
    if terminal in TERMINAL_FAILURES or row.get("runtime_error") is not None:
        status = TERMINAL_FAILURES.get(terminal, "RUNTIME_ERROR")
        result = {
            "commitment_valid": False,
            "semantic_evaluable": False,
            "correct": False,
            "value_type": None,
            "canonical_value": None,
            "failure_reason": terminal or "runtime error",
            "status": status,
        }
    else:
        parsed = evaluate_external_answer_v3(
            str(row.get("raw_output", "")),
            reference,
            truncated=bool(row.get("truncated", False)),
            runtime_error=False,
        )
        status = (
            "VALID_CORRECT"
            if parsed.correct
            else "VALID_WRONG"
            if parsed.commitment_valid and parsed.semantic_evaluable
            else "INVALID_FORMAT"
        )
        result = {
            "commitment_valid": bool(parsed.commitment_valid),
            "semantic_evaluable": bool(parsed.semantic_evaluable),
            "correct": bool(parsed.correct),
            "value_type": parsed.value_type,
            "canonical_value": parsed.canonical_value,
            "failure_reason": parsed.failure_reason,
            "status": status,
        }
    return {
        **{field: row[field] for field in KEY_FIELDS},
        **result,
        "terminal_reason": terminal,
        "generated_token_count": int(row["generated_token_count"]),
        "raw_output_sha256": hashlib.sha256(str(row.get("raw_output", "")).encode()).hexdigest(),
        "persistence_provenance": row.get("persistence_provenance", "ORIGINAL_PERSISTED_ROW"),
    }


def score_rows(
    rows: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    references: dict[str, str],
    private_dir: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    by_key = {key_of(row): row for row in rows}
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    output = private_dir / "Q3_FRESH_QUALIFICATION_SCORES.jsonl"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        for planned in schedule:
            key = key_of(planned)
            score = classify_row(by_key[key], references[key[0]])
            raw = (json.dumps(score, sort_keys=True, ensure_ascii=False) + "\n").encode()
            offset = 0
            while offset < len(raw):
                written = os.write(fd, raw[offset:])
                if written <= 0:
                    raise RuntimeError("Q3_QUALIFICATION_SCORE_SHORT_WRITE")
                offset += written
            scores[key] = score
        os.fsync(fd)
    finally:
        os.close(fd)
    return scores


def condition_summaries(
    scores: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for score in scores.values():
        grouped.setdefault(str(score["condition"]), []).append(score)
    output: dict[str, dict[str, Any]] = {}
    for condition, rows in sorted(grouped.items()):
        tokens = np.asarray([int(row["generated_token_count"]) for row in rows], dtype=float)
        status = Counter(str(row["status"]) for row in rows)
        output[condition] = {
            "rows": len(rows),
            "commitment_valid": sum(bool(row["commitment_valid"]) for row in rows),
            "commitment_validity": float(np.mean([row["commitment_valid"] for row in rows])),
            "semantic_evaluable": sum(bool(row["semantic_evaluable"]) for row in rows),
            "semantic_evaluability": float(np.mean([row["semantic_evaluable"] for row in rows])),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": float(np.mean([row["correct"] for row in rows])),
            "repetition_stops": status.get("REPETITION_STOP", 0),
            "repetition_rate": status.get("REPETITION_STOP", 0) / len(rows),
            "hard_caps": status.get("HARD_CAP", 0),
            "generated_token_mean": float(np.mean(tokens)),
            "generated_token_median": float(np.median(tokens)),
            "status_counts": dict(sorted(status.items())),
        }
    return output


def qualification_quantities(
    family_ids: list[str],
    scores: dict[tuple[str, str, int], dict[str, Any]],
    bank: list[str],
) -> dict[str, Any]:
    summaries = condition_summaries(scores)
    champion_family: list[float] = []
    router_family: list[float] = []
    oracle_family: list[float] = []
    for family_id in family_ids:
        champion_value = float(
            np.mean([scores[(family_id, CHAMPION, rollout)]["correct"] for rollout in (0, 1)])
        )
        router_value = float(
            np.mean([scores[(family_id, ROUTER, rollout)]["correct"] for rollout in (0, 1)])
        )
        bank_values = [
            float(
                np.mean([scores[(family_id, condition, rollout)]["correct"] for rollout in (0, 1)])
            )
            for condition in bank
        ]
        champion_family.append(champion_value)
        router_family.append(router_value)
        oracle_family.append(max(bank_values))
    oracle_headroom = float(np.mean(np.asarray(oracle_family) - np.asarray(champion_family)))
    routed_gain = float(np.mean(np.asarray(router_family) - np.asarray(champion_family)))
    return {
        "condition_summaries": summaries,
        "champion_accuracy": float(np.mean(champion_family)),
        "router_accuracy": float(np.mean(router_family)),
        "routed_minus_champion_accuracy": routed_gain,
        "routed_gain_is_qualification_gate": False,
        "frozen_bank_oracle_accuracy": float(np.mean(oracle_family)),
        "frozen_bank_oracle_headroom_over_champion": oracle_headroom,
    }


def classify_gates(quantities: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    summaries = quantities["condition_summaries"]
    dataset = read_json(DATASET_SEAL)
    manifest = read_json(QUALIFICATION_MANIFEST)
    checks: dict[str, tuple[Any, str, bool]] = {
        "dual_evaluator_agreement": (
            dataset["global_checks"]["dual_evaluator_agreement"],
            "= 1.0",
            dataset["global_checks"]["dual_evaluator_agreement"] == 1.0,
        ),
        "evaluator_repeat_determinism": (
            dataset["global_checks"]["reference_repeat_determinism"],
            "= 1.0",
            dataset["global_checks"]["reference_repeat_determinism"] == 1.0,
        ),
        "parser_reference_roundtrip": (
            dataset["global_checks"]["parser_reference_roundtrip"],
            "= 1.0",
            dataset["global_checks"]["parser_reference_roundtrip"] == 1.0,
        ),
        "cross_split_family_or_skeleton_collision": (
            dataset["global_checks"]["cross_split_collisions"],
            "= 0",
            dataset["global_checks"]["cross_split_collisions"] == 0,
        ),
        "structural_near_duplicate_rate": (
            manifest["structural_near_duplicate_rate"],
            "<= 0.01",
            manifest["structural_near_duplicate_rate"] <= 0.01,
        ),
        "router_commitment_validity": (
            summaries[ROUTER]["commitment_validity"],
            ">= 0.95",
            summaries[ROUTER]["commitment_validity"] >= 0.95,
        ),
        "router_semantic_evaluability": (
            summaries[ROUTER]["semantic_evaluability"],
            ">= 0.95",
            summaries[ROUTER]["semantic_evaluability"] >= 0.95,
        ),
        "champion_commitment_validity": (
            summaries[CHAMPION]["commitment_validity"],
            ">= 0.95",
            summaries[CHAMPION]["commitment_validity"] >= 0.95,
        ),
        "champion_semantic_evaluability": (
            summaries[CHAMPION]["semantic_evaluability"],
            ">= 0.95",
            summaries[CHAMPION]["semantic_evaluability"] >= 0.95,
        ),
        "champion_accuracy": (
            quantities["champion_accuracy"],
            "within [0.25, 0.90]",
            0.25 <= quantities["champion_accuracy"] <= 0.90,
        ),
        "frozen_bank_oracle_headroom_over_champion": (
            quantities["frozen_bank_oracle_headroom_over_champion"],
            ">= 0.05",
            quantities["frozen_bank_oracle_headroom_over_champion"] >= 0.05,
        ),
        "terminal_repetition_rate_all_conditions": (
            max(row["repetition_rate"] for row in summaries.values()),
            "<= 0.10 for every condition",
            all(row["repetition_rate"] <= 0.10 for row in summaries.values()),
        ),
    }
    gates = {
        name: {"observed": observed, "required": required, "pass": passed}
        for name, (observed, required, passed) in checks.items()
    }
    classification = (
        "Q3_FRESH_INSTRUMENT_QUALIFIED_CONFIRMATION_NOT_AUTHORIZED"
        if all(row["pass"] for row in gates.values())
        else "Q3_FRESH_INSTRUMENT_NOT_QUALIFIED"
    )
    return gates, classification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--private-dataset", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--release-output", type=Path, required=True)
    args = parser.parse_args()
    if args.private_output_dir.exists() and any(args.private_output_dir.iterdir()):
        raise SystemExit("private output directory must be absent or empty")
    args.private_output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    schedule, schedule_by_key = load_schedule()
    rows, raw_seal, identity = load_raw_rows(args.raw_dir, schedule_by_key)
    family_ids, references, reference_types = load_references(args.private_dataset)
    if set(family_ids) != {row["family_id"] for row in schedule}:
        raise RuntimeError("Q3_QUALIFICATION_FAMILY_SET_MISMATCH")
    lock = read_json(LOCK)
    bank = [str(row["policy_id"]) for row in lock["policies"] if row["role"] == "BANK"]
    if len(bank) != 8 or {row["condition"] for row in schedule} != set(bank) | {
        CHAMPION,
        ROUTER,
    }:
        raise RuntimeError("Q3_QUALIFICATION_CONDITION_SET_MISMATCH")
    scores = score_rows(rows, schedule, references, args.private_output_dir)
    quantities = qualification_quantities(family_ids, scores, bank)
    gates, classification = classify_gates(quantities)
    scores_path = args.private_output_dir / "Q3_FRESH_QUALIFICATION_SCORES.jsonl"
    private_seal = {
        "schema_version": "q3-fresh-qualification-private-score-seal-v1",
        "status": "QUALIFICATION_SCORED_AFTER_COMPLETE_RAW_SEAL",
        "raw_journal_sha256": EXPECTED_JOURNAL_SHA256,
        "raw_collection_seal_sha256": EXPECTED_RECOVERY_SEAL_SHA256,
        "scores_sha256": sha256_file(scores_path),
        "scores_rows": len(scores),
        "private_dataset_sha256": EXPECTED_PRIVATE_DATASET_SHA256,
        "parser": PARSER_VERSION,
        "parser_source_sha256": sha256_file(PARSER_SOURCE),
        "correctness_first_inspected_after_raw_seal": True,
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
    }
    write_exclusive_json(
        args.private_output_dir / "Q3_FRESH_QUALIFICATION_PRIVATE_SCORE_SEAL.json",
        private_seal,
    )
    release = {
        "schema_version": "q3-fresh-qualification-result-v1",
        "status": classification,
        "evidence_level": "DEVELOPMENT_INSTRUMENT_QUALIFICATION",
        "raw_provenance": {
            "journal_sha256": EXPECTED_JOURNAL_SHA256,
            "journal_rows": EXPECTED_ROWS,
            "collection_seal_sha256": EXPECTED_RECOVERY_SEAL_SHA256,
            "original_persisted_rows": raw_seal["original_persisted_rows"],
            "reexecuted_missing_rows": raw_seal["reexecuted_missing_rows"],
            "reexecution_label": RECOVERY_PROVENANCE,
        },
        "scoring": {
            "parser": PARSER_VERSION,
            "parser_source_sha256": sha256_file(PARSER_SOURCE),
            "private_scores_sha256": private_seal["scores_sha256"],
            "private_reference_type_counts": dict(
                sorted(Counter(reference_types.values()).items())
            ),
            "correctness_first_inspected_after_raw_seal": True,
        },
        "bank_policy_order": bank,
        "champion": CHAMPION,
        "router": ROUTER,
        "quantities": quantities,
        "gates": gates,
        "routed_gain_used_for_qualification": False,
        "confirmation_status": "CLOSED_NOT_AUTHORIZED",
        "reserve_status": "CLOSED_NOT_AUTHORIZED",
        "q3_confirmatory_result": "NOT_RUN",
        "raw_text_in_release_artifact": False,
        "private_identity_hash": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    write_exclusive_json(args.release_output, release)
    print(json.dumps({"status": classification, "release_output": str(args.release_output)}))


if __name__ == "__main__":
    main()
