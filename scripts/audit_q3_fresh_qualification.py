#!/usr/bin/env python3
"""Independent forensic reconstruction of the sealed Q3.4 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)

REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
SCHEDULE = REVIEW / "Q3_FRESH_QUALIFICATION_SCHEDULE.json"
LOCK = REVIEW / "Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json"
DATASET_SEAL = REVIEW / "Q3_FRESH_INSTRUMENT_DATASET_SEAL.json"
QUALIFICATION_MANIFEST = REVIEW / "QUALIFICATION_FAMILY_MANIFEST.json"
EXPECTED_JOURNAL_SHA256 = "2194646bcf25ff9512c5e3aaf35d4c2d0ed922f1f86ba6480709a1958dc89431"
EXPECTED_RECOVERY_SEAL_SHA256 = "e7eaf43da51690bd388c191283287374f3b45b7cb8f8015e33b790ff5a6e79ba"
EXPECTED_PRIVATE_DATASET_SHA256 = "c791e38c29d36a43fbac8ce00412e4c77d533665e0b8cb9eef8fa12fb918ac1d"
EXPECTED_ROWS = 6000
CHAMPION = "V4_DIRECTION_02_MEDIUM"
ROUTER = "ONLINE_ROUTED"
KEY_FIELDS = ("family_id", "condition", "rollout_index")
TERMINAL = {"EXTREME_MECHANICAL_REPETITION_V1", "max_new_tokens", "model_runtime_error"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["family_id"]), str(row["condition"]), int(row["rollout_index"])


def score(raw: dict[str, Any], reference: str) -> tuple[bool, bool, bool, str | None]:
    terminal = str(raw.get("terminal_reason", ""))
    if terminal in TERMINAL or raw.get("runtime_error") is not None:
        return False, False, False, None
    parsed = evaluate_external_answer_v3(
        str(raw.get("raw_output", "")),
        reference,
        truncated=bool(raw.get("truncated", False)),
        runtime_error=False,
    )
    return (
        bool(parsed.commitment_valid),
        bool(parsed.semantic_evaluable),
        bool(parsed.correct),
        parsed.canonical_value,
    )


def write_exclusive(path: Path, value: Any) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise RuntimeError("Q3_FORENSIC_SHORT_WRITE")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--private-dataset", type=Path, required=True)
    parser.add_argument("--private-scores", type=Path, required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(LOCK.read_bytes())
    schedule_payload = json.loads(SCHEDULE.read_bytes())
    schedule = list(schedule_payload["rows"])
    schedule_by_key = {key(row): (index, row) for index, row in enumerate(schedule)}
    if (
        sha(SCHEDULE) != lock["schedule_sha256"]
        or len(schedule) != EXPECTED_ROWS
        or len(schedule_by_key) != EXPECTED_ROWS
    ):
        raise RuntimeError("Q3_FORENSIC_SCHEDULE_INVALID")

    journal = args.raw_dir / "journal.recovered-candidate.jsonl"
    seal_path = args.raw_dir / "COLLECTION_COMPLETE_RECOVERY_SEAL.json"
    if sha(journal) != EXPECTED_JOURNAL_SHA256 or sha(seal_path) != EXPECTED_RECOVERY_SEAL_SHA256:
        raise RuntimeError("Q3_FORENSIC_RAW_HASH_MISMATCH")
    collection_seal = json.loads(seal_path.read_bytes())
    if (
        collection_seal.get("completed") != EXPECTED_ROWS
        or collection_seal.get("missing") != 0
        or collection_seal.get("duplicates") != 0
        or collection_seal.get("semantic_scoring") != "NOT_RUN"
    ):
        raise RuntimeError("Q3_FORENSIC_RAW_SEAL_INVALID")

    if sha(args.private_dataset) != EXPECTED_PRIVATE_DATASET_SHA256:
        raise RuntimeError("Q3_FORENSIC_REFERENCE_HASH_MISMATCH")
    references: dict[str, str] = {}
    reference_order: list[str] = []
    with args.private_dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reference_order.append(str(row["family_id"]))
            references[str(row["family_id"])] = str(row["reference_repr"])
    if len(references) != 300:
        raise RuntimeError("Q3_FORENSIC_REFERENCE_COVERAGE_INVALID")

    raw_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    identity = None
    identity_hash = None
    recovered = 0
    with journal.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            row = wrapper["row"]
            row_key = key(row)
            if identity is None:
                identity, identity_hash = wrapper["identity"], wrapper["identity_hash"]
            if (
                wrapper.get("identity") != identity
                or wrapper.get("identity_hash") != identity_hash
                or wrapper.get("key") != list(row_key)
                or row_key in raw_rows
                or row_key not in schedule_by_key
            ):
                raise RuntimeError(f"Q3_FORENSIC_WRAPPER_INVALID_{line_number}")
            index, planned = schedule_by_key[row_key]
            if row.get("schedule_index") != index or any(
                row.get(field) != value for field, value in planned.items()
            ):
                raise RuntimeError(f"Q3_FORENSIC_PROVENANCE_INVALID_{line_number}")
            recovered += int(
                row.get("persistence_provenance") == "REEXECUTED_MISSING_PERSISTED_KEY"
            )
            raw_rows[row_key] = row
    if set(raw_rows) != set(schedule_by_key) or recovered != 10:
        raise RuntimeError("Q3_FORENSIC_RAW_COVERAGE_INVALID")

    persisted_scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    with args.private_scores.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            persisted_scores[key(row)] = row
    if len(persisted_scores) != EXPECTED_ROWS:
        raise RuntimeError("Q3_FORENSIC_SCORE_COVERAGE_INVALID")

    values: dict[tuple[str, str, int], tuple[bool, bool, bool, str | None]] = {}
    max_parser_disagreement = 0
    for row_key, raw in raw_rows.items():
        observed = score(raw, references[row_key[0]])
        persisted = persisted_scores[row_key]
        expected = (
            bool(persisted["commitment_valid"]),
            bool(persisted["semantic_evaluable"]),
            bool(persisted["correct"]),
            persisted["canonical_value"],
        )
        max_parser_disagreement = max(max_parser_disagreement, int(observed != expected))
        values[row_key] = observed

    by_condition: dict[
        str, list[tuple[tuple[str, str, int], tuple[bool, bool, bool, str | None]]]
    ] = defaultdict(list)
    for row_key, value in values.items():
        by_condition[row_key[1]].append((row_key, value))
    summaries: dict[str, dict[str, float | int]] = {}
    for condition, rows in sorted(by_condition.items()):
        repetitions = sum(
            raw_rows[row_key].get("terminal_reason") == "EXTREME_MECHANICAL_REPETITION_V1"
            for row_key, _value in rows
        )
        summaries[condition] = {
            "rows": len(rows),
            "commitment_validity": sum(value[0] for _key, value in rows) / len(rows),
            "semantic_evaluability": sum(value[1] for _key, value in rows) / len(rows),
            "accuracy": sum(value[2] for _key, value in rows) / len(rows),
            "repetition_rate": repetitions / len(rows),
        }

    bank = [str(row["policy_id"]) for row in lock["policies"] if row["role"] == "BANK"]
    champion_values: list[float] = []
    router_values: list[float] = []
    oracle_values: list[float] = []
    for family_id in reference_order:
        champion = sum(values[(family_id, CHAMPION, rollout)][2] for rollout in (0, 1)) / 2
        router = sum(values[(family_id, ROUTER, rollout)][2] for rollout in (0, 1)) / 2
        oracle = max(
            sum(values[(family_id, condition, rollout)][2] for rollout in (0, 1)) / 2
            for condition in bank
        )
        champion_values.append(champion)
        router_values.append(router)
        oracle_values.append(oracle)
    quantities = {
        "champion_accuracy": sum(champion_values) / 300,
        "router_accuracy": sum(router_values) / 300,
        "routed_minus_champion_accuracy": sum(
            router - champion
            for router, champion in zip(router_values, champion_values, strict=True)
        )
        / 300,
        "frozen_bank_oracle_accuracy": sum(oracle_values) / 300,
        "frozen_bank_oracle_headroom_over_champion": sum(
            oracle - champion
            for oracle, champion in zip(oracle_values, champion_values, strict=True)
        )
        / 300,
    }
    dataset = json.loads(DATASET_SEAL.read_bytes())
    manifest = json.loads(QUALIFICATION_MANIFEST.read_bytes())
    passes = [
        dataset["global_checks"]["dual_evaluator_agreement"] == 1.0,
        dataset["global_checks"]["reference_repeat_determinism"] == 1.0,
        dataset["global_checks"]["parser_reference_roundtrip"] == 1.0,
        dataset["global_checks"]["cross_split_collisions"] == 0,
        manifest["structural_near_duplicate_rate"] <= 0.01,
        summaries[ROUTER]["commitment_validity"] >= 0.95,
        summaries[ROUTER]["semantic_evaluability"] >= 0.95,
        summaries[CHAMPION]["commitment_validity"] >= 0.95,
        summaries[CHAMPION]["semantic_evaluability"] >= 0.95,
        0.25 <= quantities["champion_accuracy"] <= 0.90,
        quantities["frozen_bank_oracle_headroom_over_champion"] >= 0.05,
        all(row["repetition_rate"] <= 0.10 for row in summaries.values()),
    ]
    classification = (
        "Q3_FRESH_INSTRUMENT_QUALIFIED_CONFIRMATION_NOT_AUTHORIZED"
        if all(passes)
        else "Q3_FRESH_INSTRUMENT_NOT_QUALIFIED"
    )
    primary = json.loads(args.primary_result.read_bytes())
    comparisons = [
        abs(float(primary["quantities"][name]) - float(value)) for name, value in quantities.items()
    ]
    for condition, summary in summaries.items():
        for name in ("commitment_validity", "semantic_evaluability", "accuracy", "repetition_rate"):
            comparisons.append(
                abs(
                    float(primary["quantities"]["condition_summaries"][condition][name])
                    - float(summary[name])
                )
            )
    max_difference = max(comparisons, default=0.0)
    clean = (
        max_parser_disagreement == 0
        and max_difference <= 1e-15
        and primary["status"] == classification
    )
    report = {
        "schema_version": "q3-fresh-qualification-forensic-audit-v1",
        "status": (
            "Q3_FRESH_INSTRUMENT_QUALIFICATION_FORENSIC_CLEAN"
            if clean
            else "Q3_FRESH_INSTRUMENT_QUALIFICATION_FORENSIC_DISAGREEMENT"
        ),
        "raw_journal_sha256": EXPECTED_JOURNAL_SHA256,
        "raw_rows": len(raw_rows),
        "missing": 0,
        "unexpected": 0,
        "duplicates": 0,
        "reexecuted_missing_rows": recovered,
        "parser_disagreement_rows": max_parser_disagreement,
        "max_aggregate_metric_difference": max_difference,
        "primary_classification": primary["status"],
        "forensic_classification": classification,
        "classification_agreement": primary["status"] == classification,
        "correctness_first_inspected_after_raw_seal": True,
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
    }
    write_exclusive(args.output, report)
    print(json.dumps({"status": report["status"], "max_difference": max_difference}))


if __name__ == "__main__":
    main()
