from __future__ import annotations

import json

import pytest
from scripts.seal_finalization_timing_structure import StructuralSealError, seal_structure

from epistemic_geometry.benchmarks.reasoning.finalization_timing import (
    build_manifest,
    build_schedule,
)
from epistemic_geometry.benchmarks.reasoning.finalization_timing_journal import identity_hash


def _write(path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _journal(path, schedule, identity) -> None:
    records = []
    for index, row in enumerate(schedule):
        token_ids = [9, 151668] if index % 2 else [9]
        record = {
            "latent_id": row["latent_id"],
            "rollout_index": row["rollout_index"],
            "token_ids": token_ids,
            "metadata": {
                "cap": row["cap"],
                "condition": row["condition"],
                "schedule_identity_hash": row["schedule_identity_hash"],
            },
        }
        records.append(
            {
                "identity": identity,
                "identity_hash": identity_hash(identity),
                "physical_key": [
                    row["latent_id"],
                    row["cap"],
                    row["condition"],
                    row["rollout_index"],
                ],
                "schedule_identity_hash": row["schedule_identity_hash"],
                "schedule": row,
                "record": record,
            }
        )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n")


def test_structural_seal_never_requires_semantic_fields(tmp_path) -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    manifest_path = tmp_path / "manifest.json"
    schedule_path = tmp_path / "schedule.json"
    journal_path = tmp_path / "journal.jsonl"
    _write(manifest_path, manifest)
    _write(schedule_path, schedule)
    identity = {"run": "timing", "candidate_identity_hash": "frozen"}
    _journal(journal_path, schedule, identity)
    sealed = seal_structure(manifest_path, schedule_path, journal_path, identity)
    assert sealed["logical_key_count"] == len(schedule)
    assert {record["restricted_think_close_time"] for record in sealed["records"]} == {2, 4096}
    assert all("correct" not in record for record in sealed["records"])
    assert all("raw_text" not in record for record in sealed["records"])


def test_structural_seal_rejects_extra_or_missing_rows(tmp_path) -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    manifest_path = tmp_path / "manifest.json"
    schedule_path = tmp_path / "schedule.json"
    journal_path = tmp_path / "journal.jsonl"
    _write(manifest_path, manifest)
    _write(schedule_path, schedule)
    identity = {"run": "timing"}
    _journal(journal_path, schedule[:-1], identity)
    with pytest.raises(StructuralSealError, match="exactly"):
        seal_structure(manifest_path, schedule_path, journal_path, identity)
