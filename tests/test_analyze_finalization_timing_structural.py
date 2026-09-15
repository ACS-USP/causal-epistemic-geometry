from __future__ import annotations

import json

from scripts.analyze_finalization_timing_structural import analyze
from scripts.seal_finalization_timing_structure import seal_structure

from epistemic_geometry.benchmarks.reasoning.finalization_timing import (
    build_manifest,
    build_schedule,
)
from epistemic_geometry.benchmarks.reasoning.finalization_timing_journal import identity_hash


def _seal(tmp_path, *, d75_marker: bool):
    manifest = build_manifest(n_per_cell=16)
    schedule = build_schedule(manifest)
    manifest_path = tmp_path / "manifest.json"
    schedule_path = tmp_path / "schedule.json"
    journal_path = tmp_path / "journal.jsonl"
    manifest_path.write_text(json.dumps(manifest))
    schedule_path.write_text(json.dumps(schedule))
    identity = {"run": "structural", "candidate_identity_hash": "fixed"}
    wrappers = []
    for row in schedule:
        tokens = [9, 151668] if d75_marker and row["condition"] == "D75" else [9]
        wrappers.append(
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
                "record": {
                    "latent_id": row["latent_id"],
                    "rollout_index": row["rollout_index"],
                    "token_ids": tokens,
                    "metadata": {
                        "cap": row["cap"],
                        "condition": row["condition"],
                        "schedule_identity_hash": row["schedule_identity_hash"],
                    },
                },
            }
        )
    journal_path.write_text("\n".join(json.dumps(row) for row in wrappers) + "\n")
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(
        json.dumps(seal_structure(manifest_path, schedule_path, journal_path, identity))
    )
    return seal_path


def test_analysis_supports_large_earlier_structural_transition(tmp_path) -> None:
    result = analyze(_seal(tmp_path, d75_marker=True))
    assert result["classification"] == "D75_FINALIZATION_TIMING_EARLIER_SUPPORTED"
    assert result["statuses"]["d75_earlier_by_delta"] == "TRIGGERED"


def test_analysis_excludes_large_shift_when_structural_records_match(tmp_path) -> None:
    result = analyze(_seal(tmp_path, d75_marker=False))
    assert result["classification"] == "D75_FINALIZATION_TIMING_RELEVANT_SHIFT_EXCLUDED"
    assert result["statuses"]["exclude_both_directions_by_delta"] == "TRIGGERED"
