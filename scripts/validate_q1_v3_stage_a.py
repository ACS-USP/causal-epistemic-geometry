#!/usr/bin/env python3
"""Validate a completed Q1 V3 Stage-A baseline artifact without model access."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from epistemic_geometry.reproducibility import stable_digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        rows.append(row)
    return rows


def _journal_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        wrapped = json.loads(line)
        if wrapped.get("journal_version") != "q1-v3-physical-journal-v1":
            raise ValueError(f"unsupported journal version at row {line_number}")
        if not isinstance(wrapped.get("trajectory"), dict):
            raise ValueError(f"missing trajectory at journal row {line_number}")
        rows.append(wrapped)
    return rows


def validate(run_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    source = Path(manifest_path)
    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "COMPLETE":
        raise ValueError(f"Stage-A run status is {run_manifest.get('status')!r}, not COMPLETE")
    if run_manifest.get("phase") != "stage_a_screen":
        raise ValueError("validator accepts only stage_a_screen artifacts")
    if run_manifest.get("steering_outcomes") is not False:
        raise ValueError("steering outcome flag is not false")
    if run_manifest.get("confirmatory_accessed") is not False:
        raise ValueError("confirmatory access flag is not false")
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_hash = stable_digest(
        "Q1-V3-STAGE-A-MANIFEST", source.read_text(encoding="utf-8")
    )
    if run_manifest.get("source_manifest_hash") != source_hash:
        raise ValueError("source manifest hash mismatch")
    selected_keys = list(run_manifest.get("manifest_keys", []))
    manifests = payload.get("manifests", {})
    if not selected_keys or any(key not in manifests for key in selected_keys):
        raise ValueError("run manifest keys do not match source manifest")
    rollout_count = int(run_manifest.get("rollout_count", 0))
    expected_physical: set[tuple[str, int]] = set()
    for key in selected_keys:
        if not isinstance(manifests[key].get("items"), list):
            raise ValueError(f"manifest {key} has no items")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in selected_keys:
        record = manifests[key]
        groups[(str(record["family"]), str(record["cell"]))].append(key)
    for keys in groups.values():
        source_record = max(keys, key=lambda key: int(manifests[key]["reasoning_budget"]))
        for item in manifests[source_record]["items"]:
            for rollout_index in range(rollout_count):
                expected_physical.add((str(item["latent_id"]), rollout_index))

    journal_path = run / "physical_journal.jsonl"
    journal = _journal_rows(journal_path)
    journal_keys = {
        (str(row["trajectory"]["latent_id"]), int(row["trajectory"]["rollout_index"]))
        for row in journal
    }
    if journal_keys != expected_physical:
        raise ValueError(
            f"physical key mismatch: expected {len(expected_physical)}, got {len(journal_keys)}"
        )
    physical_ids = set()
    derived_keys = set()
    for wrapped in journal:
        trajectory = wrapped["trajectory"]
        source_tokens = tuple(int(token) for token in trajectory["source_token_ids"])
        if int(trajectory["source_max_budget"]) != 2048:
            raise ValueError("Stage-A physical source budget is not 2048")
        physical_ids.add(str(trajectory["physical_generation_id"]))
        derived_records = trajectory.get("derived_records", {})
        for manifest_key, row in derived_records.items():
            budget = int(row["generation_config"]["max_new_tokens"])
            scientific_key = (str(row["latent_id"]), int(row["rollout_index"]), budget)
            if scientific_key in derived_keys:
                raise ValueError(f"duplicate scientific key: {scientific_key}")
            derived_keys.add(scientific_key)
            if row["physical_generation_id"] != trajectory["physical_generation_id"]:
                raise ValueError("derived physical generation ID mismatch")
            if tuple(int(token) for token in row["token_ids"]) != source_tokens[
                : int(row["prefix_length"])
            ]:
                raise ValueError("derived row is not an exact source prefix")
            if manifest_key not in selected_keys:
                raise ValueError(f"unauthorized manifest key in journal: {manifest_key}")

    expected_scientific = (
        len(selected_keys)
        * len(next(iter(manifests.values()))["items"])
        * rollout_count
    )
    if len(derived_keys) != expected_scientific:
        raise ValueError(
            "scientific row count mismatch: "
            f"expected {expected_scientific}, got {len(derived_keys)}"
        )
    rows = _read_jsonl(run / "rollouts.jsonl")
    row_keys = {
        (
            str(row["latent_id"]),
            int(row["rollout_index"]),
            int(row["generation_config"]["max_new_tokens"]),
        )
        for row in rows
    }
    if row_keys != derived_keys or len(rows) != len(row_keys):
        raise ValueError("rollouts.jsonl scientific keys do not match physical journal")
    if int(run_manifest.get("physical_generation_count", -1)) != len(physical_ids):
        raise ValueError("physical generation count mismatch")
    if int(run_manifest.get("scientific_budget_outcomes", -1)) != len(rows):
        raise ValueError("scientific row count in manifest mismatch")
    return {
        "valid": True,
        "status": run_manifest["status"],
        "physical_generations": len(physical_ids),
        "scientific_rows": len(rows),
        "selected_manifest_keys": selected_keys,
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "journal_identity_hashes": sorted({row["identity_hash"] for row in journal}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate(args.run_dir, args.manifest)
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else "Q1 V3 Stage-A validation: PASS"
    )


if __name__ == "__main__":
    main()
