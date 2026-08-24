#!/usr/bin/env python3
"""Bind public materialized manifests and an allowlisted source bundle before output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_heldout_geometry"
FILENAMES = {
    "SOURCE_CONSTRUCTION": "SOURCE_CONSTRUCTION_MANIFEST.json",
    "SOURCE_VALIDATION": "SOURCE_VALIDATION_MANIFEST.json",
    "MANIPULATION_QUALIFICATION": "MANIPULATION_MANIFEST.json",
    "COVARIANCE_POOL": "COVARIANCE_MANIFEST.json",
    "FINITE_SECANT_PROBES": "FINITE_SECANT_PROBE_MANIFEST.json",
    "COMMON_PANEL": "DEVELOPMENT_PANEL_MANIFEST.json",
}
SOURCE_FILES = (
    "scripts/run_q2_controller_heldout.py",
    "scripts/run_gate6_2_first_stage_repair.py",
    "scripts/run_gate11_domain_conditioned_control.py",
    "scripts/materialize_q2_public_manifests.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_tree_hashes() -> dict[str, str]:
    records = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    for path in sorted((ROOT / "src/epistemic_geometry").rglob("*.py")):
        records[str(path.relative_to(ROOT))] = sha256(path)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    lock_path = review / "CANDIDATE_PROTOCOL_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    materialization = json.loads(
        (review / "PUBLIC_DATASET_MATERIALIZATION.json").read_text(encoding="utf-8")
    )
    if materialization["classification"] != "Q2_PUBLIC_DATASET_MATERIALIZATION_PASS":
        raise RuntimeError("public dataset materialization did not pass")
    for role, filename in FILENAMES.items():
        rows = json.loads((review / filename).read_text(encoding="utf-8"))
        expected_ids = json.loads(
            (review / "REMOTE_PUBLIC_DATASET_ALLOCATION.json").read_text(encoding="utf-8")
        )["allocations"][role]
        if [row["item_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"materialized ID order differs for {role}")
        lock["allocations"][role] = {
            "n": len(rows),
            "file": filename,
            "file_sha256": sha256(review / filename),
            "content_source": "pinned public dataset materialized on execution host",
        }
    bundle_hashes = source_tree_hashes()
    write_json(review / "SOURCE_BUNDLE_HASHES.json", bundle_hashes)
    (review / "EXECUTION_SOURCE_COMMIT.txt").write_text(
        args.source_commit + "\n", encoding="utf-8"
    )
    lock["source_commit_at_preparation"] = args.source_commit
    lock["public_dataset_materialization"] = {
        "record_sha256": sha256(review / "PUBLIC_DATASET_MATERIALIZATION.json"),
        "completed_before_model_outputs": True,
        "model_outputs_existing": False,
    }
    lock["execution_checkout"] = {
        "source_commit": args.source_commit,
        "transport": "remote-safe allowlisted bundle; Git egress unavailable",
        "source_bundle_hashes_file": "SOURCE_BUNDLE_HASHES.json",
        "source_bundle_hashes_sha256": sha256(review / "SOURCE_BUNDLE_HASHES.json"),
        "git_history_transferred": False,
        "scientific_semantics_changed": False,
    }
    write_json(lock_path, lock)
    write_json(
        review / "REMOTE_SAFE_TRANSPORT_INCIDENT.json",
        {
            "classification": "CLASS_A_GIT_EGRESS_PROTECTION",
            "timing": "before all Q2 model outputs",
            "resolution": "allowlisted source files plus public-dataset materialization",
            "source_commit": args.source_commit,
            "source_bundle_hashes_sha256": sha256(review / "SOURCE_BUNDLE_HASHES.json"),
            "raw_historical_outputs_transferred": False,
            "scientific_design_changed": False,
        },
    )
    print(
        json.dumps(
            {
                "classification": "Q2_EXECUTION_BUNDLE_BOUND",
                "source_commit": args.source_commit,
                "manifest_count": len(FILENAMES),
                "source_file_count": len(bundle_hashes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
