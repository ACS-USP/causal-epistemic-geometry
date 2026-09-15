#!/usr/bin/env python3
"""Freeze a materialized Q1 finalization-timing lock without model access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epistemic_geometry.benchmarks.reasoning.finalization_timing import (
    NAMESPACE,
    validate_manifest,
    validate_schedule,
)  # noqa: E402
from epistemic_geometry.benchmarks.reasoning.finalization_timing_journal import (  # noqa: E402
    identity_hash,
)
from epistemic_geometry.benchmarks.reasoning.finalization_timing_runner import (  # noqa: E402
    candidate_identity_hash,
    validate_candidate_identity,
)

ARTIFACTS = ("MANIFEST.json", "SCHEDULE.json", "PROVENANCE.json")
CODE = (
    "scripts/materialize_finalization_timing_prelock.py",
    "scripts/run_finalization_timing_collection.py",
    "scripts/seal_finalization_timing_structure.py",
    "scripts/analyze_finalization_timing_structural.py",
    "scripts/audit_finalization_timing_structural.py",
    "src/epistemic_geometry/analysis/finalization_timing.py",
    "src/epistemic_geometry/benchmarks/reasoning/finalization_timing.py",
    "src/epistemic_geometry/benchmarks/reasoning/finalization_timing_journal.py",
    "src/epistemic_geometry/benchmarks/reasoning/finalization_timing_runner.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen file: {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def freeze(
    artifacts: Path,
    candidate_path: Path,
    controller_path: Path,
    draft: Path,
    *,
    source_commit: str = "WORKTREE_UNCOMMITTED",
) -> dict[str, Any]:
    artifacts = artifacts.resolve()
    if any(not (artifacts / x).is_file() for x in ARTIFACTS):
        raise FileNotFoundError("materialized artifacts are incomplete")
    manifest, schedule, provenance = (read(artifacts / x) for x in ARTIFACTS)
    validate_manifest(manifest)
    validate_schedule(schedule, manifest)
    if provenance.get("namespace") != NAMESPACE:
        raise ValueError("provenance namespace mismatch")
    candidate = validate_candidate_identity(read(candidate_path))
    controller = read(controller_path)
    if not isinstance(controller, dict):
        raise ValueError("controller provenance must be an object")
    candidate_hash = candidate_identity_hash(candidate)
    hashes = {x: sha(artifacts / x) for x in ARTIFACTS}
    journal = {
        "experiment_id": NAMESPACE,
        "protocol_status": "FROZEN_NOT_RUN",
        "source_commit": source_commit,
        "candidate_identity_hash": candidate_hash,
        "manifest_hash": provenance["manifest_hash"],
        "schedule_digest": provenance["schedule_digest"],
        "manifest_sha256": hashes["MANIFEST.json"],
        "schedule_sha256": hashes["SCHEDULE.json"],
        "code_sha256": {x: sha(ROOT / x) for x in CODE},
    }
    lock = {
        "schema_version": "q1-finalization-timing-prelock-v1",
        "status": "FROZEN_NOT_RUN",
        "source_commit": source_commit,
        "draft_sha256": sha(draft),
        "no_new_qwen_outcomes_before_lock": True,
        "artifact_sha256": hashes,
        "prelock_artifacts": provenance,
        "candidate_identity": candidate,
        "candidate_identity_hash": candidate_hash,
        "controller_provenance": controller,
        "journal_identity": journal,
        "journal_identity_hash": identity_hash(journal),
        "design": {
            "N": len(manifest),
            "calls": len(schedule),
            "caps": [4096],
            "conditions": ["BASELINE", "D75"],
            "rollouts_per_cell": 2,
            "no_adaptive_retries": True,
            "only_missing_schedule_keys_may_resume": True,
        },
        "decision_rule": {
            "delta": 0.1,
            "threshold": 60.0,
            "candidate_closes_after_one_sealed_analysis": True,
            "structural_endpoint": "first_think_close_restricted_at_4096",
        },
    }
    write_new(artifacts / "CANDIDATE_IDENTITY.json", candidate)
    write_new(artifacts / "LOCK.json", lock)
    return lock


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--controller", required=True, type=Path)
    p.add_argument(
        "--draft", default=ROOT / "review/q1_finalization_timing/PRELOCK_DRAFT.md", type=Path
    )
    p.add_argument("--source-commit", default="WORKTREE_UNCOMMITTED")
    a = p.parse_args(argv)
    freeze(a.artifacts, a.candidate, a.controller, a.draft, source_commit=a.source_commit)
    print("LOCK_FREEZE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
