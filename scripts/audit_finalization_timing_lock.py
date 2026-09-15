#!/usr/bin/env python3
"""Independently audit a frozen finalization-timing lock without model access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(lock_path: Path) -> dict:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "FROZEN_NOT_RUN":
        raise ValueError("lock status must be FROZEN_NOT_RUN")
    artifact = lock.get("artifact_sha256")
    if not isinstance(artifact, dict) or set(artifact) != {
        "MANIFEST.json",
        "SCHEDULE.json",
        "PROVENANCE.json",
    }:
        raise ValueError("lock must pin exactly the three materialized artifacts")
    for name, expected in artifact.items():
        if sha(lock_path.parent / name) != expected:
            raise ValueError(f"artifact digest mismatch: {name}")
    manifest = json.loads((lock_path.parent / "MANIFEST.json").read_text())
    schedule = json.loads((lock_path.parent / "SCHEDULE.json").read_text())
    candidate = lock.get("candidate_identity", {})
    if candidate.get("think_close_token_id") != 151668:
        raise ValueError("structural close token is not pinned")
    if lock.get("design", {}).get("calls") != 768 or len(manifest) != 192 or len(schedule) != 768:
        raise ValueError("frozen design counts are inconsistent")
    if {x.get("cap") for x in schedule} != {4096} or {x.get("condition") for x in schedule} != {
        "BASELINE",
        "D75",
    }:
        raise ValueError("frozen schedule conditions or cap are inconsistent")
    if any(
        any(k in row for k in ("correct", "target", "raw_text", "parsed"))
        for row in manifest + schedule
    ):
        raise ValueError("prelock unexpectedly contains semantic outcomes")
    return {
        "status": "FINALIZATION_TIMING_LOCK_AUDIT_PASS",
        "manifest_rows": len(manifest),
        "schedule_rows": len(schedule),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("lock", type=Path)
    a = p.parse_args(argv)
    print(json.dumps(audit(a.lock), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
