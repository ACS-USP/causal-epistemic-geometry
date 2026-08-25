#!/usr/bin/env python3
"""Verify and hash the design-only Q2 geometry-foundations bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_geometry_foundations"

REQUIRED = (
    "REPORT.md",
    "MATHEMATICAL_AUDIT.json",
    "SYNTHETIC_VALIDATION.json",
    "SHELL_IDENTIFIABILITY_SIMULATION.json",
    "Q2_V3_RADIAL_ANGULAR_PROTOCOL_DRAFT.md",
    "Q2_V3_REVISED_POWER_PRECISION_COST_PLAN.md",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    missing = [name for name in REQUIRED if not (REVIEW / name).is_file()]
    if missing:
        raise RuntimeError(f"Q2 geometry-foundations bundle incomplete: {missing}")
    mathematical = json.loads((REVIEW / "MATHEMATICAL_AUDIT.json").read_text())
    synthetic = json.loads((REVIEW / "SYNTHETIC_VALIDATION.json").read_text())
    if mathematical["semantic_outcomes_read"]:
        raise RuntimeError("mathematical audit crossed the scientific firewall")
    if not synthetic["checks"]["pass"]:
        raise RuntimeError("synthetic geometry validation did not pass")
    audit = {
        "schema_version": "q2-geometry-foundations-review-audit-v1",
        "classification": "Q2_GEOMETRY_FOUNDATIONS_REPRODUCIBLE",
        "frozen_q2_v2_classification": "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL",
        "frozen_q2_v2_forensic": "Q2_V2_FORENSIC_CLEAN",
        "frozen_q2_v2_changed": False,
        "semantic_outcomes_read": False,
        "new_scientific_rows": 0,
        "model_inference": False,
        "runpod_or_dgx_used": False,
        "m3_status": "NOT_QUALIFIED",
        "q2_v3": "DRAFT_AWAITING_PRINCIPAL_RESEARCHER_FREEZE_NOT_RUN",
        "q3": "NOT_RUN",
    }
    (REVIEW / "REVIEW_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    internal = sorted(
        path
        for path in REVIEW.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    )
    external = [
        ROOT / "docs/Q2_GEOMETRY_FOUNDATIONS.md",
        ROOT / "docs/Q2_M3_FEASIBILITY_AND_QUALIFICATION_PLAN.md",
        ROOT / "experiments/specs/q2_v3_out_of_bank_finite_secant.DRAFT.yaml",
        ROOT / "scripts/analyze_q2_geometry_foundations.py",
        ROOT / "scripts/finalize_q2_geometry_foundations.py",
        ROOT / "src/epistemic_geometry/analysis/control_geometry.py",
        ROOT / "tests/test_control_geometry.py",
    ]
    payload = {
        "schema_version": "q2-geometry-foundations-bundle-v1",
        "status": "DESIGN_ONLY_COMPLETE_Q2_V3_NOT_FROZEN_NOT_RUN",
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in [*internal, *external]
        },
    }
    (REVIEW / "artifact_hashes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Q2 geometry-foundations bundle: {len(payload['files'])} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
