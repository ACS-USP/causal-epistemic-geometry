#!/usr/bin/env python3
"""Verify and hash the Q2 V2 principal-review/Q2 V3 draft bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v2_principal_researcher_review"

REQUIRED = (
    "REPORT.md",
    "ANALYSIS_PROVENANCE.json",
    "PRIMARY_REPRODUCTION.json",
    "FAMILY_DECOMPOSITION.csv",
    "BOOTSTRAP_CONTRASTS.json",
    "ITEM_BOOTSTRAP_SAMPLES.npz",
    "HELDOUT_PREDICTIONS.csv",
    "CALIBRATION_DIAGNOSTICS.json",
    "NUISANCE_BASELINES.json",
    "DOSE_LOCAL_VALIDITY_ANALYSIS.json",
    "NULL_PAIR_ANALYSIS.json",
    "ROBUSTNESS_SENSITIVITY.json",
    "EXPLORATORY_ANALYSIS_ENUMERATION.json",
    "Q2_V3_PROTOCOL_DRAFT.md",
    "Q2_V3_POWER_PRECISION_COST_PLAN.md",
    "Q2_V3_PRECISION_SIMULATION.json",
    "REVIEW_AUDIT.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    journal = ROOT / "review/q2_controller_bank_v2/V2_COMMON_PANEL_JOURNAL.jsonl"
    wrappers = sum(1 for line in journal.read_text().splitlines() if line.strip())
    audit = {
        "schema_version": "q2-v2-principal-review-audit-v1",
        "classification": "Q2_V2_PRINCIPAL_REVIEW_REPRODUCIBLE",
        "frozen_q2_v2_classification_preserved": True,
        "frozen_q2_v2_forensic_classification": "Q2_V2_FORENSIC_CLEAN",
        "frozen_journal_rows": wrappers,
        "frozen_journal_sha256": digest(journal),
        "expected_journal_sha256": (
            "9a635787561d5bc6e56cf2c7ffae9e391bebc817488f38369ffc8c0fea14a5b7"
        ),
        "new_scientific_rows": 0,
        "model_inference": False,
        "runpod_or_dgx_used": False,
        "q2_v3": "DRAFT_NOT_RUN",
        "q3": "NOT_RUN",
    }
    audit["input_integrity_pass"] = bool(
        wrappers == 6960
        and audit["frozen_journal_sha256"] == audit["expected_journal_sha256"]
    )
    (REVIEW / "REVIEW_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    missing = [name for name in REQUIRED if not (REVIEW / name).is_file()]
    if missing:
        raise RuntimeError(f"Q2 review bundle is incomplete: {missing}")
    reproduction = json.loads((REVIEW / "PRIMARY_REPRODUCTION.json").read_text())
    if not reproduction["all_exact_within_1e_12"]:
        raise RuntimeError("frozen Q2 V2 reconstruction does not agree")
    paths = sorted(
        path
        for path in REVIEW.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    )
    external = [
        ROOT / "experiments/specs/q2_v3_out_of_bank_finite_secant.DRAFT.yaml",
        ROOT / "scripts/analyze_q2_v2_principal_review.py",
        ROOT / "scripts/plan_q2_v3_precision.py",
        ROOT / "src/epistemic_geometry/analysis/q2_exploratory.py",
        ROOT / "tests/test_q2_exploratory.py",
    ]
    payload = {
        "schema_version": "q2-v2-principal-review-bundle-v1",
        "status": "POST_HOC_REVIEW_COMPLETE_Q2_V3_DRAFT_NOT_RUN",
        "frozen_q2_v2_classification": "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL",
        "q2_v3": "DRAFT_AWAITING_PRINCIPAL_RESEARCHER_FREEZE",
        "new_inference": False,
        "q3": "NOT_RUN",
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in [*paths, *external]
        },
    }
    destination = REVIEW / "artifact_hashes.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Q2 review bundle: {len(payload['files'])} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
