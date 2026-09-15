#!/usr/bin/env python3
"""Analyze a sealed structural finalization-timing artifact only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.finalization_timing import (  # noqa: E402
    directional_evalues,
    equal_family_cell_mean,
    paired_shortening_from_restricted_times,
)

CAP = 4096
DELTA = 0.10
THRESHOLD = 60.0
EXPECTED_RECORD_COUNT = 768


class StructuralAnalysisError(ValueError):
    """Raised when a structural seal cannot be analyzed as locked."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_seal(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuralAnalysisError(f"could not read structural seal: {path}") from exc
    if not isinstance(payload, dict):
        raise StructuralAnalysisError("structural seal must be an object")
    if payload.get("status") != "STRUCTURAL_TIMING_JOURNAL_SEALED":
        raise StructuralAnalysisError("structural journal is not sealed")
    if payload.get("close_token_id") != 151668:
        raise StructuralAnalysisError("structural seal has a different think-close token")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise StructuralAnalysisError("structural seal does not contain the 768-row locked design")
    if payload.get("logical_key_count") != EXPECTED_RECORD_COUNT:
        raise StructuralAnalysisError("structural seal logical key count is not 768")
    return payload


def analyze(seal_path: str | Path) -> dict[str, Any]:
    """Return e-value decisions using the structural seal and nothing else."""

    seal_file = Path(seal_path)
    seal = _read_seal(seal_file)
    contrasts = paired_shortening_from_restricted_times(seal["records"], cap=CAP)
    if len(contrasts) != 192:
        raise StructuralAnalysisError("structural seal does not contain 192 latent contrasts")
    evalues = directional_evalues(contrasts.values(), delta=DELTA)
    statuses = {
        "d75_earlier_by_delta": "TRIGGERED"
        if evalues["d75_earlier_by_delta"] >= THRESHOLD
        else "NOT_TRIGGERED",
        "d75_later_by_delta": "TRIGGERED"
        if evalues["d75_later_by_delta"] >= THRESHOLD
        else "NOT_TRIGGERED",
        "exclude_both_directions_by_delta": "TRIGGERED"
        if min(
            evalues["exclude_d75_earlier_by_delta"],
            evalues["exclude_d75_later_by_delta"],
        )
        >= THRESHOLD
        else "NOT_TRIGGERED",
    }
    if statuses["d75_earlier_by_delta"] == "TRIGGERED":
        classification = "D75_FINALIZATION_TIMING_EARLIER_SUPPORTED"
    elif statuses["d75_later_by_delta"] == "TRIGGERED":
        classification = "D75_FINALIZATION_TIMING_LATER_SUPPORTED"
    elif statuses["exclude_both_directions_by_delta"] == "TRIGGERED":
        classification = "D75_FINALIZATION_TIMING_RELEVANT_SHIFT_EXCLUDED"
    else:
        classification = "D75_FINALIZATION_TIMING_INCONCLUSIVE_AT_N192"
    return {
        "schema_version": "q1-finalization-timing-structural-analysis-v1",
        "status": "STRUCTURAL_TIMING_ANALYSIS_COMPLETE",
        "structural_seal_sha256": _sha256(seal_file),
        "cap": CAP,
        "delta": DELTA,
        "threshold": THRESHOLD,
        "latent_count": len(contrasts),
        "equal_family_cell_mean_shortening": equal_family_cell_mean(contrasts),
        "evalues": evalues,
        "statuses": statuses,
        "classification": classification,
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise StructuralAnalysisError(f"refusing to overwrite structural analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = analyze(args.seal)
    _write_new_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "classification": payload["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
