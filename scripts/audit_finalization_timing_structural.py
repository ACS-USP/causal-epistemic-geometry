#!/usr/bin/env python3
"""Independently recompute structural D75 finalization-timing analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

CAP = 4096
DELTA = 0.10
THRESHOLD = 60.0
EXPECTED_RECORD_COUNT = 768
REQUIRED_RECORD_FIELDS = {
    "family",
    "cell",
    "latent",
    "condition",
    "rollout",
    "cap",
    "restricted_think_close_time",
    "think_close_observed",
}


class StructuralAuditError(ValueError):
    """Raised when a structural timing analysis cannot be independently verified."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuralAuditError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise StructuralAuditError(f"{label} must be an object")
    return value


def _contrasts(records: list[Any]) -> tuple[dict[tuple[str, str, str], float], float]:
    if len(records) != EXPECTED_RECORD_COUNT:
        raise StructuralAuditError("seal does not contain 768 structural records")
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, str, str, str, int]] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != REQUIRED_RECORD_FIELDS:
            raise StructuralAuditError("structural record schema mismatch")
        family, cell, latent = record["family"], record["cell"], record["latent"]
        condition, rollout = record["condition"], record["rollout"]
        if not all(isinstance(value, str) and value for value in (family, cell, latent)):
            raise StructuralAuditError("structural record identifiers are invalid")
        if condition not in {"BASELINE", "D75"} or rollout not in {0, 1}:
            raise StructuralAuditError("structural record condition or rollout is invalid")
        if record["cap"] != CAP or type(record["restricted_think_close_time"]) is not int:
            raise StructuralAuditError("structural record cap or time is invalid")
        if not 1 <= record["restricted_think_close_time"] <= CAP:
            raise StructuralAuditError("structural restricted time is outside the horizon")
        if type(record["think_close_observed"]) is not bool:
            raise StructuralAuditError("structural event flag is invalid")
        key = (family, cell, latent, condition, rollout)
        if key in keys:
            raise StructuralAuditError("duplicate structural record")
        keys.add(key)
        groups[(family, cell, latent, condition)].append(record)
    units = {(family, cell, latent) for family, cell, latent, _ in groups}
    contrast: dict[tuple[str, str, str], float] = {}
    for family, cell, latent in units:
        base = sorted(
            groups.get((family, cell, latent, "BASELINE"), []), key=lambda row: row["rollout"]
        )
        d75 = sorted(
            groups.get((family, cell, latent, "D75"), []), key=lambda row: row["rollout"]
        )
        if len(base) != 2 or len(d75) != 2:
            raise StructuralAuditError("incomplete structural condition pair")
        if [row["rollout"] for row in base] != [0, 1] or [row["rollout"] for row in d75] != [0, 1]:
            raise StructuralAuditError("incomplete structural rollout pair")
        contrast[(family, cell, latent)] = sum(
            (first["restricted_think_close_time"] - second["restricted_think_close_time"]) / CAP
            for first, second in zip(base, d75, strict=True)
        ) / 2.0
    if len(contrast) != 192:
        raise StructuralAuditError("seal does not contain 192 latent contrasts")
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (family, cell, _latent), value in contrast.items():
        by_cell[(family, cell)].append(value)
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _cell), values in by_cell.items():
        if len(values) != 16:
            raise StructuralAuditError("cell does not contain 16 latent contrasts")
        by_family[family].append(sum(values) / len(values))
    if len(by_family) != 3 or any(len(values) != 4 for values in by_family.values()):
        raise StructuralAuditError("family/cell timing structure does not match locked design")
    mean = sum(sum(values) / len(values) for values in by_family.values()) / len(by_family)
    return contrast, mean


def _evalues(contrast: dict[tuple[str, str, str], float]) -> dict[str, float]:
    values = tuple(contrast.values())
    return {
        "d75_earlier_by_delta": math.prod(1.0 + (value - DELTA) / 2.0 for value in values),
        "d75_later_by_delta": math.prod(1.0 + (-value - DELTA) / 2.0 for value in values),
        "exclude_d75_earlier_by_delta": math.prod(1.0 + (DELTA - value) / 2.0 for value in values),
        "exclude_d75_later_by_delta": math.prod(1.0 + (DELTA + value) / 2.0 for value in values),
    }


def audit(seal_path: str | Path, analysis_path: str | Path) -> dict[str, Any]:
    """Audit all numeric and categorical fields of one structural analysis."""

    seal_file, analysis_file = Path(seal_path), Path(analysis_path)
    seal = _read(seal_file, "structural seal")
    analysis = _read(analysis_file, "structural analysis")
    if seal.get("status") != "STRUCTURAL_TIMING_JOURNAL_SEALED":
        raise StructuralAuditError("structural seal is not sealed")
    if analysis.get("structural_seal_sha256") != _sha256(seal_file):
        raise StructuralAuditError("analysis does not bind the supplied structural seal")
    contrast, mean = _contrasts(seal.get("records", []))
    evalues = _evalues(contrast)
    expected_statuses = {
        "d75_earlier_by_delta": "TRIGGERED"
        if evalues["d75_earlier_by_delta"] >= THRESHOLD
        else "NOT_TRIGGERED",
        "d75_later_by_delta": "TRIGGERED"
        if evalues["d75_later_by_delta"] >= THRESHOLD
        else "NOT_TRIGGERED",
        "exclude_both_directions_by_delta": "TRIGGERED"
        if min(evalues["exclude_d75_earlier_by_delta"], evalues["exclude_d75_later_by_delta"])
        >= THRESHOLD
        else "NOT_TRIGGERED",
    }
    expected_classification = (
        "D75_FINALIZATION_TIMING_EARLIER_SUPPORTED"
        if expected_statuses["d75_earlier_by_delta"] == "TRIGGERED"
        else "D75_FINALIZATION_TIMING_LATER_SUPPORTED"
        if expected_statuses["d75_later_by_delta"] == "TRIGGERED"
        else "D75_FINALIZATION_TIMING_RELEVANT_SHIFT_EXCLUDED"
        if expected_statuses["exclude_both_directions_by_delta"] == "TRIGGERED"
        else "D75_FINALIZATION_TIMING_INCONCLUSIVE_AT_N192"
    )
    comparisons = {
        "mean": (analysis.get("equal_family_cell_mean_shortening"), mean),
        **{name: (analysis.get("evalues", {}).get(name), value) for name, value in evalues.items()},
    }
    differences = {
        name: abs(float(observed) - expected) for name, (observed, expected) in comparisons.items()
    }
    if any(value > 1e-12 for value in differences.values()):
        raise StructuralAuditError("numeric structural analysis mismatch")
    if analysis.get("statuses") != expected_statuses:
        raise StructuralAuditError("structural analysis status mismatch")
    if analysis.get("classification") != expected_classification:
        raise StructuralAuditError("structural analysis classification mismatch")
    return {
        "schema_version": "q1-finalization-timing-structural-audit-v1",
        "classification": "STRUCTURAL_TIMING_ANALYSIS_INDEPENDENT_AUDIT_PASS",
        "structural_seal_sha256": _sha256(seal_file),
        "analysis_sha256": _sha256(analysis_file),
        "max_absolute_difference": max(differences.values(), default=0.0),
        "latent_count": len(contrast),
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise StructuralAuditError(f"refusing to overwrite structural audit: {path}")
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
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(args.seal, args.analysis)
    _write_new_json(args.output, payload)
    print(json.dumps({"classification": payload["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
