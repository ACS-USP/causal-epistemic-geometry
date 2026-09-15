#!/usr/bin/env python3
"""Run the frozen Q1 budget-by-D75 analysis after seal and independent audit."""

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

from epistemic_geometry.analysis.budget_interaction import (  # noqa: E402
    aggregate_outcomes,
    fixed_rule_prelock_decisions,
    stratified_latent_bootstrap,
)
from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import (  # noqa: E402
    BudgetInteractionJournal,
    physical_key,
)
from scripts.run_budget_interaction_collection import load_and_validate_lock  # noqa: E402


class SealedAnalysisError(ValueError):
    """Raised when sealed data cannot be analyzed under the frozen protocol."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SealedAnalysisError(f"cannot read {path}") from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedAnalysisError(f"cannot parse {label}") from exc
    if not isinstance(value, dict):
        raise SealedAnalysisError(f"{label} must be a JSON object")
    return value


def _validate_seal_chain(
    lock_path: Path,
    journal_path: Path,
    raw_seal_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = _read_object(raw_seal_path, "raw seal")
    audit = _read_object(audit_path, "independent raw-seal audit")
    if seal.get("status") != "RAW_JOURNAL_SEALED":
        raise SealedAnalysisError("journal is not raw sealed")
    if audit.get("classification") != "RAW_SEAL_INDEPENDENT_AUDIT_PASS":
        raise SealedAnalysisError("independent raw-seal audit did not pass")
    if seal.get("lock_sha256") != _sha256(lock_path):
        raise SealedAnalysisError("raw seal lock hash mismatch")
    if seal.get("journal_sha256") != _sha256(journal_path):
        raise SealedAnalysisError("raw seal journal hash mismatch")
    if audit.get("primary_raw_seal_sha256") != _sha256(raw_seal_path):
        raise SealedAnalysisError("audit primary-seal hash mismatch")
    if audit.get("journal_sha256") != seal.get("journal_sha256"):
        raise SealedAnalysisError("audit journal hash mismatch")
    if seal.get("logical_key_count") != 768 or audit.get("logical_key_count") != 768:
        raise SealedAnalysisError("seal chain does not establish 768 journal keys")
    return seal, audit


def _observations_from_journal(
    *,
    loaded: Mapping[str, Any],
    journal_path: Path,
) -> tuple[
    list[dict[str, Any]], dict[tuple[str, str], tuple[str, ...]], dict[str, tuple[str, ...]]
]:
    schedule = loaded["schedule"]
    identity = loaded["journal_identity"]
    journal = BudgetInteractionJournal(journal_path, identity=identity)
    expected = {physical_key(row): row for row in schedule}
    if set(journal.entries) != set(expected):
        raise SealedAnalysisError("sealed journal keys do not match frozen schedule")

    observations: list[dict[str, Any]] = []
    latents: dict[tuple[str, str], list[str]] = {}
    cells: dict[str, list[str]] = {}
    for key in sorted(expected):
        row = expected[key]
        record = journal.entries[key].get("record")
        if not isinstance(record, Mapping):
            raise SealedAnalysisError("sealed journal record is not an object")
        family = row["family"]
        cell = row["cell"]
        latent_id = row["latent_id"]
        latents.setdefault((family, cell), []).append(latent_id)
        cells.setdefault(family, []).append(cell)
        observations.append(
            {
                "surface": row["surface"],
                "family": family,
                "cell": cell,
                "latent": latent_id,
                "cap": row["cap"],
                "condition": row["condition"],
                "rollout": row["rollout_index"],
                "correct": record.get("correct"),
                "parse_status": record.get("parse_status"),
                "stop_reason": record.get("stop_reason"),
            }
        )
    expected_latents = {
        stratum: tuple(sorted(set(ids))) for stratum, ids in sorted(latents.items())
    }
    canonical_cells = {family: tuple(sorted(set(names))) for family, names in sorted(cells.items())}
    if len(observations) != 768 or sum(map(len, expected_latents.values())) != 96:
        raise SealedAnalysisError("sealed schedule does not contain the 96-latent, 768-row design")
    return observations, expected_latents, canonical_cells


def _public_analysis_summary(
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "category_rates": aggregate["category_rates"],
        "delta_correct": aggregate["delta_correct"],
        "interaction": aggregate["interaction"],
        "descriptive_bootstrap": {
            "method": bootstrap["method"],
            "n_resamples": bootstrap["n_resamples"],
            "seed": bootstrap["seed"],
            "delta_correct_percentile_95": bootstrap["delta_correct_percentile_95"],
            "interaction_percentile_95": bootstrap["interaction_percentile_95"],
        },
        "fixed_prelock_rule": {
            "method": decisions["method"],
            "delta": decisions["delta"],
            "threshold": decisions["threshold"],
            "n_latents": decisions["n_latents"],
            "weighted_delta": decisions["weighted_delta"],
            "weighted_interaction": decisions["weighted_interaction"],
            "decisions": decisions["decisions"],
            "statuses": decisions["statuses"],
        },
    }


def analyze_sealed_journal(
    lock_path: str | Path,
    journal_path: str | Path,
    raw_seal_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    """Run only the analysis pinned by the frozen prelock implementation."""
    lock_file = Path(lock_path).expanduser().resolve()
    journal_file = Path(journal_path).expanduser().resolve()
    seal_file = Path(raw_seal_path).expanduser().resolve()
    audit_file = Path(audit_path).expanduser().resolve()
    seal, audit = _validate_seal_chain(lock_file, journal_file, seal_file, audit_file)
    loaded = load_and_validate_lock(lock_file)
    expected_analysis_hash = (
        loaded["lock"].get("decision_rule", {}).get("decision_implementation_sha256")
    )
    analysis_file = ROOT / "src/epistemic_geometry/analysis/budget_interaction.py"
    if expected_analysis_hash != _sha256(analysis_file):
        raise SealedAnalysisError("frozen decision implementation hash mismatch")
    observations, expected_latents, canonical_cells = _observations_from_journal(
        loaded=loaded,
        journal_path=journal_file,
    )
    aggregate = aggregate_outcomes(
        observations,
        expected_latents=expected_latents,
        canonical_cells=canonical_cells,
    )
    bootstrap = stratified_latent_bootstrap(
        observations,
        n_resamples=2_000,
        seed=0,
        expected_latents=expected_latents,
        canonical_cells=canonical_cells,
    )
    decisions = fixed_rule_prelock_decisions(
        observations,
        expected_latents=expected_latents,
        canonical_cells=canonical_cells,
        delta=loaded["lock"]["decision_rule"]["delta"],
        threshold=loaded["lock"]["decision_rule"]["threshold"],
    )
    return {
        "schema_version": "q1-v3-budget-interaction-sealed-analysis-v1",
        "status": "Q1_BUDGET_INTERACTION_ANALYSIS_COMPLETE",
        "raw_seal_sha256": _sha256(seal_file),
        "raw_seal_independent_audit_sha256": _sha256(audit_file),
        "journal_sha256": seal["journal_sha256"],
        "lock_sha256": seal["lock_sha256"],
        "analysis_implementation_sha256": expected_analysis_hash,
        "observation_count": len(observations),
        "latent_count": 96,
        "analysis": _public_analysis_summary(aggregate, bootstrap, decisions),
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise SealedAnalysisError(f"refusing to overwrite existing analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        raise SealedAnalysisError("cannot write analysis") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a sealed Q1 budget-interaction journal")
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "review/q1_budget_causal_interaction/PRELOCK_ARTIFACTS/LOCK.json",
    )
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--raw-seal", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = analyze_sealed_journal(args.lock, args.journal, args.raw_seal, args.audit)
    _write_immutable_json(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "observation_count": payload["observation_count"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
