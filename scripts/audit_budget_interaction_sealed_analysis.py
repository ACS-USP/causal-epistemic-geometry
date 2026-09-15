#!/usr/bin/env python3
"""Independently recompute the sealed Q1 budget-by-D75 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPS = (2048, 4096)
CONDITIONS = ("BASELINE", "D75")
CATEGORIES = ("correct", "valid_wrong", "incomplete", "invalid")
INCOMPLETE = {"MISSING_FINAL", "THINKING_UNCLOSED", "TRUNCATED_NO_FINAL"}
VALID_STATUSES = {"OK", *INCOMPLETE, "INVALID_FINAL"}


class SealedAnalysisAuditError(ValueError):
    """Raised when a sealed-analysis audit cannot reproduce the report."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SealedAnalysisAuditError(f"cannot read {path}") from exc


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedAnalysisAuditError(f"cannot parse {label}") from exc
    if not isinstance(value, dict):
        raise SealedAnalysisAuditError(f"{label} must be an object")
    return value


def _category(correct: Any, status: Any) -> str:
    if type(correct) is not bool or not isinstance(status, str) or status not in VALID_STATUSES:
        raise SealedAnalysisAuditError("journal outcome fields are invalid")
    if correct:
        if status != "OK":
            raise SealedAnalysisAuditError("correct journal outcome lacks OK status")
        return "correct"
    if status == "OK":
        return "valid_wrong"
    if status in INCOMPLETE:
        return "incomplete"
    return "invalid"


def _physical_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    try:
        return (
            str(row["latent_id"]),
            int(row["cap"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SealedAnalysisAuditError("invalid physical schedule key") from exc


def _load_rows(lock_path: Path, journal_path: Path) -> list[dict[str, Any]]:
    lock = _json_object(lock_path, "lock")
    schedule_path = lock_path.parent / "SCHEDULE.json"
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedAnalysisAuditError("cannot parse schedule") from exc
    if not isinstance(schedule, list) or len(schedule) != 768:
        raise SealedAnalysisAuditError("schedule does not contain 768 rows")
    expected = {_physical_key(row): row for row in schedule if isinstance(row, Mapping)}
    if len(expected) != 768:
        raise SealedAnalysisAuditError("schedule has invalid or duplicate keys")
    try:
        lines = journal_path.read_bytes().splitlines()
    except OSError as exc:
        raise SealedAnalysisAuditError("cannot read journal") from exc
    if len(lines) != 768:
        raise SealedAnalysisAuditError("journal does not have 768 persisted lines")
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, int, str, int]] = set()
    for index, line in enumerate(lines, start=1):
        try:
            wrapped = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SealedAnalysisAuditError(f"journal line {index} is invalid JSON") from exc
        if not isinstance(wrapped, Mapping) or not isinstance(wrapped.get("schedule"), Mapping):
            raise SealedAnalysisAuditError(f"journal line {index} lacks a schedule")
        schedule_row = wrapped["schedule"]
        key = _physical_key(schedule_row)
        if key in observed or expected.get(key) != schedule_row:
            raise SealedAnalysisAuditError("journal schedule does not match frozen schedule")
        observed.add(key)
        record = wrapped.get("record")
        if not isinstance(record, Mapping):
            raise SealedAnalysisAuditError("journal record is invalid")
        category = _category(record.get("correct"), record.get("parse_status"))
        rows.append(
            {
                "family": schedule_row["family"],
                "cell": schedule_row["cell"],
                "latent": schedule_row["latent_id"],
                "cap": schedule_row["cap"],
                "condition": schedule_row["condition"],
                "rollout": schedule_row["rollout_index"],
                "category": category,
            }
        )
    if observed != set(expected):
        raise SealedAnalysisAuditError("journal key coverage does not match schedule")
    if lock.get("design", {}).get("calls") != 768:
        raise SealedAnalysisAuditError("lock does not declare the 768-call design")
    return rows


def _latent_category_rates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, int, str], dict[str, float]],
    dict[tuple[str, str], tuple[str, ...]],
]:
    grouped: dict[tuple[str, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    latents: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["family"], row["cell"], row["latent"], row["cap"], row["condition"])
        grouped[key].append(row)
        latents[(row["family"], row["cell"])].add(row["latent"])
    result: dict[tuple[str, str, str, int, str], dict[str, float]] = {}
    for key, items in grouped.items():
        if len(items) != 2 or {item["rollout"] for item in items} != {0, 1}:
            raise SealedAnalysisAuditError("factorial rollout coverage is invalid")
        result[key] = {
            category: sum(item["category"] == category for item in items) / 2.0
            for category in CATEGORIES
        }
    if len(result) != 384:
        raise SealedAnalysisAuditError("factorial latent-condition coverage is invalid")
    return result, {stratum: tuple(sorted(ids)) for stratum, ids in sorted(latents.items())}


def _aggregate(
    latent_rates: Mapping[tuple[str, str, str, int, str], Mapping[str, float]],
    latents_by_stratum: Mapping[tuple[str, str], Sequence[str]],
    sampled: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    strata = sorted(latents_by_stratum)
    families = sorted({family for family, _cell in strata})
    cells_by_family = {
        family: tuple(cell for fam, cell in strata if fam == family) for family in families
    }
    grid: dict[int, dict[str, dict[str, float]]] = {}
    for cap in CAPS:
        grid[cap] = {}
        for condition in CONDITIONS:
            family_values: dict[str, dict[str, float]] = {}
            for family in families:
                cell_values: list[dict[str, float]] = []
                for cell in cells_by_family[family]:
                    ids = (
                        tuple(sampled[(family, cell)])
                        if sampled is not None
                        else latents_by_stratum[(family, cell)]
                    )
                    if not ids:
                        raise SealedAnalysisAuditError("empty latent stratum")
                    cell_values.append(
                        {
                            category: float(
                                np.mean(
                                    [
                                        latent_rates[(family, cell, latent, cap, condition)][
                                            category
                                        ]
                                        for latent in ids
                                    ]
                                )
                            )
                            for category in CATEGORIES
                        }
                    )
                family_values[family] = {
                    category: float(np.mean([cell[category] for cell in cell_values]))
                    for category in CATEGORIES
                }
            grid[cap][condition] = {
                category: float(np.mean([family_values[family][category] for family in families]))
                for category in CATEGORIES
            }
    return grid


def _weighted_contrasts(
    latent_rates: Mapping[tuple[str, str, str, int, str], Mapping[str, float]],
    latents_by_stratum: Mapping[tuple[str, str], Sequence[str]],
) -> tuple[
    dict[int, dict[tuple[str, str, str], float]],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
]:
    strata = sorted(latents_by_stratum)
    families = sorted({family for family, _cell in strata})
    cells_by_family = {
        family: tuple(cell for fam, cell in strata if fam == family) for family in families
    }
    weights: dict[tuple[str, str, str], float] = {}
    for family in families:
        for cell in cells_by_family[family]:
            ids = latents_by_stratum[(family, cell)]
            for latent in ids:
                weights[(family, cell, latent)] = (
                    1.0 / len(families) / len(cells_by_family[family]) / len(ids)
                )
    contrasts = {
        cap: {
            key: float(
                latent_rates[(*key, cap, "D75")]["correct"]
                - latent_rates[(*key, cap, "BASELINE")]["correct"]
            )
            for key in sorted(weights)
        }
        for cap in CAPS
    }
    interaction = {key: contrasts[4096][key] - contrasts[2048][key] for key in sorted(weights)}
    return contrasts, interaction, weights


def _evalue(
    values: Mapping[tuple[str, str, str], float],
    weights: Mapping[tuple[str, str, str], float],
    delta: float,
    direction: int,
) -> float:
    n = len(values)
    a = {key: n * weights[key] for key in values}
    c = 1.0 / (2.0 * max(a.values()))
    return float(
        prod(1.0 + direction * c * a[key] * (values[key] - delta) for key in sorted(values))
    )


def _bootstrap_summary(
    latent_rates: Mapping[tuple[str, str, str, int, str], Mapping[str, float]],
    latents_by_stratum: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    deltas = {cap: [] for cap in CAPS}
    interactions: list[float] = []
    for _ in range(2_000):
        sampled = {
            stratum: tuple(rng.choice(ids, size=len(ids), replace=True).tolist())
            for stratum, ids in sorted(latents_by_stratum.items())
        }
        grid = _aggregate(latent_rates, latents_by_stratum, sampled)
        current = {
            cap: grid[cap]["D75"]["correct"] - grid[cap]["BASELINE"]["correct"] for cap in CAPS
        }
        for cap in CAPS:
            deltas[cap].append(float(current[cap]))
        interactions.append(float(current[4096] - current[2048]))
    return {
        "method": "stratified_latent_bootstrap_descriptive",
        "n_resamples": 2_000,
        "seed": 0,
        "delta_correct_percentile_95": {
            cap: [float(np.quantile(deltas[cap], 0.025)), float(np.quantile(deltas[cap], 0.975))]
            for cap in CAPS
        },
        "interaction_percentile_95": [
            float(np.quantile(interactions, 0.025)),
            float(np.quantile(interactions, 0.975)),
        ],
    }


def _max_numeric_difference(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise SealedAnalysisAuditError("analysis/audit keys differ")
        return max((_max_numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
    ):
        if len(left) != len(right):
            raise SealedAnalysisAuditError("analysis/audit sequence lengths differ")
        return max(
            (_max_numeric_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    if left != right:
        raise SealedAnalysisAuditError("analysis/audit values differ")
    return 0.0


def audit_sealed_analysis(
    lock_path: str | Path,
    journal_path: str | Path,
    analysis_path: str | Path,
    raw_seal_path: str | Path,
    raw_audit_path: str | Path,
) -> dict[str, Any]:
    lock_file = Path(lock_path).expanduser().resolve()
    journal_file = Path(journal_path).expanduser().resolve()
    analysis_file = Path(analysis_path).expanduser().resolve()
    seal_file = Path(raw_seal_path).expanduser().resolve()
    raw_audit_file = Path(raw_audit_path).expanduser().resolve()
    analysis = _json_object(analysis_file, "analysis")
    seal = _json_object(seal_file, "raw seal")
    raw_audit = _json_object(raw_audit_file, "raw audit")
    if analysis.get("status") != "Q1_BUDGET_INTERACTION_ANALYSIS_COMPLETE":
        raise SealedAnalysisAuditError("primary analysis did not complete")
    if raw_audit.get("classification") != "RAW_SEAL_INDEPENDENT_AUDIT_PASS":
        raise SealedAnalysisAuditError("raw independent audit did not pass")
    if analysis.get("raw_seal_sha256") != _sha256(seal_file):
        raise SealedAnalysisAuditError("analysis raw-seal binding mismatch")
    if analysis.get("raw_seal_independent_audit_sha256") != _sha256(raw_audit_file):
        raise SealedAnalysisAuditError("analysis raw-audit binding mismatch")
    if analysis.get("journal_sha256") != _sha256(journal_file) or analysis.get(
        "journal_sha256"
    ) != seal.get("journal_sha256"):
        raise SealedAnalysisAuditError("analysis journal binding mismatch")
    if analysis.get("lock_sha256") != _sha256(lock_file):
        raise SealedAnalysisAuditError("analysis lock binding mismatch")

    rows = _load_rows(lock_file, journal_file)
    latent_rates, latents = _latent_category_rates(rows)
    grid = _aggregate(latent_rates, latents)
    delta = {cap: grid[cap]["D75"]["correct"] - grid[cap]["BASELINE"]["correct"] for cap in CAPS}
    interaction = float(delta[4096] - delta[2048])
    bootstrap = _bootstrap_summary(latent_rates, latents)
    contrasts, latent_interaction, weights = _weighted_contrasts(latent_rates, latents)
    lock = _json_object(lock_file, "lock")
    delta_threshold = float(lock["decision_rule"]["delta"])
    threshold = float(lock["decision_rule"]["threshold"])
    eplus = {cap: _evalue(contrasts[cap], weights, delta_threshold, 1) for cap in CAPS}
    eminus = {cap: _evalue(contrasts[cap], weights, delta_threshold, -1) for cap in CAPS}
    interaction_plus = _evalue(
        {key: value / 2.0 for key, value in latent_interaction.items()}, weights, 0.0, 1
    )
    interaction_minus = _evalue(
        {key: value / 2.0 for key, value in latent_interaction.items()}, weights, 0.0, -1
    )
    decisions = {
        "relevant_gain_at_either_cap": {
            "status": "TRIGGERED"
            if np.mean([eplus[cap] for cap in CAPS]) >= threshold
            else "NOT_TRIGGERED",
            "triggered": bool(np.mean([eplus[cap] for cap in CAPS]) >= threshold),
            "statistic": float(np.mean([eplus[cap] for cap in CAPS])),
            "threshold": threshold,
            "criterion": "average Eplus(delta, dL) >= threshold",
        },
        "ten_pp_gain_excluded_at_both_caps": {
            "status": "TRIGGERED" if min(eminus.values()) >= threshold else "NOT_TRIGGERED",
            "triggered": bool(min(eminus.values()) >= threshold),
            "statistic": float(min(eminus.values())),
            "threshold": threshold,
            "criterion": "min Eminus(delta, dL) >= threshold",
        },
        "interaction_nonzero": {
            "status": "TRIGGERED"
            if (interaction_plus + interaction_minus) / 2.0 >= threshold
            else "NOT_TRIGGERED",
            "triggered": bool((interaction_plus + interaction_minus) / 2.0 >= threshold),
            "statistic": float((interaction_plus + interaction_minus) / 2.0),
            "threshold": threshold,
            "criterion": "(Eplus(0, j/2) + Eminus(0, j/2)) / 2 >= threshold",
        },
    }
    recomputed = json.loads(
        json.dumps(
            {
                "category_rates": grid,
                "delta_correct": delta,
                "interaction": interaction,
                "descriptive_bootstrap": bootstrap,
                "fixed_prelock_rule": {
                    "method": "fixed_rule_prelock_candidates",
                    "delta": delta_threshold,
                    "threshold": threshold,
                    "n_latents": 96,
                    "weighted_delta": {
                        cap: float(sum(weights[key] * contrasts[cap][key] for key in weights))
                        for cap in CAPS
                    },
                    "weighted_interaction": float(
                        sum(weights[key] * latent_interaction[key] for key in weights)
                    ),
                    "decisions": decisions,
                    "statuses": {name: details["status"] for name, details in decisions.items()},
                },
            },
            sort_keys=True,
        )
    )
    primary = analysis.get("analysis")
    if not isinstance(primary, Mapping):
        raise SealedAnalysisAuditError("primary analysis payload is missing")
    maximum_difference = _max_numeric_difference(primary, recomputed)
    if maximum_difference > 1e-12:
        raise SealedAnalysisAuditError("primary analysis differs from independent recomputation")
    return {
        "schema_version": "q1-v3-budget-interaction-analysis-independent-audit-v1",
        "classification": "Q1_BUDGET_INTERACTION_ANALYSIS_AUDIT_PASS",
        "primary_analysis_sha256": _sha256(analysis_file),
        "raw_seal_sha256": _sha256(seal_file),
        "raw_seal_independent_audit_sha256": _sha256(raw_audit_file),
        "journal_sha256": _sha256(journal_file),
        "lock_sha256": _sha256(lock_file),
        "observation_count": len(rows),
        "latent_count": 96,
        "maximum_numeric_difference": maximum_difference,
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise SealedAnalysisAuditError(f"refusing to overwrite existing analysis audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        raise SealedAnalysisAuditError("cannot write analysis audit") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit sealed Q1 budget analysis")
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "review/q1_budget_causal_interaction/PRELOCK_ARTIFACTS/LOCK.json",
    )
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--raw-seal", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = audit_sealed_analysis(
        args.lock,
        args.journal,
        args.analysis,
        args.raw_seal,
        args.raw_audit,
    )
    _write_immutable_json(args.output, payload)
    print(
        json.dumps(
            {"classification": payload["classification"], "observation_count": 768},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
