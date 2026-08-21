#!/usr/bin/env python3
"""Fail-closed validator for the offline Gate 6.3 semantic-validity audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "review/gate6_3_semantic_validity_audit"
SOURCE = ROOT / "review/gate6_3_single_mean_semantic_evaluation"
EXPECTED_HISTORICAL = "GATE6_3_SINGLE_MEAN_DESTRUCTIVE"
ALLOWED_DIAGNOSTICS = {
    "GATE6_3_V3_STRONG_SPECIFIC_CONTROL_SIGNAL",
    "GATE6_3_V3_ERROR_PROFILE_MOVEMENT_ONLY",
    "GATE6_3_V3_VALIDITY_COST_CONFIRMED",
    "GATE6_3_V3_NO_SPECIFIC_MOVEMENT",
    "GATE6_3_V3_AUDIT_INCONCLUSIVE",
}

csv.field_size_limit(sys.maxsize)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []
    lock = load_json(AUDIT / "AUDIT_LOCK.json")
    estimands = load_json(AUDIT / "ESTIMANDS_V3.json")
    hashes = load_json(AUDIT / "artifact_hashes.json")
    bootstrap = load_json(AUDIT / "BOOTSTRAP_INTERVALS_V3.json")
    rows = list(csv.DictReader((AUDIT / "ROW_REANALYSIS_V3.csv").open(encoding="utf-8")))
    summaries = list(csv.DictReader((AUDIT / "CONDITION_SUMMARY_V3.csv").open(encoding="utf-8")))
    loo = list(csv.DictReader((AUDIT / "LOO_SENSITIVITY.csv").open(encoding="utf-8")))

    if len(rows) != 920:
        errors.append(f"expected 920 reanalysis rows, found {len(rows)}")
    keys = [(row["item_id"], row["condition"], row["rollout_index"]) for row in rows]
    if len(set(keys)) != len(keys):
        errors.append("duplicate logical row in V3 reanalysis")
    phase_counts = Counter(row["phase"] for row in rows)
    if phase_counts != {
        "GATE6_3_MATCHED_RANDOM_SUPPLEMENT": 80,
        "GATE6_3_PRIMARY_EVALUATION": 840,
    }:
        errors.append(f"unexpected phase counts: {dict(phase_counts)}")
    primary = [row for row in rows if row["phase"] == "GATE6_3_PRIMARY_EVALUATION"]
    condition_counts = Counter(row["condition"] for row in primary)
    if set(condition_counts.values()) != {120} or len(condition_counts) != 7:
        errors.append(f"unexpected primary condition counts: {dict(condition_counts)}")
    if len(summaries) != 7:
        errors.append(f"expected 7 condition summaries, found {len(summaries)}")
    if len(loo) != 60 or len({row["left_out_item_id"] for row in loo}) != 60:
        errors.append("LOO sensitivity does not cover exactly 60 unique items")
    if not bootstrap or any(int(record["resamples"]) != 5000 for record in bootstrap.values()):
        errors.append("bootstrap does not use 5,000 item-cluster resamples throughout")
    if any(record["cluster"] != "item_id" for record in bootstrap.values()):
        errors.append("bootstrap unit is not item_id throughout")

    if estimands.get("historical_classification") != EXPECTED_HISTORICAL:
        errors.append("historical Gate 6.3 classification changed")
    if estimands.get("historical_result_modified") is not False:
        errors.append("audit claims historical result modification")
    if estimands.get("diagnostic_classification") not in ALLOWED_DIAGNOSTICS:
        errors.append("unknown V3 diagnostic classification")
    if estimands.get("model_inference") is not False or estimands.get("gpu_cost_usd") != 0.0:
        errors.append("offline audit provenance incorrectly records inference or GPU cost")
    if estimands["v2_crosscheck"]["historical_v2_crosscheck_failures"] != 0:
        errors.append("historical V2 crosscheck has failures")
    if lock.get("historical_result_mutable") is not False:
        errors.append("audit lock permits historical mutation")
    if lock["firewall"] != {
        "character_count": "NOT_RUN",
        "confirmatory_holdout": "UNTOUCHED",
        "new_trajectories": 0,
        "q2": "NOT_RUN",
        "runpod": "NOT_ACCESSED",
    }:
        errors.append("audit lock firewall differs from the frozen local-only design")
    if digest(SOURCE / "journal.jsonl") != lock["source_journal_sha256"]:
        errors.append("historical journal digest changed")
    for name, expected in lock["immutable_source_files"].items():
        if digest(SOURCE / name) != expected:
            errors.append(f"immutable historical artifact changed: {name}")
    for name, expected in hashes.items():
        if digest(AUDIT / name) != expected:
            errors.append(f"audit artifact digest mismatch: {name}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "Gate 6.3 semantic-validity audit: valid "
        f"({len(rows)} rows, {len(bootstrap)} bootstrap intervals, "
        f"{estimands['diagnostic_classification']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
