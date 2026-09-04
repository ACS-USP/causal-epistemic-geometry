#!/usr/bin/env python3
"""Create a release-safe, model-free severity audit of the frozen Q3 Tier-B pool."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_final_system_and_evaluation_supply"
LEDGER = ROOT / "review/q3_realizable_utility_design/ITEM_EXPOSURE_LEDGER.json"
PROVENANCE = (
    ROOT / "review/q2_m3_qualification_cruxeval_provenance/CRUXEVAL_PROVENANCE_LEDGER.jsonl"
)
PRECHECK = REVIEW / "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_ids(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def classify(item: dict[str, Any], provenance: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    influence = {
        "exact_candidate_policy_outcome": bool(item["candidate_q3_policy_outcome_observed"]),
        "controller_selection": bool(provenance.get("used_for_controller_selection")),
        "hyperparameter_selection": bool(provenance.get("used_for_hyperparameter_selection")),
        "metric_selection": bool(provenance.get("used_for_metric_selection")),
        "threshold_calibration": bool(provenance.get("used_for_threshold_calibration")),
        "source_axis_construction": bool(provenance.get("source_axis_construction")),
    }
    reasons.extend(key for key, value in influence.items() if value)
    if reasons:
        return "F", reasons
    if provenance.get("known_manual_inspection"):
        return "E", ["known_manual_raw_inspection"]
    if provenance.get("semantic_correctness_scored") or provenance.get(
        "outcome_inspected_by_researchers"
    ):
        return "D", [
            key
            for key, value in {
                "semantic_correctness_scored": provenance.get("semantic_correctness_scored"),
                "outcome_inspected": provenance.get("outcome_inspected_by_researchers"),
            }.items()
            if value
        ]
    if provenance.get("free_generation_inference"):
        return "C", ["unrelated_model_generation"]
    if item.get("prompt_sha256") or item.get("reference_sha256"):
        return "B", ["model_free_content_hashing_or_provenance"]
    return "A", ["stable_identifier_or_manifest_only"]


def main() -> int:
    precheck = read_json(PRECHECK)
    if precheck["status"] != "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_FROZEN":
        raise RuntimeError("Q3.3 precheck is not frozen")
    items = read_json(LEDGER)["items"]
    tier = [row for row in items if row["eligible_fresh_evaluation_tier_b"]]
    provenance_rows = [json.loads(line) for line in PROVENANCE.read_text().splitlines() if line]
    provenance = {row["item_id"]: row for row in provenance_rows}
    if len(tier) != 500 or len({row["item_id"] for row in tier}) != 500:
        raise RuntimeError("Tier-B population is not the frozen 500-family pool")
    if any(row["item_id"] not in provenance for row in tier):
        raise RuntimeError("Tier-B provenance is incomplete")

    by_stratum: dict[str, list[str]] = {key: [] for key in "ABCDEF"}
    reasons: Counter[str] = Counter()
    for row in tier:
        stratum, row_reasons = classify(row, provenance[row["item_id"]])
        by_stratum[stratum].append(row["item_id"])
        reasons.update(row_reasons)
    counts = {key: len(value) for key, value in by_stratum.items()}
    internal = by_stratum["A"] + by_stratum["B"]
    confirmatory = by_stratum["A"]
    result = {
        "schema_version": "q3-tier-b-exposure-severity-audit-v1",
        "status": "Q3_TIER_B_EXPOSURE_SEVERITY_AUDIT_COMPLETE",
        "evidence_class": "MODEL_FREE_PROVENANCE_AUDIT",
        "population": {
            "families": len(tier),
            "family_definition": "one CRUXEval output-prediction item",
            "population_id_sha256": sha256_ids([row["item_id"] for row in tier]),
            "permanent_allocation": False,
        },
        "stratum_counts": counts,
        "stratum_id_hashes": {key: sha256_ids(value) for key, value in by_stratum.items()},
        "reason_counts": dict(sorted(reasons.items())),
        "eligibility": {
            "confirmatory_families": len(confirmatory),
            "confirmatory_ids_sha256": sha256_ids(confirmatory),
            "bounded_internal_validation_families": len(internal),
            "bounded_internal_validation_ids_sha256": sha256_ids(internal),
            "ineligible_families": len(tier) - len(internal),
            "ruling": "TIER_B_NUMERICALLY_INADEQUATE_FOR_INTERNAL_VALIDATION"
            if len(internal) < 100
            else "TIER_B_POWER_REVIEW_REQUIRED",
        },
        "interpretation": {
            "A": precheck["tier_b_audit"]["strata"]["A"],
            "B": precheck["tier_b_audit"]["strata"]["B"],
            "C": precheck["tier_b_audit"]["strata"]["C"],
            "D": precheck["tier_b_audit"]["strata"]["D"],
            "E": precheck["tier_b_audit"]["strata"]["E"],
            "F": precheck["tier_b_audit"]["strata"]["F"],
            "tier_b_is_not_globally_fresh": True,
            "no_outcome_from_exact_47_candidate_controllers": True,
        },
        "release_safety": {
            "stable_ids_listed": False,
            "only_counts_and_set_hashes": True,
            "prompt_or_reference_text": False,
            "model_output": False,
            "correctness_values": False,
            "future_correctness_inspected": False,
        },
    }
    output = REVIEW / "Q3_TIER_B_EXPOSURE_SEVERITY_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "eligibility": result["eligibility"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
