#!/usr/bin/env python3
"""Validate the machine-readable experiment registry."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registry.yaml"
REQUIRED = {
    "id",
    "version",
    "stage",
    "question",
    "instrument",
    "model_policy",
    "status",
    "outcome",
    "scientific_interpretation",
    "source_commit",
    "artifact",
    "cost_usd",
    "holdout_status",
    "next_action",
}


def main() -> int:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        print("registry must contain an experiments list", file=sys.stderr)
        return 1
    identifiers: set[str] = set()
    for index, item in enumerate(payload["experiments"]):
        if not isinstance(item, dict):
            errors.append(f"experiment {index} is not a mapping")
            continue
        missing = REQUIRED - item.keys()
        if missing:
            errors.append(f"experiment {index} missing {sorted(missing)}")
        identifier = str(item.get("id", ""))
        if not identifier or identifier in identifiers:
            errors.append(f"duplicate or empty experiment id: {identifier!r}")
        identifiers.add(identifier)
        if item.get("stage") == "CONFIRMATORY":
            errors.append(f"{identifier}: no confirmatory experiment is registered as executed")
        if item.get("holdout_status") not in {
            "UNTOUCHED",
            "NOT_APPLICABLE",
            "SEALED_ASSIGNED_UNACCESSED",
            "CONSUMED_CONFIRMATORY_CLOSED",
        }:
            errors.append(f"{identifier}: unexpected holdout status")
    required_ids = {
        "Q1_V1_MMLU_PRO",
        "Q1_V2_E3_10",
        "Q1_V3_REASONING_AGENT_STAGE_A",
        "EXTERNAL_BENCHMARK_QUALIFICATION",
        "Q1_V4_CHARCOUNT",
        "Q1_V4_GEOMETRY",
        "Q1_V4_DENSE_CODE",
    }
    missing_ids = required_ids - identifiers
    if missing_ids:
        errors.append(f"missing historical experiments: {sorted(missing_ids)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"experiment registry: valid ({len(identifiers)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
