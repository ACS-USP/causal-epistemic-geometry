#!/usr/bin/env python3
"""Build Q1 V3 calibration manifests without loading a model.

The script is intentionally phase-gated.  It can build Stage A before model
outcomes exist, select Stage B only from stored Stage-A outcomes, and generate
fresh scientific manifests only after stored Stage-B qualification outcomes
show at least two qualifying families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.reasoning.calibration import (
    generate_stage_a_manifests,
    generate_stage_b_manifests,
    select_qualified_families,
    select_stage_b_cells,
)
from epistemic_geometry.benchmarks.reasoning.splits import (
    generate_fresh_scientific_splits,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_payload(phase: str, manifests: tuple[Any, ...], **extra: Any) -> dict[str, Any]:
    payload = {
        "suite": "Q1-V3-REASONING",
        "phase": phase,
        "manifest_schema_version": (
            "q1-v3-stage-a-paired-budget-v1" if phase == "stage_a_screen" else "q1-v3-v1"
        ),
        "model_outcomes": False,
        "steering_outcomes": False,
        "manifests": {
            f"{manifest.family}/{manifest.cell}/{manifest.reasoning_budget}": manifest.to_record()
            for manifest in manifests
        },
        **extra,
    }
    if phase == "stage_a_screen":
        paired_groups: dict[str, Any] = {}
        for manifest in manifests:
            group_key = f"{manifest.family}/{manifest.cell}"
            group = paired_groups.setdefault(
                group_key,
                {
                    "family": manifest.family,
                    "cell": manifest.cell,
                    "seed": manifest.seed,
                    "budgets": [],
                    "latent_ids": [item.latent_id for item in manifest.items],
                    "paired_item_set_hash": manifest.metadata["paired_item_set_hash"],
                },
            )
            group["budgets"].append(manifest.reasoning_budget)
        for group in paired_groups.values():
            group["budgets"] = sorted(group["budgets"])
            group["latent_id_count"] = len(group["latent_ids"])
        payload["paired_budget_groups"] = paired_groups
    payload["manifest_hash"] = stable_digest(
        "Q1-V3-MANIFEST-CONTENT", canonical_json(payload)
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage_a", "stage_b", "fresh"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--outcomes", type=Path)
    args = parser.parse_args()

    if args.phase == "stage_a":
        if args.gate is None:
            raise ValueError("stage_a requires --gate")
        gate = _load(args.gate)
        from epistemic_geometry.benchmarks.reasoning.calibration import eligible_cells_from_gate

        eligible = eligible_cells_from_gate(gate)
        manifests = generate_stage_a_manifests(eligible, seed=args.seed)
        _write(
            args.output,
            _manifest_payload(
                "stage_a_screen",
                manifests,
                eligible_cells=[f"{family}/{cell}" for family, cell in eligible],
                planned_rollouts_per_view=2,
            ),
        )
        print(json.dumps({"phase": "stage_a_screen", "manifests": len(manifests)}))
        return

    if args.outcomes is None:
        raise ValueError(f"{args.phase} requires --outcomes")
    outcomes_payload = _load(args.outcomes)
    outcomes = outcomes_payload.get("outcomes", outcomes_payload)
    if not isinstance(outcomes, list):
        raise ValueError("outcomes must be a JSON list or an object with an outcomes list")

    if args.phase == "stage_b":
        selected = select_stage_b_cells(outcomes)
        manifests = generate_stage_b_manifests(selected, seed=args.seed)
        _write(
            args.output,
            _manifest_payload(
                "stage_b_calibration",
                manifests,
                stage_a_selected=selected,
                planned_rollouts_per_view=4,
            ),
        )
        status = "REASONING_INSTRUMENT_SCREEN_FAILED" if len(selected) < 2 else "STAGE_B_READY"
        print(
            json.dumps(
                {"phase": "stage_b_calibration", "status": status, "families": sorted(selected)}
            )
        )
        return

    qualified = select_qualified_families(outcomes)
    if len(qualified) < 2:
        _write(
            args.output,
            {
                "suite": "Q1-V3-REASONING",
                "phase": "fresh_scientific_splits",
                "generated": False,
                "status": "REASONING_INSTRUMENT_NOT_QUALIFIED",
                "qualified_families": sorted(qualified),
                "manifests": {},
            },
        )
        print(json.dumps({"status": "REASONING_INSTRUMENT_NOT_QUALIFIED"}))
        return
    manifests = generate_fresh_scientific_splits(qualified, seed=args.seed)
    _write(
        args.output,
        _manifest_payload(
            "fresh_scientific_splits",
            manifests,
            generated=True,
            qualified_families=sorted(qualified),
            qwen_evaluated=False,
        ),
    )
    print(json.dumps({"status": "FRESH_SPLITS_READY", "families": sorted(qualified)}))


if __name__ == "__main__":
    main()
