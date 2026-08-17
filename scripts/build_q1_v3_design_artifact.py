#!/usr/bin/env python3
"""Build the model-free Q1 V3 design bundle.

This command never loads a model, tokenizer, dataset, or activation.  It
materializes the frozen procedural design and the already-computed structural
gate so a principal researcher can review the instrument before RunPod use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.reasoning.base import (
    FINAL_ANSWER_INSTRUCTION,
    GENERATOR_VERSION,
    SUITE_VERSION,
)
from epistemic_geometry.benchmarks.reasoning.calibration import (
    REASONING_BUDGETS,
    STAGE_A_ITEMS,
    STAGE_A_MAX_ACCURACY,
    STAGE_A_MAX_SEED_GAP,
    STAGE_A_MIN_ACCURACY,
    STAGE_A_MIN_PARSE_SUCCESS,
    STAGE_B_ITEMS,
    STAGE_B_MAX_ACCURACY,
    STAGE_B_MAX_SEED_SD,
    STAGE_B_MAX_TWIN_ACCURACY_GAP,
    STAGE_B_MIN_ACCURACY,
    STAGE_B_MIN_PARSE_SUCCESS,
    STAGE_B_MIN_TWIN_AGREEMENT,
)
from epistemic_geometry.benchmarks.reasoning.families import FAMILY_CELLS, generate_item
from epistemic_geometry.benchmarks.reasoning.rendering import render_reasoning
from epistemic_geometry.benchmarks.reasoning.validation import (
    ANSWER_COLLAPSE_FAILURE,
    ANSWER_COLLAPSE_WARNING,
    SHORTCUT_FAILURE,
    validate_suite,
)
from epistemic_geometry.reproducibility import canonical_json


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _family_definitions() -> dict[str, Any]:
    return {
        "MODREG-R": {
            "source_structure": "four modulo-10 registers",
            "oracle": "straight-line exact interpreter",
            "cells": list(FAMILY_CELLS["MODREG-R"]),
            "surface_twin": "bijective register rename and deterministic formatting change",
            "answer_space": "0..9",
        },
        "FSM-R": {
            "source_structure": "ten-state machine with three bijective transitions",
            "oracle": "exact transition-table simulation",
            "cells": list(FAMILY_CELLS["FSM-R"]),
            "surface_twin": "row/symbol presentation reorder with sequence remapping",
            "answer_space": "0..9",
        },
        "SATCOUNT-R": {
            "source_structure": "small CNF Boolean model counting",
            "oracle": "exhaustive assignment enumeration",
            "cells": list(FAMILY_CELLS["SATCOUNT-R"]),
            "surface_twin": "clause/literal order change",
            "answer_space": "0..2^n_variables, exact raw count; no modulo",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("review/q1_v3_reasoning_instrument"),
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("review/q1_v3_reasoning_instrument/structural_gate_summary.json"),
    )
    args = parser.parse_args()

    gate = (
        json.loads(args.gate.read_text(encoding="utf-8"))
        if args.gate.exists()
        else validate_suite(n_per_cell=5000)
    )
    if gate.get("status") != "PASS":
        raise SystemExit("Q1 V3 design artifact refuses a failed structural gate")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol = (
        "Q1 V3 model-free design artifact. No Qwen, tokenizer, dataset, model "
        "outcome, activation, steering direction, DEV split, or holdout was used.\n"
    )
    (output / "protocol_draft.md").write_text(protocol, encoding="utf-8")
    summary = f"""# Q1 V3 — Reasoning-Agent Structural Reset

## OLD INSTRUMENT

Q1 V1–V1.2 and Q1 V2 / E3-10 are closed as DEVELOPMENT instruments. No Q1
scientific result is frozen.

## LOCAL GENERATORS

MODREG-R, FSM-R, and SATCOUNT-R passed deterministic oracle, serialization,
surface-twin, answer-distribution, and shallow-shortcut audits. The structural
gate status is **{gate['status']}** using the stored model-free report.

## CALIBRATION

Stage A: **NOT RUN**. Stage B: **NOT RUN**. Qwen outcomes: none. The Stage-A
manifest builder is model-free and creates 36 budget conditions (12 cells × 3
budgets) over 12 frozen 60-item latent sets. Corresponding rollout seed
identities are shared across budgets.

## INSTRUMENT

Q1 V3 reasoning-agent instrument: **PRE-CALIBRATION / NOT QUALIFIED YET**.
The policy is Qwen3-8B with thinking enabled, deterministic recorded sampling
configuration, exact `FINAL:` parsing, and raw trajectory retention.

## SPLITS AND FIREWALL

Fresh geometry, steering-development, and confirmatory splits are **NOT
GENERATED**. Confirmatory access is **UNTOUCHED**.

## STEERING

One-shot reasoning-policy steering is **NOT READY** and was not constructed.
No PCA, random control, DEV evaluation, or geometry experiment was run.

## INFRASTRUCTURE

The model-free bundle contains no weights or model outputs. The RunPod remains
stopped during local implementation. Real baseline-only calibration requires
principal review and the documented remote cache/cost gates.
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    _write(output / "family_definitions.json", _family_definitions())
    _write(
        output / "difficulty_cells.json",
        {family: list(cells) for family, cells in FAMILY_CELLS.items()},
    )
    _write(
        output / "frozen_qualification_rules.json",
        {
            "stage_a": {
                "items_per_cell_budget": STAGE_A_ITEMS,
                "rollouts": 2,
                "budgets": list(REASONING_BUDGETS),
                "accuracy_range_inclusive": [STAGE_A_MIN_ACCURACY, STAGE_A_MAX_ACCURACY],
                "parse_success_min": STAGE_A_MIN_PARSE_SUCCESS,
                "max_seed_accuracy_gap": STAGE_A_MAX_SEED_GAP,
            },
            "stage_b": {
                "items_per_selected_family": STAGE_B_ITEMS,
                "rollouts": 4,
                "accuracy_range_inclusive": [STAGE_B_MIN_ACCURACY, STAGE_B_MAX_ACCURACY],
                "parse_success_min": STAGE_B_MIN_PARSE_SUCCESS,
                "max_twin_accuracy_gap": STAGE_B_MAX_TWIN_ACCURACY_GAP,
                "min_twin_agreement": STAGE_B_MIN_TWIN_AGREEMENT,
                "max_seed_accuracy_sd": STAGE_B_MAX_SEED_SD,
            },
            "suite_minimum_qualifying_families": 2,
        },
    )
    _write(
        output / "split_policy.json",
        {
            "calibration_namespace": "REASONING_INSTRUMENT_CALIBRATION",
            "stage_a_namespace": "REASONING_STAGE_A_SCREEN",
            "stage_b_namespace": "REASONING_STAGE_B_CALIBRATION",
            "fresh_counts_per_retained_family": {
                "GEOMETRY_CALIBRATION": 400,
                "STEERING_DEVELOPMENT": 400,
                "CONFIRMATORY_HOLDOUT": 800,
            },
            "target_balancing": "not_applied",
            "item_filtering_after_generation": "forbidden",
            "confirmatory_access_during_development": False,
        },
    )
    _write(
        output / "answer_distribution_audit_policy.json",
        {
            "target_balance": "not_applicable; no forced target balancing",
            "modal_warning_frequency": ANSWER_COLLAPSE_WARNING,
            "modal_failure_frequency": ANSWER_COLLAPSE_FAILURE,
            "shortcut_failure_accuracy": SHORTCUT_FAILURE,
            "source": "procedural oracle only; no model outcomes",
        },
    )

    examples: list[dict[str, Any]] = []
    prompt_templates: dict[str, Any] = {}
    for family, cells in FAMILY_CELLS.items():
        cell = cells[0]
        item = generate_item(family, cell, 17)
        canonical = render_reasoning(item)
        twin = render_reasoning(item, surface="surface_twin")
        examples.extend(
            [
                {"item": item.to_record(), "view": canonical.to_record()},
                {"item": item.to_record(), "view": twin.to_record()},
            ]
        )
        prompt_templates[family] = {
            "canonical_template_hash": canonical.template_hash,
            "surface_twin_template_hash": twin.template_hash,
            "final_answer_instruction": FINAL_ANSWER_INSTRUCTION,
            "canonical_prompt_hash_example": canonical.prompt_hash,
            "surface_twin_prompt_hash_example": twin.prompt_hash,
        }
    (output / "example_items.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in examples), encoding="utf-8"
    )
    _write(output / "prompt_templates.json", prompt_templates)
    _write(output / "generator_validation.json", gate)
    _write(
        output / "target_balance_audit.json",
        {
            "status": "NOT_APPLIED_BY_DESIGN",
            "reason": "Q1 V3 uses procedural answer-distribution audits, not forced target balance",
        },
    )
    manifest = {
        "suite_version": SUITE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "artifact": "q1_v3_reasoning_instrument_design",
        "model_accessed": False,
        "steering_accessed": False,
        "confirmatory_accessed": False,
        "files": sorted(path.name for path in output.iterdir()),
        "protocol_hash": _sha256_text(protocol),
        "gate_status": gate["status"],
        "manifest_hash_basis": canonical_json({"suite": SUITE_VERSION, "gate": gate["status"]}),
    }
    _write(output / "design_manifest.json", manifest)
    print(json.dumps({"output": str(output), "status": "PASS", "model_accessed": False}))


if __name__ == "__main__":
    main()
