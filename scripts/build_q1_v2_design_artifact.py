#!/usr/bin/env python3
"""Build the local, model-free E3-10 design review artifact."""

from __future__ import annotations

import json
from pathlib import Path

from epistemic_geometry.benchmarks.e3.base import (
    DECIMAL_ANSWER_INSTRUCTION,
    GENERATOR_VERSION,
    NUMBER_WORD_ANSWER_INSTRUCTION,
    SUITE_VERSION,
)
from epistemic_geometry.benchmarks.e3.rendering import render_latent, template_hashes
from epistemic_geometry.benchmarks.e3.splits import (
    CALIBRATION_SPLIT,
    FAMILY_CELLS,
    generate_balanced_items,
)
from epistemic_geometry.benchmarks.e3.validation import validate_family

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review" / "q1_v2_instrument_design"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "family_definitions.json",
        {
            "suite_version": SUITE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "families": {
                "MODREG10": "four-register modular straight-line execution",
                "FSM10": "composition of three random bijective state transitions",
                "REACHCOUNT10": "bounded directed graph reachability count",
                "SATCOUNT10": "exhaustive Boolean model count modulo ten",
            },
        },
    )
    write_json(OUT / "difficulty_cells.json", FAMILY_CELLS)
    write_json(
        OUT / "prompt_templates.json",
        {
            "decimal_answer_instruction": DECIMAL_ANSWER_INSTRUCTION,
            "number_word_answer_instruction": NUMBER_WORD_ANSWER_INSTRUCTION,
            "template_hashes": template_hashes(),
            "few_shot": False,
            "chain_of_thought": False,
        },
    )
    write_json(
        OUT / "split_policy.json",
        {
            "calibration": {"name": CALIBRATION_SPLIT, "n_per_cell": 200, "per_digit": 20},
            "fresh_after_selection": {
                "GEOMETRY_CALIBRATION": {"n": 500, "per_digit": 50},
                "DEV_EVALUATION": {"n": 500, "per_digit": 50},
                "CONFIRMATORY_HOLDOUT": {"n": 1000, "per_digit": 100},
            },
            "holdout_firewall": "development code refuses CONFIRMATORY_HOLDOUT",
        },
    )
    validation = {
        family: validate_family(family, cells, n_per_cell=500)
        for family, cells in FAMILY_CELLS.items()
    }
    write_json(OUT / "generator_validation.json", {"status": "PASS", "families": validation})

    balance_audit: dict[str, object] = {}
    examples: list[dict[str, object]] = []
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            items = generate_balanced_items(
                family, cell, 100, 20260817, split_name=CALIBRATION_SPLIT
            )
            balance_audit[f"{family}/{cell}"] = {
                "n_items": len(items),
                "target_counts": {
                    str(digit): sum(item.target == digit for item in items) for digit in range(10)
                },
                "model_outcomes_used": False,
            }
            for item in items[:1]:
                examples.append(
                    {
                        "latent": item.to_record(),
                        "canonical_decimal": render_latent(item).to_record(),
                        "surface_twin_decimal": render_latent(
                            item, surface="surface_twin"
                        ).to_record(),
                        "canonical_number_word": render_latent(
                            item, response_channel="number_word"
                        ).to_record(),
                    }
                )
    write_json(OUT / "target_balance_audit.json", balance_audit)
    (OUT / "example_items.jsonl").write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    (OUT / "protocol_draft.md").write_text(
        "# E3-10 Local Design Artifact\n\n"
        "This artifact is model-free. It records deterministic generators, exact oracles, "
        "frozen prompt templates, target balance, and split rules. No Qwen output, activation, "
        "steering vector, or family-selection outcome is present.\n\n"
        "The baseline-only qualification protocol is documented in "
        "`docs/Q1_V2_EXACT_SEMANTIC_INSTRUMENT.md`.\n",
        encoding="utf-8",
    )
    write_json(
        OUT / "design_manifest.json",
        {
            "suite_version": SUITE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "model_outcomes": False,
            "files": sorted(path.name for path in OUT.iterdir()),
        },
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
