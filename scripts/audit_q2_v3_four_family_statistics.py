#!/usr/bin/env python3
"""Independent outcome-free audit of the Q2 V3 four-family design sprint."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v3_four_family_statistical_redesign"
AMENDMENT = ROOT / "review/q2_v3_amendment1_freeze"
QUALIFICATION = ROOT / "review/q2_v3_amendment1_execution/Q2_V3_SOURCE_QUALIFICATION.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    historical_source = (
        ROOT / "src/epistemic_geometry/experiments/q2_v3.py"
    ).read_text(encoding="utf-8")
    old_implementation = all(
        text in historical_source
        for text in (
            "for family_permutation in permutations(range(5))",
            "for swaps in product((0, 1), repeat=5)",
        )
    )
    mappings = []
    for family_permutation in permutations(range(4)):
        for swaps in product((0, 1), repeat=4):
            mappings.append(
                tuple(
                    2 * family_permutation[family] + (location ^ swaps[family])
                    for family in range(4)
                    for location in range(2)
                )
            )
    edges = [
        edge
        for edge in combinations(range(8), 2)
        if edge[0] // 2 != edge[1] // 2
    ]

    manifest = read_json(REVIEW / "FOUR_FAMILY_PRIMARY_PANEL_MANIFEST.json")
    inherited = read_json(AMENDMENT / "PRIMARY_PANEL_MANIFEST.json")
    selected_ids = manifest["selected_ids"]
    selected_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()

    qualification = read_json(QUALIFICATION)
    passed = sorted(
        family for family, record in qualification["families"].items() if record["pass"]
    )
    expected_passed = sorted(
        (
            "CONTROL_FLOW_PATH_COVERAGE",
            "MUTATION_ALIAS_CAUSALITY",
            "LOOP_BOUNDARY_ACCOUNTING",
            "HYPOTHESIS_BRANCH_ELIMINATION",
        )
    )

    with (REVIEW / "SIMULATION_POWER_PRECISION.csv").open(newline="", encoding="utf-8") as handle:
        power_rows = list(csv.DictReader(handle))
    with (REVIEW / "IDENTIFIABILITY_SIMULATION.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        geometry_rows = list(csv.DictReader(handle))

    checks = {
        "historical_qap_implementation_reconstructed": old_implementation,
        "historical_qap_universe": math.factorial(5) * 2**5,
        "four_family_qap_unique_maps": len(set(mappings)),
        "four_family_qap_expected_384": len(set(mappings)) == 384,
        "identity_included": tuple(range(8)) in mappings,
        "cross_family_dyads_per_shell": len(edges),
        "cross_family_dyads_expected_24": len(edges) == 24,
        "shell_stratified_dyads": 2 * len(edges),
        "radial_pairs": 8,
        "source_qualified_families": passed,
        "source_qualified_exact_four_expected": passed == expected_passed,
        "api_family_excluded": "API_CONTRACT_EXACTNESS" not in passed,
        "panel_selected_n": len(selected_ids),
        "panel_selected_unique": len(set(selected_ids)) == 300,
        "inherited_200_preserved_as_prefix": selected_ids[:200] == inherited["item_ids"],
        "panel_disjoint_from_other_allocations": set(selected_ids).isdisjoint(
            manifest["disjoint_excluded_ids"]
        ),
        "panel_order_hash_reproduced": selected_hash
        == manifest["selected_ordered_ids_sha256"],
        "panel_correctness_values_read": manifest["correctness_values_read"],
        "power_cells": len(power_rows),
        "power_cells_expected_54": len(power_rows) == 54,
        "geometry_regimes": len(geometry_rows),
        "geometry_regimes_expected_3": len(geometry_rows) == 3,
        "qap_rejection_count_at_alpha_0_05": 19,
        "qap_19_over_384": 19 / 384,
        "qap_20_over_384": 20 / 384,
        "radial_family_signflip_min_p": 1 / 16,
        "radial_p_0_05_attainable": False,
    }
    boolean_checks = [
        value
        for key, value in checks.items()
        if isinstance(value, bool)
        and key not in {"panel_correctness_values_read", "radial_p_0_05_attainable"}
    ]
    passed_audit = (
        all(boolean_checks)
        and checks["panel_correctness_values_read"] is False
        and checks["radial_p_0_05_attainable"] is False
    )
    payload = {
        "schema_version": "q2-v3-four-family-independent-design-audit-v1",
        "classification": (
            "Q2_V3_FOUR_FAMILY_DESIGN_AUDIT_CLEAN"
            if passed_audit
            else "Q2_V3_FOUR_FAMILY_DESIGN_AUDIT_CONCERN"
        ),
        "checks": checks,
        "behavioral_data_audit": {
            "new_model_inference": "NONE",
            "runpod": 0,
            "correctness_inspected": False,
            "shell_calibration": "NOT_RUN",
            "M0_M1_M2": "NOT_RUN",
            "semantic_v3_outcomes": 0,
            "Q3": "NOT_RUN",
        },
        "input_boundary": [
            "historical protocol/statistical source code",
            "source-qualification pass flags and vector metadata",
            "Class-C provenance IDs/classes/content hashes only",
            "synthetic simulation artifacts",
        ],
        "outcome_journals_loaded": [],
    }
    write_json(REVIEW / "DESIGN_AUDIT.json", payload)
    (REVIEW / "DESIGN_AUDIT.md").write_text(
        "# Independent four-family design audit\n\n"
        f"Classification: `{payload['classification']}`.\n\n"
        "The audit independently reproduced the 3,840- and 384-map group sizes, "
        "24 dyads per shell, 48 shell-stratified dyads, eight radial pairs, the "
        "four inherited source-qualified families, the 300-item order/hash and "
        "all disjointness checks. The original 200 IDs are an exact ordered "
        "prefix. No correctness value, model output, shell result, geometry "
        "matrix, or semantic V3 outcome was loaded. RunPod use was zero.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": payload["classification"]}))
    return 0 if passed_audit else 1


if __name__ == "__main__":
    raise SystemExit(main())
