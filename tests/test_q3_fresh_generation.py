from __future__ import annotations

import importlib.util
from pathlib import Path

from epistemic_geometry.benchmarks.q3_fresh.instrument import build_family

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_q3_fresh_instrument", ROOT / "scripts/generate_q3_fresh_instrument.py"
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_effective_amendment_is_exact_and_pregeneration() -> None:
    assert GENERATOR.sha256(GENERATOR.AMENDMENT) == GENERATOR.EXPECTED_AMENDMENT_SHA
    amendment = GENERATOR.json.loads(GENERATOR.AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["scientific_generation_before_amendment"] == 0
    assert amendment["scientific_outcomes_before_amendment"] == 0
    assert amendment["status"] == "FROZEN_BEFORE_SCIENTIFIC_GENERATION"


def test_namespace_seeds_are_deterministic_and_distinct() -> None:
    first = {namespace: GENERATOR.derive_seed(namespace) for namespace, _ in GENERATOR.ALLOCATION}
    second = {namespace: GENERATOR.derive_seed(namespace) for namespace, _ in GENERATOR.ALLOCATION}
    assert first == second
    assert len(set(first.values())) == 3
    assert all(0 <= seed < 2**63 for seed in first.values())


def test_structural_near_duplicate_rule_is_symmetric() -> None:
    family = build_family("excluded-generation-fixture", 0, 111)
    same = build_family("excluded-generation-fixture", 0, 111)
    other = build_family("excluded-generation-fixture", 1, 111)
    assert GENERATOR.structural_near_duplicate(family, same)
    assert GENERATOR.structural_near_duplicate(
        family, other
    ) == GENERATOR.structural_near_duplicate(other, family)


def test_public_manifest_row_excludes_content() -> None:
    row = GENERATOR.public_row(build_family("excluded-generation-fixture", 4, 222))
    assert "source" not in row
    assert "prompt" not in row
    assert "reference_repr" not in row
    assert row["source_sha256"]
    assert row["prompt_sha256"]
    assert row["reference_sha256"]
