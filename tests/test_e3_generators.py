from __future__ import annotations

from epistemic_geometry.benchmarks.e3 import FAMILY_CELLS
from epistemic_geometry.benchmarks.e3.oracle import oracle_for
from epistemic_geometry.benchmarks.e3.splits import generate_latent
from epistemic_geometry.benchmarks.e3.validation import validate_item


def test_all_families_are_deterministic_and_exact() -> None:
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            first = generate_latent(family, cell, 314159)
            second = generate_latent(family, cell, 314159)
            assert first == second
            assert first.target in range(10)
            assert oracle_for(first) == first.target
            assert validate_item(first)["surface_oracle_equal"]


def test_latent_serialization_roundtrip_preserves_identity() -> None:
    item = generate_latent("SATCOUNT10", "vars5_clauses8", 27)
    assert type(item).from_record(item.to_record()) == item
    assert item.latent_id.endswith(item.latent_hash[:16])


def test_family_cells_have_expected_structural_metadata() -> None:
    assert generate_latent("MODREG10", "depth_16", 1).difficulty["depth"] == 16
    assert generate_latent("FSM10", "length_12", 1).difficulty["sequence_length"] == 12
    assert generate_latent("REACHCOUNT10", "H3_p015", 1).difficulty["max_hops"] == 3
    assert generate_latent("SATCOUNT10", "vars6_clauses10", 1).difficulty["variables"] == 6
