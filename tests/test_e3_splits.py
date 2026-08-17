from collections import Counter

import pytest

from epistemic_geometry.benchmarks.e3.splits import (
    CALIBRATION_SPLIT,
    CONFIRMATORY_HOLDOUT,
    DEV_SPLIT,
    FAMILY_CELLS,
    assert_development_access,
    assert_split_disjoint,
    generate_balanced_items,
    generate_fresh_split_manifest,
)


def test_balanced_rejection_sampling_is_exact_for_every_cell() -> None:
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            items = generate_balanced_items(family, cell, 20, 99, split_name=CALIBRATION_SPLIT)
            assert Counter(item.target for item in items) == Counter(
                {digit: 2 for digit in range(10)}
            )


def test_fresh_split_manifests_are_disjoint_and_holdout_is_firewalled() -> None:
    geometry = generate_fresh_split_manifest(
        "FSM10", "length_4", "GEOMETRY_CALIBRATION", seed=1, n_items=20
    )
    development = generate_fresh_split_manifest("FSM10", "length_4", DEV_SPLIT, seed=2, n_items=20)
    holdout = generate_fresh_split_manifest(
        "FSM10", "length_4", CONFIRMATORY_HOLDOUT, seed=3, n_items=20
    )
    assert_split_disjoint((geometry, development, holdout))
    assert_development_access(DEV_SPLIT)
    with pytest.raises(PermissionError):
        assert_development_access(CONFIRMATORY_HOLDOUT)
    assert holdout.metadata["development_access"] is False
