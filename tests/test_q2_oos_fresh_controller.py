from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    angular_cross_block,
    cross_block_shape,
    fresh_candidate_bank,
    fresh_row_permutations,
    leave_one_fresh_out,
    row_permutation_test,
    semantic_schedule,
    shell_mean_spearman,
)


def test_candidate_stream_is_deterministic_and_in_subspace() -> None:
    rng = np.random.default_rng(4)
    basis, _ = np.linalg.qr(rng.standard_normal((32, 8)))
    first_c, first_v = fresh_candidate_bank(basis, count=12, seed=91)
    second_c, second_v = fresh_candidate_bank(basis, count=12, seed=91)
    assert np.array_equal(first_c, second_c)
    assert np.array_equal(first_v, second_v)
    assert np.max(np.abs(np.linalg.norm(first_c, axis=1) - 1.0)) < 1e-12
    assert np.max(np.abs(first_v - (first_v @ basis) @ basis.T)) < 1e-12


def test_cross_block_statistic_and_row_permutation_preserve_rows() -> None:
    rng = np.random.default_rng(7)
    reference = rng.standard_normal((9, 4))
    fresh = rng.standard_normal((6, 4))
    geometry = angular_cross_block(fresh, reference)
    outcomes = {"MEDIUM": geometry.copy(), "STRONG": geometry.copy()}
    geometries = {"MEDIUM": geometry.copy(), "STRONG": geometry.copy()}
    permutations = fresh_row_permutations(6, 720, seed=18)
    assert len(permutations) == 720
    assert len({tuple(row) for row in permutations}) == 720
    assert np.array_equal(permutations[0], np.arange(6))
    result = row_permutation_test(geometries, outcomes, permutations)
    assert result["observed_aggregate_rho"] > 0.999999
    assert result["p_value"] == 1 / 720
    assert np.all(leave_one_fresh_out(geometries, outcomes) > 0.999999)


def test_shell_mean_is_equal_weighted() -> None:
    geometry = np.arange(24, dtype=float).reshape(4, 6)
    medium = geometry.copy()
    strong = -geometry
    shell, aggregate = shell_mean_spearman(
        {"MEDIUM": geometry, "STRONG": geometry},
        {"MEDIUM": medium, "STRONG": strong},
    )
    assert shell == {"MEDIUM": 1.0, "STRONG": -1.0}
    assert aggregate == 0.0


def test_cross_block_shape_matches_scalar_reference() -> None:
    fresh = np.asarray([[[0, 1], [1, 1], [0, 0], [1, 0]]], dtype=float)
    reference = np.asarray([[[1, 1], [0, 1], [0, 1], [1, 0]]], dtype=float)
    result = cross_block_shape(fresh, reference)[0, 0]
    d0 = fresh[0, :, 0] - reference[0, :, 0]
    d1 = fresh[0, :, 1] - reference[0, :, 1]
    expected = len(d0) / (len(d0) - 1) * (np.mean(d0 * d1) - np.mean(d0) * np.mean(d1))
    assert result == expected


def test_future_schedule_contains_only_fresh_conditions() -> None:
    rows = semantic_schedule(
        ["item_a", "item_b"],
        ["Q2_OOS_DIRECTION_00", "Q2_OOS_DIRECTION_01"],
        "a" * 40,
    )
    assert len(rows) == 2 * 2 * 2 * 2
    assert len({row["seed"] for row in rows}) == len(rows)
    assert {row["condition"] for row in rows} == {
        "Q2_OOS_DIRECTION_00_MEDIUM",
        "Q2_OOS_DIRECTION_00_STRONG",
        "Q2_OOS_DIRECTION_01_MEDIUM",
        "Q2_OOS_DIRECTION_01_STRONG",
    }
    assert all(row["condition"] != "BASELINE" for row in rows)
