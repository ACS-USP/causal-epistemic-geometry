from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.q2_v4_presemantic import (
    QAP_MAPS,
    bank_algebraic_checks,
    baseline_centered_angle,
    candidate_bank,
    prelock_seed,
    retained_subspace,
    select_first_safe,
    selected_bank_checks,
    semantic_schedule,
    unique_controller_permutations,
    unique_shell_swaps,
)

PRELOCK = "1" * 40


def test_prelock_seed_is_big_endian_deterministic() -> None:
    assert prelock_seed(PRELOCK) == prelock_seed(PRELOCK)
    assert prelock_seed(PRELOCK) != prelock_seed("2" * 40)


def test_subspace_and_single_candidate_stream() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(64, 8))
    q, report = retained_subspace(matrix)
    assert report["pass"]
    coefficients, vectors, seed = candidate_bank(q, PRELOCK)
    coefficients2, vectors2, seed2 = candidate_bank(q, PRELOCK)
    np.testing.assert_array_equal(coefficients, coefficients2)
    np.testing.assert_array_equal(vectors, vectors2)
    assert seed == seed2
    assert bank_algebraic_checks(coefficients, vectors)["pass"]


def test_first_32_safe_selection_does_not_optimize() -> None:
    records = {
        f"V4_DIRECTION_{index:02d}": {"both_shells_pass": index not in {1, 4, 7}}
        for index in range(40)
    }
    selected = select_first_safe(records)
    assert len(selected) == 32
    assert selected[:3] == ["V4_DIRECTION_00", "V4_DIRECTION_02", "V4_DIRECTION_03"]
    assert selected[-1] == "V4_DIRECTION_34"


def test_selected_identifiability() -> None:
    rng = np.random.default_rng(9)
    rows = rng.normal(size=(32, 8))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    amplitudes = np.column_stack((np.full(32, 0.25), np.full(32, 0.50)))
    report = selected_bank_checks(rows, amplitudes)
    assert report["pass"]


def test_baseline_centered_a2_recovers_valid_angle() -> None:
    baseline = np.asarray([[3.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    fingerprints = {
        "a": baseline + np.asarray([[0.2, -0.1, 0.0], [0.1, -0.1, 0.0]]),
        "b": baseline + np.asarray([[-0.1, 0.2, 0.0], [-0.2, 0.1, 0.0]]),
    }
    result = baseline_centered_angle(baseline, fingerprints, noise_floor_squared=1e-12)
    assert result["radius_floor_pass"]
    np.testing.assert_allclose(np.diag(result["dissimilarity"]), 0.0)
    np.testing.assert_allclose(result["dissimilarity"], result["dissimilarity"].T)


def test_qap_and_semantic_schedules_are_deterministic_and_unique() -> None:
    permutations, seed = unique_controller_permutations(PRELOCK)
    assert permutations.shape == (QAP_MAPS, 32)
    assert len({row.tobytes() for row in permutations}) == QAP_MAPS
    np.testing.assert_array_equal(permutations[0], np.arange(32))
    assert seed == unique_controller_permutations(PRELOCK)[1]
    swaps, swap_seed = unique_shell_swaps(PRELOCK)
    assert swaps.shape == (QAP_MAPS, 32)
    assert not swaps[0].any()
    assert len({row.tobytes() for row in swaps}) == QAP_MAPS
    assert swap_seed == unique_shell_swaps(PRELOCK)[1]
    selected = [f"V4_DIRECTION_{index:02d}" for index in range(8, 40)]
    schedule = semantic_schedule(["a", "b"], selected, PRELOCK)
    assert len(schedule) == 2 * 2 * 65
    keys = {(r["item_id"], r["condition"], r["rollout_index"]) for r in schedule}
    assert len(keys) == len(schedule)
    assert len({r["seed"] for r in schedule}) == len(schedule)
    assert {r["condition"] for r in schedule if r["condition"] != "BASELINE"} == {
        f"{direction}_{shell}" for direction in selected for shell in ("MEDIUM", "STRONG")
    }
