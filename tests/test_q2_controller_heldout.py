from __future__ import annotations

import numpy as np

from epistemic_geometry.analysis.q2_geometries import (
    finite_secant_geometry,
    fit_whitening,
    flat_geometry,
    whitened_geometry,
)
from epistemic_geometry.analysis.q2_prediction import (
    classify_q2,
    edge_indices,
    heldout_prediction,
    qap_permutation,
)
from epistemic_geometry.experiments.q2_controller_heldout import (
    CONDITIONS,
    CONTROLLER_IDS,
    LOCATIONS,
    NULL_IDS,
    SOURCE_AXES,
    build_null_bank,
    build_schedule,
    controller_split,
    expand_meaningful_bank,
    pairwise_unbiased_distance_matrix,
    qualification_decision,
    validate_bank,
)


def synthetic_base(hidden: int = 32) -> tuple[dict[tuple[str, str], np.ndarray], dict]:
    keys = [(axis.axis_id, location) for axis in SOURCE_AXES for location in LOCATIONS]
    basis = np.eye(hidden)[: len(keys)]
    directions = dict(zip(keys, basis, strict=True))
    pairs = {
        key: np.stack([basis[index] + 0.01 * np.eye(hidden)[8 + row] for row in range(4)])
        for index, key in enumerate(keys)
    }
    return directions, pairs


def test_bank_has_twelve_meaningful_four_nulls_and_exact_sign_pairs() -> None:
    base, pairs = synthetic_base()
    bank = expand_meaningful_bank(base)
    nulls, _metadata = build_null_bank(base, pairs)
    bank.update(nulls)
    checks = validate_bank(bank)
    assert len(bank) == 16
    assert len(NULL_IDS) == 4
    assert checks["unit_norm_pass"]
    assert checks["sign_pair_pass"]
    assert checks["base_diversity_pass"]
    assert checks["null_orthogonality_pass"]


def test_controller_split_is_source_family_heldout_ten_six() -> None:
    split = controller_split()
    assert split["train_n"] == 10
    assert split["test_n"] == 6
    assert split["train_edge_count"] == 45
    assert split["heldout_edge_count"] == 75
    assert not (set(split["train_controllers"]) & set(split["test_controllers"]))
    heldout_axis = split["heldout_axis"]
    heldout_meaningful = [name for name in split["test_controllers"] if name.startswith("MEAN_")]
    assert len(heldout_meaningful) == 4
    assert all(heldout_axis in name for name in heldout_meaningful)


def test_common_panel_schedule_is_complete_independent_and_deterministic() -> None:
    ids = [f"sample_{index}" for index in range(120)]
    first = build_schedule(ids)
    second = build_schedule(ids)
    assert first == second
    assert len(first) == 120 * len(CONDITIONS) * 2 == 4080
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in first}
    assert len(keys) == len(first)
    assert len({row["seed"] for row in first}) == len(first)


def test_pairwise_unbiased_d_preserves_negative_estimates() -> None:
    arrays = {
        name: np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
        for name in CONTROLLER_IDS
    }
    arrays[CONTROLLER_IDS[1]] = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
    matrix = pairwise_unbiased_distance_matrix(arrays)
    expected = np.mean(
        (arrays[CONTROLLER_IDS[0]][:, 0] - arrays[CONTROLLER_IDS[1]][:, 0])
        * (arrays[CONTROLLER_IDS[0]][:, 1] - arrays[CONTROLLER_IDS[1]][:, 1])
    )
    assert matrix[0, 1] == expected
    assert expected < 0
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)


def test_flat_and_whitened_geometry_are_symmetric() -> None:
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(16, 20))
    flat = flat_geometry(vectors)
    assert flat["algebraic_identity_max_error"] < 1e-12
    fit = fit_whitening(rng.normal(size=(64, 20)), regularization_fraction=0.10)
    white = whitened_geometry(vectors, fit)
    for matrix in (flat["normalized_euclidean"], white["normalized_euclidean"]):
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0)
    assert fit.regularization_fraction == 0.10
    assert fit.condition_number >= 1


def test_finite_secant_uses_pairwise_full_vocabulary_js() -> None:
    names = ("a", "b", "c")
    logits = {
        "a": np.asarray([[3.0, 0.0], [1.0, 1.0]]),
        "b": np.asarray([[0.0, 3.0], [1.0, 1.0]]),
        "c": np.asarray([[3.0, 0.0], [1.0, 1.0]]),
    }
    result = finite_secant_geometry(logits, names)
    matrix = result["sqrt_mean_js"]
    assert matrix[0, 1] > 0
    assert matrix[0, 2] == 0
    assert np.array_equal(matrix, matrix.T)


def test_prediction_calibrates_only_on_train_edges_and_qap_permutes_labels() -> None:
    names = tuple(f"c{i}" for i in range(6))
    train = names[:4]
    coordinates = np.arange(6, dtype=float)
    geometry = np.abs(coordinates[:, None] - coordinates[None, :])
    target = 0.2 + 0.5 * geometry
    np.fill_diagonal(target, 0)
    score = heldout_prediction(geometry, target, names, train)
    assert score["train_edge_count"] == 6
    assert score["heldout_edge_count"] == 9
    assert score["test_targets_used_in_fit"] is False
    assert score["heldout_rmse"] < 1e-12
    qap = qap_permutation(
        geometry, target, names, train, permutations=200, seed=91
    )
    assert qap["permutation_unit"] == "controller label"
    assert qap["observed_rho"] > 0.99


def test_qualification_is_not_correctness_ranked() -> None:
    source = {
        axis.axis_id: {
            "positive_commitment_validity": 1.0,
            "negative_commitment_validity": 1.0,
            "positive_semantic_evaluability": 1.0,
            "negative_semantic_evaluability": 1.0,
            "cross_disagreement": 0.2,
            "excess_disagreement": 0.1,
            "positive_negative_mean_token_ratio": 1.0,
            "positive_minus_negative_median_tokens": 0,
            "activation": {
                location: {"standardized_mean_gap": 0.5, "positive_gap_fraction": 0.8}
                for location in LOCATIONS
            },
        }
        for axis in SOURCE_AXES
    }
    controller = {
        name: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "semantic_change_rate": 0.2,
            "raw_sequence_change_rate": 0.5,
        }
        for name in CONTROLLER_IDS
    }
    bank = {
        "unit_norm_pass": True,
        "sign_pair_pass": True,
        "base_diversity_pass": True,
        "null_orthogonality_pass": True,
    }
    decision = qualification_decision(source, controller, bank)
    assert decision["qualified"]
    assert decision["accuracy_used_for_qualification"] is False
    assert decision["G_C_D_used_for_qualification"] is False


def test_classification_hierarchy() -> None:
    def record(rho: float, p: float, ratio: float, rmse: float) -> dict:
        return {
            "score": {
                "heldout_spearman_rho": rho,
                "rmse_ratio_to_constant": ratio,
                "heldout_rmse": rmse,
            },
            "qap": {"p_value_one_sided": p},
        }

    result = classify_q2(
        {
            "M0_FLAT": record(0.31, 0.04, 0.89, 1.0),
            "M1_WHITENED": record(0.50, 0.02, 0.70, 0.8),
            "M2_FINITE_SECANT": record(0.10, 0.5, 1.1, 1.2),
        }
    )
    assert result["classification"] == "Q2_PILOT_CONTROL_GEOMETRY_OUTPERFORMS_FLAT"


def test_edge_policy_holds_out_every_edge_touching_test_controller() -> None:
    names = tuple(f"c{i}" for i in range(16))
    split = edge_indices(names, names[:10])
    assert len(split["train"]) == 45
    assert len(split["heldout"]) == 75
    assert all(left < 10 and right < 10 for left, right in split["train"])
    assert all(left >= 10 or right >= 10 for left, right in split["heldout"])
