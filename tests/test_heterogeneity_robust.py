import numpy as np

from epistemic_geometry.experiments.heterogeneity_robust import (
    exact_sign_test,
    node_jackknife_test,
    rank_cluster_regression,
    regularized_incomplete_beta,
    row_spearman,
    student_t_survival,
    studentized_mean_test,
)


def test_exact_sign_and_zero_handling() -> None:
    result = exact_sign_test([1.0] * 12 + [-1.0] * 4 + [0.0])
    assert result["nonzero"] == 16
    assert result["positives"] == 12
    assert np.isclose(result["p_value"], 0.0384063720703125)
    assert result["reject_0_05"]


def test_student_t_known_values() -> None:
    assert np.isclose(regularized_incomplete_beta(1.0, 1.0, 0.25), 0.25)
    assert np.isclose(student_t_survival(0.0, 15), 0.5)
    assert np.isclose(student_t_survival(1.75305, 15), 0.05, atol=2e-6)
    test = studentized_mean_test(np.arange(1.0, 17.0))
    assert test["reject_0_05"]


def test_row_spearman_and_cluster_rank_regression() -> None:
    x = np.random.default_rng(4).normal(size=(4, 6))
    geometry = {"MEDIUM": x, "STRONG": x + 1.0}
    outcome = {"MEDIUM": 2.0 * x, "STRONG": 3.0 * x}
    assert np.allclose(row_spearman(geometry, outcome), 1.0)
    regression = rank_cluster_regression(geometry, outcome)
    assert regression["slope"] > 0.0


def test_node_jackknife_detects_positive_symmetric_relation() -> None:
    rng = np.random.default_rng(9)
    coefficients = rng.standard_normal((16, 4))
    matrix = np.linalg.norm(coefficients[:, None] - coefficients[None, :], axis=2)
    geometry = {"MEDIUM": matrix, "STRONG": matrix}
    outcome = {"MEDIUM": matrix + 0.01 * rng.standard_normal(matrix.shape), "STRONG": matrix}
    for values in outcome.values():
        values[:] = 0.5 * (values + values.T)
        np.fill_diagonal(values, 0.0)
    result = node_jackknife_test(geometry, outcome)
    assert result["full_association"] > 0.99
    assert result["reject_0_05"]
