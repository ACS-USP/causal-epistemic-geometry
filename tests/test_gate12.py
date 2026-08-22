from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epistemic_geometry.experiments import gate12


def test_fisher_energy_matches_explicit_matrix() -> None:
    logits = np.array([[0.3, -0.2, 1.1]], dtype=np.float64)
    jvp = np.array([[1.0, -2.0, 0.5]], dtype=np.float64)
    p = gate12.softmax64(logits)[0]
    matrix = np.diag(p) - np.outer(p, p)
    expected = float(jvp[0] @ matrix @ jvp[0])
    assert gate12.fisher_energy(logits, jvp)[0] == pytest.approx(expected)


def test_utility_slope_matches_log_softmax_central_difference() -> None:
    logits = np.array([[0.3, -0.2, 1.1]], dtype=np.float64)
    jvp = np.array([[1.0, -2.0, 0.5]], dtype=np.float64)
    target = np.array([1])
    epsilon = 1e-6

    def logp(values: np.ndarray) -> float:
        return float(values[0, 1] - np.logaddexp.reduce(values[0]))

    finite = (logp(logits + epsilon * jvp) - logp(logits - epsilon * jvp)) / (2 * epsilon)
    assert gate12.utility_slope(logits, jvp, target)[0] == pytest.approx(finite, rel=1e-8)


def test_local_kl_quadratic_identity() -> None:
    logits = np.array([[0.3, -0.2, 1.1]], dtype=np.float64)
    jvp = np.array([[1.0, -2.0, 0.5]], dtype=np.float64)
    epsilon = 1e-4
    q = gate12.fisher_energy(logits, jvp)[0]
    observed = 2 * gate12.categorical_kl(logits, logits + epsilon * jvp)[0] / epsilon**2
    assert observed == pytest.approx(q, rel=2e-4)


def test_fisher_cosine_identity_and_h_global_shift_null() -> None:
    logits = np.array([[0.2, 0.1, -0.4]])
    vector = np.array([[1.0, -2.0, 0.5]])
    assert gate12.fisher_cosine(logits, vector, vector)[0] == pytest.approx(1.0)
    assert gate12.fisher_energy(logits, np.ones_like(vector))[0] == pytest.approx(0.0)


def test_rank_selection_is_stable_and_domain_namespaced() -> None:
    values = [f"item-{index}" for index in range(20)]
    assert gate12.rank_utility_ids("CRUXEval", values) == gate12.rank_utility_ids(
        "CRUXEval", list(reversed(values))
    )
    assert gate12.rank_utility_ids("CRUXEval", values) != gate12.rank_utility_ids(
        "CHARCOUNT", values
    )


def test_canonical_answer_policy() -> None:
    assert gate12.canonical_answer("CRUXEval", "[1, 2]") == "FINAL: [1, 2]"
    assert gate12.canonical_answer("CHARCOUNT", 7) == "FINAL: 7"


def test_domain_centered_spearman_removes_domain_offset() -> None:
    left = np.array([1, 2, 101, 102], dtype=float)
    right = np.array([10, 20, -100, -90], dtype=float)
    domains = ["a", "a", "b", "b"]
    assert gate12.domain_centered_spearman(left, right, domains) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("control", "item", "domain", "expected"),
    [
        (True, True, True, "GATE12_UTILITY_ALIGNED_PULLBACK_SUPPORTED"),
        (
            True,
            False,
            True,
            "GATE12_PULLBACK_CONTROL_WITH_DOMAIN_LEVEL_UTILITY_ALIGNMENT",
        ),
        (True, False, False, "GATE12_PULLBACK_CONTROL_WITHOUT_UTILITY_PREDICTION"),
        (
            False,
            True,
            False,
            "GATE12_UTILITY_ALIGNMENT_WITHOUT_PULLBACK_CONTROL_PREDICTION",
        ),
        (False, False, False, "GATE12_LOCAL_GEOMETRY_NOT_PREDICTIVE"),
    ],
)
def test_classification(control: bool, item: bool, domain: bool, expected: str) -> None:
    assert (
        gate12.classify(
            control_supported=control,
            item_utility_supported=item,
            domain_utility_supported=domain,
        )
        == expected
    )


def test_historical_utility_target_requires_two_rollouts() -> None:
    assert gate12.historical_utility_target({0: False, 1: True}, {0: True, 1: True}) == 0.5
    with pytest.raises(ValueError):
        gate12.historical_utility_target({0: False}, {0: True, 1: True})


def test_forward_mode_scalar_jvp_matches_tiny_fixture_central_difference() -> None:
    torch = pytest.importorskip("torch")
    matrix = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    vector = torch.tensor([0.25, -0.75])

    def function(alpha):
        hidden = torch.tensor([0.4, -0.2]) + alpha * vector
        return torch.tanh(hidden) @ matrix

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(0.0), torch.tensor(1.0))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(function(dual))
    epsilon = 1e-4
    finite = (function(torch.tensor(epsilon)) - function(torch.tensor(-epsilon))) / (2 * epsilon)
    assert torch.allclose(tangent, finite, rtol=2e-3, atol=2e-3)
    assert primal.shape == tangent.shape == (2,)


def test_geometry_runner_has_no_historical_outcome_path() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_gate12_utility_aligned_pullback.py"
    ).read_text(encoding="utf-8")
    assert "gate9_selected_d75_evaluation/journal.jsonl" not in source
    assert "gate10_cross_domain_charcount/journal.jsonl" not in source
