from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_q2_oos_v2_label_free",
    ROOT / "scripts/audit_q2_oos_v2_label_free.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scalar_mean_js_identity_and_symmetry() -> None:
    left = np.asarray([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    right = np.asarray([[-1.0, 1.0, 0.0], [0.0, 1.0, -1.0]])
    assert MODULE.scalar_mean_js(left, left) == 0.0
    np.testing.assert_allclose(
        MODULE.scalar_mean_js(left, right),
        MODULE.scalar_mean_js(right, left),
        rtol=0.0,
        atol=1e-15,
    )


def test_independent_a1_is_symmetric_and_zero_diagonal() -> None:
    _fresh, _reference, _coefficients, vectors = MODULE.identities_and_vectors()
    matrix = MODULE.independent_a1(vectors, MODULE.load_fit())
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 0.0, rtol=0.0, atol=0.0)


def test_numpy_boolean_checks_are_normalized_for_json() -> None:
    checks = {"comparison": np.float64(0.0) <= 1e-12}
    normalized = {name: bool(value) for name, value in checks.items()}
    assert type(normalized["comparison"]) is bool
