from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "consolidate_q2_oos_v2_label_free",
    ROOT / "scripts/consolidate_q2_oos_v2_label_free.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def scalar_js(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for a_row, b_row in zip(left, right, strict=True):
        a_row = np.asarray(a_row, dtype=np.float64)
        b_row = np.asarray(b_row, dtype=np.float64)
        a_log = a_row - np.logaddexp.reduce(a_row)
        b_log = b_row - np.logaddexp.reduce(b_row)
        m_log = np.logaddexp(a_log, b_log) - np.log(2.0)
        a_p, b_p = np.exp(a_log), np.exp(b_log)
        values.append(
            0.5 * sum(float(p * (x - m)) for p, x, m in zip(a_p, a_log, m_log, strict=True))
            + 0.5
            * sum(float(p * (x - m)) for p, x, m in zip(b_p, b_log, m_log, strict=True))
        )
    return float(sum(values) / len(values))


def test_mean_js_matches_scalar_natural_log_reference() -> None:
    left = np.asarray([[1.0, -1.0, 0.5], [0.0, 2.0, -2.0]], dtype=np.float32)
    right = np.asarray([[-0.5, 1.5, 0.0], [1.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(
        MODULE.mean_js(left, right), scalar_js(left, right), rtol=1e-13, atol=1e-13
    )


def test_pairwise_js_is_symmetric_with_zero_diagonal() -> None:
    arrays = {
        "a": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "b": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "c": np.asarray([[0.5, 0.5], [0.5, 0.5]]),
    }
    matrix = MODULE.pairwise_js(list(arrays), arrays, workers=1)
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.diag(matrix), 0.0, rtol=0.0, atol=0.0)


def test_block_matrices_preserve_fresh_then_reference_order() -> None:
    matrix = np.arange(25).reshape(5, 5)
    fresh_fresh, fresh_reference = MODULE.block_matrices(matrix, 2)
    np.testing.assert_array_equal(fresh_fresh, matrix[:2, :2])
    np.testing.assert_array_equal(fresh_reference, matrix[:2, 2:])
