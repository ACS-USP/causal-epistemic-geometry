from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import cross_block_shape

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_uniform_weighted_shape_reduces_to_frozen_r2_estimator() -> None:
    calibration = load_script(
        "q2_item_bootstrap_calibration_test",
        "calibrate_q2_oos_v2_item_bootstrap.py",
    )
    rng = np.random.Generator(np.random.PCG64DXSM(1209))
    fresh = rng.integers(0, 2, size=(4, 17, 2)).astype(np.float64)
    reference = rng.integers(0, 2, size=(6, 17, 2)).astype(np.float64)
    observed = calibration.weighted_shape(fresh, reference, np.ones(17))
    expected = cross_block_shape(fresh, reference)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)


def test_structural_index_audit_is_seed_deterministic() -> None:
    audit = load_script(
        "q2_item_bootstrap_audit_test",
        "audit_q2_oos_v2_item_bootstrap.py",
    )
    first = audit.structural_index_audit(items=30, resamples=100, seed=881920231)
    second = audit.structural_index_audit(items=30, resamples=100, seed=881920231)
    assert first == second
    assert first["unique_items"]["maximum"] <= 30
    assert first["multiplicity_effective_support"]["maximum"] <= 30


def test_continuous_weights_fail_closed_at_single_item_support() -> None:
    calibration = load_script(
        "q2_item_bootstrap_single_support_test",
        "calibrate_q2_oos_v2_item_bootstrap.py",
    )
    fresh = np.zeros((2, 5, 2), dtype=np.float64)
    reference = np.zeros((3, 5, 2), dtype=np.float64)
    weights = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.all(np.isnan(calibration.weighted_shape(fresh, reference, weights)))


def test_occurrence_weighting_matches_explicit_historical_bootstrap_sample() -> None:
    calibration = load_script(
        "q2_item_bootstrap_occurrence_test",
        "calibrate_q2_oos_v2_item_bootstrap.py",
    )
    rng = np.random.Generator(np.random.PCG64DXSM(9913))
    fresh = rng.integers(0, 2, size=(3, 11, 2)).astype(np.float64)
    reference = rng.integers(0, 2, size=(5, 11, 2)).astype(np.float64)
    indices = np.asarray([0, 0, 1, 3, 3, 3, 6, 7, 8, 8, 10])
    counts = np.bincount(indices, minlength=11)
    observed = calibration.weighted_shape(
        fresh,
        reference,
        counts,
        fixed_occurrence_count=len(indices),
    )
    expected = cross_block_shape(fresh[:, indices, :], reference[:, indices, :])
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)
