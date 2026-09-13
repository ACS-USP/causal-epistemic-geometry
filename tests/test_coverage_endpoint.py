from __future__ import annotations

import pytest

from epistemic_geometry.analysis.coverage_endpoint import (
    bootstrap_coverage,
    calculate_coverage_endpoint,
)

EXPECTED = {"f1": ("x", "y"), "f2": ("x", "y")}


def row(family: str, cell: str, latent: str, *, a=(1, 0), b=(0, 1), t=(1, 1)):
    return {
        "family": family,
        "cell": cell,
        "latent": latent,
        "A": list(a),
        "B": list(b),
        "T": list(t),
    }


def balanced_rows():
    return [
        row(family, cell, f"{family}-{cell}-{index}")
        for family in EXPECTED
        for cell in EXPECTED[family]
        for index in range(2)
    ]


def test_per_latent_coverage_identities_and_global_g():
    report = calculate_coverage_endpoint(
        [row("f1", "x", "one", a=(1, 0), b=(0, 1), t=(1, 1))]
        + [row("f1", "y", "two", a=(1, 0), b=(0, 1), t=(1, 1))]
        + [row("f2", "x", "three", a=(1, 0), b=(0, 1), t=(1, 1))]
        + [row("f2", "y", "four", a=(1, 0), b=(0, 1), t=(1, 1))],
        expected_cells=EXPECTED,
    )
    assert report.values == {
        "V_AA": 1.0,
        "V_BB": 1.0,
        "V_TT": 1.0,
        "V_AB": 0.75,
        "V_AT": 1.0,
    }
    assert report.g == 0.75 - 1.0
    assert report["G"] == report.values["V_AB"] - max(report.values[name] for name in (
        "V_AA", "V_BB", "V_AT", "V_TT"
    ))


def test_equal_cell_then_equal_family_aggregation_differs_from_raw_mean():
    rows = [
        row("f1", "x", "f1-x-0", a=(1, 1), b=(1, 1), t=(1, 1)),
        row("f1", "y", "f1-y-0", a=(0, 0), b=(0, 0), t=(0, 0)),
        row("f1", "y", "f1-y-1", a=(0, 0), b=(0, 0), t=(0, 0)),
        row("f1", "y", "f1-y-2", a=(0, 0), b=(0, 0), t=(0, 0)),
        row("f2", "x", "f2-x-0", a=(0, 0), b=(0, 0), t=(0, 0)),
        row("f2", "y", "f2-y-0", a=(0, 0), b=(0, 0), t=(0, 0)),
    ]
    report = calculate_coverage_endpoint(rows, expected_cells=EXPECTED)
    raw_v_aa = sum(item["A"][0] for item in rows) / len(rows)
    assert report.values["V_AA"] == pytest.approx(0.25)
    assert report.values["V_AA"] != pytest.approx(raw_v_aa)
    assert report.aggregation["weights"]["cell_within_family"]["f1"] == {
        "x": 0.5,
        "y": 0.5,
    }


def test_incomplete_and_duplicate_compositions_raise():
    rows = balanced_rows()
    with pytest.raises(ValueError, match="incomplete composition"):
        calculate_coverage_endpoint(
            rows[:-1], expected_cells=EXPECTED, n_latents_per_cell=2
        )
    with pytest.raises(ValueError, match="duplicate latent identities"):
        calculate_coverage_endpoint(
            rows + [rows[0]], expected_cells=EXPECTED, n_latents_per_cell=2
        )


def test_stratified_latent_bootstrap_is_reproducible():
    rows = balanced_rows()
    first = bootstrap_coverage(
        rows,
        expected_cells=EXPECTED,
        n_latents_per_cell=2,
        n_replicates=80,
        seed=20260913,
    )
    second = bootstrap_coverage(
        rows,
        expected_cells=EXPECTED,
        n_latents_per_cell=2,
        n_replicates=80,
        seed=20260913,
    )
    assert first.to_dict() == second.to_dict()
    assert first.lower_percentile == 2.5
    assert set(first.lower_bounds) == {
        "V_AA", "V_BB", "V_TT", "V_AB", "V_AT", "G",
        "AB_minus_AA", "AB_minus_BB", "AB_minus_AT", "AB_minus_TT",
        "AB_minus_control",
    }
