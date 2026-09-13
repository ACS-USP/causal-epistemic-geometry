from __future__ import annotations

import pytest

from epistemic_geometry.analysis.coverage_qualification import (
    FAMILY_CELLS,
    qualify_coverage,
    validate_composition,
)


def rows(a_fn=lambda _label: 1, t_fn=lambda _label: 1):
    return [
        {
            "family": family,
            "cell": cell,
            "latent_id": str(latent),
            "a1": a_fn(latent % 2),
            "a2": a_fn(latent % 2),
            "t1": t_fn(latent % 2),
            "t2": t_fn(latent % 2),
            "answer": latent % 2,
        }
        for family, cells in FAMILY_CELLS.items()
        for cell in cells
        for latent in range(8)
    ]


def adequate_rows():
    data = rows()
    # Keep a modest, shared error rate so the opportunity gate is informative
    # while the two rollout mean remains above the balanced constant baseline.
    for index, row in enumerate(data):
        if index % 4 == 0:
            row.update(a1=0, a2=0, t1=0, t2=0)
    return data


def test_complete_adequate_study_passes_and_reports_constant_baseline() -> None:
    report = qualify_coverage(adequate_rows(), delta=0.01, alpha=0.8)

    assert report.status == "PASS"
    assert report.composition.passed
    assert report.competence["MODREG-R"]["A"]["accuracy"] == 0.75
    assert report.competence["MODREG-R"]["A"]["constant_accuracy"] == 0.5
    assert report.opportunity["means"] == {"V_AA": 0.75, "V_AT": 0.75, "V_TT": 0.75}
    assert len(report.opportunity["per_latent"]) == 96


def test_constant_response_fails_against_label_derived_constant() -> None:
    report = qualify_coverage(rows(lambda _label: 0, lambda _label: 0), delta=0.05, alpha=0.2)

    assert report.status == "FAIL"
    assert any("constant predictor" in reason for reason in report.reasons)
    assert not report.competence["MODREG-R"]["A"]["passed"]


def test_low_opportunity_is_reported_with_conservative_h_lower() -> None:
    # Both views are correct on every rollout, leaving no opportunity for an
    # error based distinction even though competence itself is high.
    report = qualify_coverage(rows(), delta=0.05, alpha=0.2)

    assert report.status == "FAIL"
    assert report.opportunity["H"] == 0.0
    assert report.opportunity["H_lower"] == 0.0
    assert any("insufficient opportunity" in reason for reason in report.reasons)


def test_missing_latent_is_a_structured_composition_failure() -> None:
    data = rows()[:-1]

    report = qualify_coverage(data, delta=0.05, alpha=0.2)

    assert report.status == "FAIL"
    assert not report.composition.passed
    assert any("expected 96 rows" in reason for reason in report.reasons)


def test_replacing_frozen_family_or_cell_fails_composition() -> None:
    replaced_family = rows()
    replaced_family[0]["family"] = "OTHER-R"
    family_report = qualify_coverage(replaced_family, delta=0.05, alpha=0.2)
    assert not family_report.composition.passed
    assert any("expected families" in reason for reason in family_report.reasons)

    replaced_cell = rows()
    replaced_cell[0]["cell"] = "depth_20"
    cell_report = qualify_coverage(replaced_cell, delta=0.05, alpha=0.2)
    assert not cell_report.composition.passed
    assert any("expected cells" in reason for reason in cell_report.reasons)


def test_answer_may_be_an_arbitrary_hashable_value() -> None:
    data = rows()
    for row in data:
        row["answer"] = ("oracle", row["answer"])

    report = qualify_coverage(data, delta=0.05, alpha=0.2)

    assert report.composition.passed
    assert report.competence["MODREG-R"]["A"]["constant_prediction"] in {
        ("oracle", 0),
        ("oracle", 1),
    }


def test_malformed_binary_data_raises_in_strict_validator_and_fails_gate() -> None:
    data = rows()
    data[0]["a1"] = 2

    with pytest.raises(ValueError, match="binary"):
        validate_composition(data)
    report = qualify_coverage(data, delta=0.05, alpha=0.2)
    assert report.status == "FAIL"
    assert report.reasons[0].startswith("malformed data:")
