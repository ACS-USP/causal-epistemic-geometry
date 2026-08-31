from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epistemic_geometry.publication.q2.figure_tables import build_all_tables, write_tables
from epistemic_geometry.publication.q2.loaders import (
    ROOT,
    expected_source_hashes,
    load_sources,
    validate_frozen_sources,
)
from epistemic_geometry.publication.q2.pipeline import generate_visual_evidence
from epistemic_geometry.publication.q2.plotting import (
    FIGURE_SIZES,
    _coefficient_projection,
    qap_null_values,
)


@pytest.fixture(scope="module")
def q2_data() -> dict:
    return load_sources()


@pytest.fixture(scope="module")
def q2_tables(q2_data: dict) -> dict[str, pd.DataFrame]:
    return build_all_tables(q2_data)


def test_frozen_q2_sources_and_terminal_state(q2_data: dict) -> None:
    assert validate_frozen_sources() == expected_source_hashes()
    assert q2_data["estimands"]["classification"] == "Q2_V4_1_G2"
    assert q2_data["radial"]["R_shape"]["classification"] == "RS+"
    assert q2_data["radial"]["R_total"]["classification"] == "RT+"
    assert q2_data["forensic"]["classification"] == "Q2_V4_1_SEMANTIC_FORENSIC_CLEAN"
    assert q2_data["forensic"]["maximum_difference"] == 0.0
    assert q2_data["estimands"]["q3"] == "NOT_RUN"


def test_campaign_and_relational_cardinality(q2_data: dict, q2_tables: dict) -> None:
    completeness = q2_data["completeness"]
    assert completeness["expected_logical_rows"] == 37_800
    assert completeness["observed_logical_rows"] == 37_800
    assert completeness["missing"] == 0
    assert completeness["unexpected"] == 0
    assert completeness["duplicates"] == 0
    assert completeness["replacements"] == 0
    assert len(q2_data["estimands"]["controller_order"]) == 31
    pairwise = q2_tables["pairwise_geometry"]
    for metric in ("A0", "A1", "A2"):
        for shell in ("MEDIUM", "STRONG"):
            subset = pairwise[(pairwise["metric"] == metric) & (pairwise["shell"] == shell)]
            assert len(subset) == 465
            assert subset["pair_index"].tolist() == list(range(465))


def test_exact_full_sample_associations_and_qualification(q2_tables: dict) -> None:
    frame = q2_tables["association_summary"].set_index("metric")
    expected = {
        "A0": (0.5548224632268762, 0.5728142335797248, 0.5638183484033006),
        "A1": (0.5546810909321492, 0.5579408630791556, 0.5563109770056525),
        "A2": (0.44256928065741025, 0.43968766745532575, 0.4411284740563680),
    }
    for metric, values in expected.items():
        assert frame.loc[metric, "medium_rho"] == pytest.approx(values[0], abs=1e-15)
        assert frame.loc[metric, "strong_rho"] == pytest.approx(values[1], abs=1e-15)
        assert frame.loc[metric, "aggregate_full_sample_rho"] == pytest.approx(values[2], abs=1e-15)
        assert bool(frame.loc[metric, "qualifies"])
        assert bool(frame.loc[metric, "loo_all_shell_signs_positive"])
        assert frame.loc[metric, "qap_raw_p"] == 0.00002
        assert frame.loc[metric, "qap_maxT_p"] == 0.00002


def test_g3_contrasts_are_negative_and_fail(q2_tables: dict) -> None:
    frame = q2_tables["g3_contrasts"].set_index("contrast")
    assert frame.loc["A2_minus_A0", "observed_full_sample"] == pytest.approx(-0.1226898743469326)
    assert frame.loc["A2_minus_A1", "observed_full_sample"] == pytest.approx(-0.1151825029492845)
    assert frame.loc["A2_minus_A0", "bootstrap_q975"] < 0
    assert frame.loc["A2_minus_A1", "bootstrap_q975"] < 0
    assert (frame["superiority_maxT_p"] == 1.0).all()
    assert not frame["g3_pass"].any()


def test_radial_results_include_all_31_directions(q2_tables: dict) -> None:
    by_direction = q2_tables["radial_by_direction"]
    assert len(by_direction) == 31
    assert (by_direction["shape_strong_minus_medium"] > 0).all()
    assert (by_direction["total_strong_minus_medium"] > 0).all()
    summary = q2_tables["radial_summary"].set_index("endpoint")
    assert summary.loc["D_shape", "positive_directions"] == 31
    assert summary.loc["D_total", "positive_directions"] == 31
    assert summary.loc["D_shape", "observed_median_strong_minus_medium"] == pytest.approx(
        0.04405797101449277
    )
    assert summary.loc["D_total", "observed_median_strong_minus_medium"] == pytest.approx(
        0.043333333333333335
    )


def test_behavioral_context_matches_closeout(q2_tables: dict) -> None:
    frame = q2_tables["behavioral_context"].set_index("condition")
    assert frame.loc["BASELINE", "accuracy"] == pytest.approx(0.455)
    assert frame.loc["MEDIUM", "accuracy"] == pytest.approx(0.4675806451612903)
    assert frame.loc["STRONG", "accuracy"] == pytest.approx(0.4496236559139786)
    assert frame.loc["MEDIUM", "mean_C_vs_baseline"] == pytest.approx(0.0112278383140936)
    assert frame.loc["STRONG", "mean_C_vs_baseline"] == pytest.approx(0.0341283669579602)
    assert frame.loc["MEDIUM", "mean_D_total_vs_baseline"] == pytest.approx(0.0255913978494624)
    assert frame.loc["STRONG", "mean_D_total_vs_baseline"] == pytest.approx(0.0666666666666667)


def test_qap_reconstruction_matches_frozen_p_values(q2_data: dict, q2_tables: dict) -> None:
    null = qap_null_values(q2_data)
    associations = q2_tables["association_summary"].set_index("metric")
    for metric in ("A0", "A1", "A2"):
        observed = associations.loc[metric, "aggregate_full_sample_rho"]
        assert null[metric].shape == (50_000,)
        assert null[metric][0] == pytest.approx(observed, abs=1e-14)
        assert float(np.mean(null[metric] >= observed)) == 0.00002


def test_projection_uses_only_preoutcome_coefficients(q2_data: dict) -> None:
    original_coordinates, original_explained = _coefficient_projection(q2_data)
    altered = copy.deepcopy(q2_data)
    altered["estimands"] = {"semantic_outcomes_replaced": True}
    altered_coordinates, altered_explained = _coefficient_projection(altered)
    np.testing.assert_array_equal(original_coordinates, altered_coordinates)
    np.testing.assert_array_equal(original_explained, altered_explained)
    assert original_coordinates.shape == (31, 2)


def test_publication_package_has_no_livecodebench_or_q3_result_access() -> None:
    package = ROOT / "src/epistemic_geometry/publication/q2"
    forbidden = ("livecodebench", "q1_second_task", "q1-second-task")
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(any(token in name.lower() for token in forbidden) for name in imports)
    spec_text = (ROOT / "manuscript/figures/paper1_q2/FIGURE_SPEC.json").read_text().lower()
    assert "livecodebench" not in "\n".join(expected_source_hashes()).lower()
    assert '"q3_results_allowed": false' in spec_text
    assert all("/q3" not in source.lower() for source in expected_source_hashes())


def test_tables_are_byte_deterministic(tmp_path: Path, q2_tables: dict) -> None:
    first = write_tables(tmp_path / "first", q2_tables)
    second = write_tables(tmp_path / "second", q2_tables)
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()


def test_figure_package_is_byte_deterministic() -> None:
    first = generate_visual_evidence(ROOT)
    first_bytes = {
        (figure_id, suffix): path.read_bytes()
        for figure_id, outputs in first["figures"].items()
        for suffix, path in outputs.items()
    }
    second = generate_visual_evidence(ROOT)
    for figure_id, outputs in second["figures"].items():
        for suffix, path in outputs.items():
            assert path.read_bytes() == first_bytes[(figure_id, suffix)]
    assert FIGURE_SIZES["figure2"] == (7.2, 5.4)
    assert FIGURE_SIZES["figure3"] == (7.2, 4.5)
    assert FIGURE_SIZES["figure4"] == (7.2, 4.4)
