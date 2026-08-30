from __future__ import annotations

import pandas as pd
import pytest

from epistemic_geometry.publication.q1.figure_tables import (
    _error,
    build_all_tables,
    confirmatory_item_profiles,
    write_tables,
)
from epistemic_geometry.publication.q1.loaders import ROOT, expected_source_hashes, load_sources
from epistemic_geometry.publication.q1.plotting import (
    FIGURE2_FINAL_SIZE,
    FIGURE2_FONT_SIZES,
    figure3_annotation_data,
    format_figure3_annotation,
)


@pytest.fixture(scope="module")
def q1_data() -> dict:
    missing = [path for path in expected_source_hashes() if not (ROOT / path).is_file()]
    if missing:
        pytest.skip(
            "private/hash-pinned Q1 publication sources are not present in this clone"
        )
    return load_sources()


@pytest.fixture(scope="module")
def q1_tables(q1_data: dict) -> dict[str, pd.DataFrame]:
    return build_all_tables(q1_data)


def test_invalid_or_unevaluable_rows_retain_frozen_error() -> None:
    fixture = {
        "correct": False,
        "commitment_valid": False,
        "semantic_evaluable": False,
    }
    assert _error(fixture) == 1
    with pytest.raises(RuntimeError, match="invalid/unevaluable"):
        _error({**fixture, "correct": True})


def test_item_profiles_use_all_items_in_manifest_order(
    q1_data: dict, q1_tables: dict[str, pd.DataFrame]
) -> None:
    frame = q1_tables["confirmatory_item_profiles"]
    assert len(frame) == 57 * 2 * 2
    for model in ("Qwen", "Ministral"):
        for condition in ("BASELINE", "MEANINGFUL_FIXED"):
            subset = frame[(frame["model_role"] == model) & (frame["condition"] == condition)]
            assert subset["item_id"].tolist() == q1_data["item_ids"]
            assert set(subset["q_hat_error"]) <= {0.0, 0.5, 1.0}


def test_itemwise_plot_order_is_invariant_to_journal_row_order(q1_data: dict) -> None:
    reordered = {
        **q1_data,
        "qwen_rows": list(reversed(q1_data["qwen_rows"])),
        "ministral_rows": list(reversed(q1_data["ministral_rows"])),
    }
    observed = confirmatory_item_profiles(reordered)
    for model in ("Qwen", "Ministral"):
        for condition in ("BASELINE", "MEANINGFUL_FIXED"):
            subset = observed[
                (observed["model_role"] == model) & (observed["condition"] == condition)
            ]
            assert subset["item_id"].tolist() == q1_data["item_ids"]


def test_transition_decomposition_uses_cross_rollouts_and_reconciles(
    q1_data: dict, q1_tables: dict[str, pd.DataFrame]
) -> None:
    frame = q1_tables["confirmatory_transition_decomposition"]
    for model in ("Qwen", "Ministral"):
        subset = frame[frame["model_role"] == model].set_index("component")
        assert subset["fraction"].sum() == pytest.approx(1.0, abs=1e-12)
        point = q1_data["confirmatory"]["models"][model]["estimands"]["MEANINGFUL_FIXED"]
        assert subset.loc["rescue", "fraction"] == pytest.approx(point["rescue"])
        assert subset.loc["damage", "fraction"] == pytest.approx(point["damage"])
        assert set(subset["rollout_convention"]) == {"all_four_cross_products"}


def test_all_four_prospective_nulls_are_in_every_confirmatory_model(
    q1_tables: dict[str, pd.DataFrame],
) -> None:
    frame = q1_tables["confirmatory_effects"]
    expected = {"RANDOM_R0", "RANDOM_R1", "RANDOM_R2", "RANDOM_R3"}
    for model in ("Qwen", "Ministral"):
        subset = frame[(frame["model_role"] == model) & (frame["controller_kind"] == "random")]
        assert set(subset["condition"]) == expected


def test_figure3_annotation_uses_observed_estimands_and_frozen_intervals(
    q1_data: dict, q1_tables: dict[str, pd.DataFrame]
) -> None:
    effects = q1_tables["confirmatory_effects"]
    values = figure3_annotation_data(effects, q1_data["confirmatory"])
    qwen = effects[effects["model_role"] == "Qwen"].set_index("condition")
    canonical = q1_data["confirmatory"]["models"]["Qwen"]

    observed_c = canonical["estimands"]["MEANINGFUL_FIXED"]["C"]
    observed_random_mean = float(qwen.loc["RANDOM_MEAN", "C"])
    assert values["observed_c"] == pytest.approx(observed_c, abs=1e-15)
    assert values["observed_delta_c"] == pytest.approx(
        observed_c - observed_random_mean, abs=1e-15
    )

    intervals = canonical["intervals"]
    assert values["c_ci_lower"] == intervals["C_meaningful"]["q025"]
    assert values["c_ci_upper"] == intervals["C_meaningful"]["q975"]
    assert values["delta_c_ci_lower"] == intervals["delta_C_nullmean"]["q025"]
    assert values["delta_c_ci_upper"] == intervals["delta_C_nullmean"]["q975"]

    annotation = format_figure3_annotation(values)
    assert "Meaningful C = 0.054" in annotation
    assert "Contrast vs random mean: ΔC = 0.039" in annotation
    assert "95% bootstrap CI: [0.014, 0.097]" in annotation
    assert "95% bootstrap CI: [0.006, 0.075]" in annotation
    assert "Meaningful C = 0.052" not in annotation
    assert "ΔC = 0.037" not in annotation


def test_figure2_is_authored_at_readable_final_paper_size() -> None:
    assert FIGURE2_FINAL_SIZE == (7.2, 5.7)
    assert min(FIGURE2_FONT_SIZES.values()) >= 7.0
    assert FIGURE2_FONT_SIZES["title"] >= 10.0
    assert FIGURE2_FONT_SIZES["gate"] >= 8.0


def test_derived_tables_are_byte_deterministic(
    tmp_path, q1_tables: dict[str, pd.DataFrame]
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = write_tables(first_root, q1_tables)
    second = write_tables(second_root, q1_tables)
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
