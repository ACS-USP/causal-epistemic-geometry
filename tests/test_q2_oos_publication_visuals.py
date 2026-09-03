from __future__ import annotations

import csv
import json

import numpy as np

from epistemic_geometry.publication.q2_oos.loaders import ROOT, load_sources, sha256
from epistemic_geometry.publication.q2_oos.pipeline import generate_visual_evidence

ANALYSIS = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
    "Q2_OOS_V2_SEMANTIC_ANALYSIS.json"
)
DIAGNOSTIC = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
    "item_bootstrap_diagnostic/Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_RESULT.json"
)


def test_controller_table_reconciles_every_primary_value() -> None:
    data = load_sources()
    analysis = json.loads(ANALYSIS.read_text())
    table = data["tables"]["controller_associations"]
    assert table["controller_order"].tolist() == list(range(1, 17))
    assert table["primary_positive"].tolist() == [True] * 16
    np.testing.assert_allclose(
        table["equal_shell_r_i"].to_numpy(), analysis["primary"]["r_i"], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        (table["medium_rho"] + table["strong_rho"]) / 2,
        table["equal_shell_r_i"],
        rtol=0,
        atol=1e-15,
    )
    assert analysis["primary"]["positive_count"] == 16
    assert analysis["primary"]["exact_binomial"]["p_value"] == 1.52587890625e-05


def test_global_and_fresh_fresh_values_reconcile() -> None:
    data = load_sources()
    analysis = json.loads(ANALYSIS.read_text())
    global_table = data["tables"]["global_associations"].set_index("metric")
    for metric in ("A0", "A1", "A2", "D2"):
        archived = analysis["secondary"]["global_fresh_reference"][metric]["global"]
        assert np.isclose(global_table.loc[metric, "equal_shell_rho"], archived["equal_shell_mean"])
        assert np.isclose(global_table.loc[metric, "medium_rho"], archived["shell"]["MEDIUM"])
        assert np.isclose(global_table.loc[metric, "strong_rho"], archived["shell"]["STRONG"])
    fresh_fresh = data["tables"]["fresh_fresh_summary"].iloc[0]
    archived_ff = analysis["secondary"]["fresh_fresh_node_jackknife"]
    assert fresh_fresh["role"] == "SECONDARY_ONLY_CANNOT_RESCUE_PRIMARY"
    assert np.isclose(fresh_fresh["association"], archived_ff["full_association"])
    assert np.isclose(
        fresh_fresh["jackknife_standard_error"], archived_ff["jackknife_standard_error"]
    )
    assert data["tables"]["fresh_fresh_pairs"].shape[0] == 120


def test_bootstrap_plot_uses_authorized_non_ci_language() -> None:
    data = load_sources()
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    table = data["tables"]["bootstrap_diagnostic"]
    item_rows = table[table["object"].str.contains("item_resampling")]
    assert (
        item_rows["interpretation"] == "PANEL_PERTURBATION_SENSITIVITY_NOT_CONVENTIONAL_CI"
    ).all()
    assert diagnostic["ruling"] == "Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED"
    assert diagnostic["primary_oos_classification_changed"] is False


def test_generated_figure_manifest_matches_outputs() -> None:
    manifest_path = ROOT / "manuscript/figures/paper1_q2_oos/SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["classification"] == "Q2_OOS_V2_A0_PASS"
    assert manifest["forensic_status"] == "Q2_OOS_V2_FORENSIC_CLEAN"
    assert manifest["raw_text_in_git"] is False
    assert set(manifest["figures"]) == {"main", "s1", "s2", "s3", "s4", "s5"}
    for figure in manifest["figures"].values():
        assert set(figure["outputs"]) == {"svg", "pdf", "png"}
        for output in figure["outputs"].values():
            path = ROOT / output["path"]
            assert path.exists()
            assert sha256(path) == output["sha256"]


def test_release_safe_tables_contain_no_raw_content_fields() -> None:
    table_dir = ROOT / "manuscript/data/paper1_q2_oos/derived_figure_tables"
    forbidden = {"prompt", "response", "generated_text", "reference_answer", "correct"}
    for path in table_dir.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            headers = {value.lower() for value in next(reader)}
        assert forbidden.isdisjoint(headers)


def test_notebook_is_executed_and_contains_all_public_figures() -> None:
    notebook = json.loads((ROOT / "notebooks/q2_oos_visual_story.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["execution_count"] for cell in code_cells] == list(range(1, 9))
    assert not any(
        output["output_type"] == "error" for cell in code_cells for output in cell["outputs"]
    )
    images = [
        output
        for cell in code_cells
        for output in cell["outputs"]
        if "image/png" in output.get("data", {})
    ]
    assert len(images) == 5
    narrative = "".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "one fresh controller" in narrative
    assert "item-panel perturbation sensitivity distribution" in narrative
    assert "random-subspace specificity" in narrative


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
