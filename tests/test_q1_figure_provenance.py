from __future__ import annotations

import json
from pathlib import Path

from epistemic_geometry.publication.q1.loaders import ROOT, sha256


def test_figure_data_manifest_hashes_match_generated_tables() -> None:
    path = ROOT / "manuscript/data/paper1/FIGURE_DATA_MANIFEST.json"
    payload = json.loads(path.read_text())
    assert payload["scientific_scope"] == "Q1_ONLY"
    assert payload["q2_semantic_sources"] == 0
    for record in payload["tables"].values():
        artifact = ROOT / record["path"]
        assert artifact.is_file()
        assert sha256(artifact) == record["sha256"]


def test_source_manifest_hashes_match_generated_outputs() -> None:
    path = ROOT / "manuscript/figures/paper1/SOURCE_MANIFEST.json"
    payload = json.loads(path.read_text())
    assert payload["scientific_scope"] == "Q1_ONLY"
    assert payload["q2_semantic_sources"] == 0
    assert payload["contains_raw_outputs"] is False
    for figure_id, figure in payload["implemented_figures"].items():
        assert set(figure["outputs"]) == {"pdf", "png", "svg"}
        assert figure["source_artifacts"] or figure_id == "figure1"
        if figure_id != "figure1":
            assert figure["derived_tables"]
        for output in figure["outputs"].values():
            artifact = ROOT / output["path"]
            assert artifact.is_file()
            assert sha256(artifact) == output["sha256"]


def test_manifest_preserves_historical_figure_package() -> None:
    payload = json.loads((ROOT / "manuscript/figures/paper1/SOURCE_MANIFEST.json").read_text())
    historical = payload["historical_package"]
    assert historical["generator"] == "scripts/generate_paper1_figures.py"
    assert historical["contains_raw_outputs"] is False


def test_no_q2_semantic_path_in_figure_manifests() -> None:
    paths = [
        ROOT / "manuscript/figures/paper1/FIGURE_SPEC.json",
        ROOT / "manuscript/data/paper1/FIGURE_DATA_MANIFEST.json",
        ROOT / "manuscript/figures/paper1/SOURCE_MANIFEST.json",
    ]
    forbidden = "q2_v4_1_semantic_execution"
    for path in paths:
        assert forbidden not in Path(path).read_text().lower()
