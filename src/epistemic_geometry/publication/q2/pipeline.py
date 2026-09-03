"""End-to-end deterministic generation of the Q2 visual evidence package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from epistemic_geometry.publication.q2.figure_tables import build_all_tables, write_tables
from epistemic_geometry.publication.q2.loaders import ROOT, load_sources
from epistemic_geometry.publication.q2.plotting import generate_all_figures
from epistemic_geometry.publication.q2.provenance import (
    write_data_manifest,
    write_source_manifest,
)


def generate_visual_evidence(root: Path = ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise RuntimeError("Q2 publication pipeline must run against the canonical repository")
    data = load_sources()
    tables = build_all_tables(data)
    table_paths = write_tables(root, tables)
    data_manifest = write_data_manifest(root, table_paths, data["source_hashes"])
    figure_paths = generate_all_figures(root, tables, data)
    source_manifest = write_source_manifest(
        root,
        figure_paths=figure_paths,
        table_manifest=data_manifest,
        source_hashes=data["source_hashes"],
        spec=data["spec"],
    )
    return {
        "tables": table_paths,
        "figures": figure_paths,
        "figure_data_manifest": data_manifest,
        "source_manifest": source_manifest,
    }


__all__ = ["generate_visual_evidence"]
