"""End-to-end generation of the Q2 OOS visual evidence package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from epistemic_geometry.publication.q2_oos.loaders import ROOT, load_sources
from epistemic_geometry.publication.q2_oos.plotting import generate_all
from epistemic_geometry.publication.q2_oos.provenance import write_manifests


def generate_visual_evidence(root: Path = ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise RuntimeError("Q2 OOS publication pipeline requires the canonical repository")
    data = load_sources()
    figures = generate_all(root, data)
    data_manifest, source_manifest = write_manifests(data["source_hashes"], figures)
    return {
        "figures": figures,
        "figure_data_manifest": data_manifest,
        "source_manifest": source_manifest,
    }


__all__ = ["generate_visual_evidence"]
