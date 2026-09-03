"""Provenance manifests for Q2 OOS visual evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from epistemic_geometry.publication.q2_oos.loaders import ROOT, TABLE_DIR, sha256

CODE_PATHS = (
    "scripts/derive_q2_oos_figure_tables.py",
    "scripts/generate_q2_oos_paper_figures.py",
    "src/epistemic_geometry/publication/q2_oos/loaders.py",
    "src/epistemic_geometry/publication/q2_oos/plotting.py",
    "src/epistemic_geometry/publication/q2_oos/provenance.py",
    "src/epistemic_geometry/publication/q2_oos/pipeline.py",
)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_manifests(
    source_hashes: dict[str, str],
    figure_paths: dict[str, dict[str, Path]],
) -> tuple[Path, Path]:
    table_records = {
        path.stem: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for path in sorted(TABLE_DIR.glob("*.csv"))
    }
    code_hashes = {path: sha256(ROOT / path) for path in CODE_PATHS}
    data_payload: dict[str, Any] = {
        "schema_version": "q2-oos-v2-figure-data-manifest-v1",
        "scientific_scope": "Q2_OOS_V2_CLOSED_RESULT",
        "classification": "Q2_OOS_V2_A0_PASS",
        "forensic_status": "Q2_OOS_V2_FORENSIC_CLEAN",
        "item_bootstrap_ruling": "Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED",
        "contains_raw_outputs": False,
        "contains_benchmark_text": False,
        "private_Dshape_source_sha256": (
            "a6a6b4889e2c86df04ce42c4415281dde82af0d2deb1347b8083015e95089ea5"
        ),
        "private_inputs_required_to_regenerate_tables": ["sealed D_SHAPE.npz"],
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "tables": table_records,
        "code_commit": git_head(),
        "code_state": "WORKTREE_PINNED_BY_CODE_HASHES",
    }
    data_path = ROOT / "manuscript/data/paper1_q2_oos/FIGURE_DATA_MANIFEST.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(data_payload, indent=2, sort_keys=True) + "\n")

    figure_records: dict[str, Any] = {}
    for figure_id, outputs in figure_paths.items():
        figure_records[figure_id] = {
            "outputs": {
                suffix: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for suffix, path in outputs.items()
            }
        }
    source_payload = {
        "schema_version": "q2-oos-v2-figure-source-manifest-v1",
        "scientific_scope": "Q2_OOS_V2_CLOSED_RESULT",
        "classification": "Q2_OOS_V2_A0_PASS",
        "forensic_status": "Q2_OOS_V2_FORENSIC_CLEAN",
        "public_clone_reproducible_from_committed_tables": True,
        "raw_text_in_git": False,
        "figure_spec": {
            "path": "manuscript/figures/paper1_q2_oos/FIGURE_SPEC.json",
            "sha256": sha256(ROOT / "manuscript/figures/paper1_q2_oos/FIGURE_SPEC.json"),
        },
        "figure_data_manifest": {
            "path": str(data_path.relative_to(ROOT)),
            "sha256": sha256(data_path),
        },
        "code_hashes": code_hashes,
        "figures": figure_records,
        "code_commit": git_head(),
        "code_state": "WORKTREE_PINNED_BY_CODE_HASHES",
    }
    source_path = ROOT / "manuscript/figures/paper1_q2_oos/SOURCE_MANIFEST.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(source_payload, indent=2, sort_keys=True) + "\n")
    return data_path, source_path


__all__ = ["write_manifests"]
