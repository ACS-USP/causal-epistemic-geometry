"""Provenance manifests for deterministic Q2 figure tables and outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from epistemic_geometry.publication.q2.loaders import sha256

TABLE_SOURCES = {
    "pairwise_geometry": [
        "review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz",
        "review/q2_v4_1_prediction_lock/PREDICTION_MATRIX_METADATA.json",
        "review/q2_v4_1_semantic_execution/ESTIMANDS.json",
    ],
    "association_summary": [
        "review/q2_v4_1_semantic_execution/ESTIMANDS.json",
        "review/q2_v4_1_semantic_execution/BOOTSTRAP_INTERVALS.json",
    ],
    "g3_contrasts": [
        "review/q2_v4_1_semantic_execution/ESTIMANDS.json",
        "review/q2_v4_1_semantic_execution/BOOTSTRAP_INTERVALS.json",
    ],
    "radial_by_direction": [
        "review/q2_v4_1_semantic_execution/ESTIMANDS.json",
        "review/q2_v4_1_semantic_execution/RADIAL_RESULTS.json",
    ],
    "radial_summary": ["review/q2_v4_1_semantic_execution/RADIAL_RESULTS.json"],
    "behavioral_context": ["review/q2_v4_1_semantic_execution/ESTIMANDS.json"],
    "loo_robustness": ["review/q2_v4_1_semantic_execution/ESTIMANDS.json"],
}

TABLE_RULES = {
    "pairwise_geometry": (
        "relational_geometry",
        "metric, shell, frozen upper-triangle controller order",
        "all 465 controller pairs in both shells for A0/A1/A2",
    ),
    "association_summary": (
        "association_summary",
        "A0, A1, A2",
        "all frozen primary geometries",
    ),
    "g3_contrasts": (
        "g3_contrasts",
        "A2-A0, A2-A1",
        "both frozen superiority contrasts",
    ),
    "radial_by_direction": (
        "radial_by_direction",
        "frozen controller order",
        "all 31 directions; no outcome sorting",
    ),
    "radial_summary": (
        "radial_summary",
        "shape, total",
        "both independent frozen radial endpoints",
    ),
    "behavioral_context": (
        "behavioral_context",
        "baseline, MEDIUM, STRONG",
        "baseline reference and means across all 31 controllers per shell",
    ),
    "loo_robustness": (
        "loo_robustness",
        "A0/A1/A2 then frozen dropped-controller order",
        "all 31 delete-one-controller results per geometry",
    ),
}

FIGURE_TABLES = {
    "figure1": [],
    "figure2": ["pairwise_geometry", "association_summary"],
    "figure3": ["association_summary", "g3_contrasts"],
    "figure4": ["radial_by_direction", "radial_summary"],
    "s1": ["behavioral_context"],
    "s2": ["association_summary", "loo_robustness"],
}

FIGURE_DIRECT_SOURCES = {
    "figure1": [
        "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json",
        "review/q2_v4_spark1_presemantic/SPARK1_SUBSPACE_QUALIFICATION.json",
        "review/q2_v4_1_prediction_lock/Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json",
    ],
    "figure2": [],
    "figure3": [],
    "figure4": [],
    "s1": [],
    "s2": [
        "review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz",
        "review/q2_v4_1_prediction_lock/QAP_CONTROLLER_PERMUTATIONS.npy",
        "review/q2_v4_1_prediction_lock/QAP_SCHEDULE.json",
        "review/q2_v4_1_semantic_execution/ESTIMANDS.json",
    ],
}


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def code_hashes(root: Path) -> dict[str, str]:
    files = [
        "src/epistemic_geometry/publication/q2/loaders.py",
        "src/epistemic_geometry/publication/q2/figure_tables.py",
        "src/epistemic_geometry/publication/q2/plotting.py",
        "src/epistemic_geometry/publication/q2/provenance.py",
        "src/epistemic_geometry/publication/q2/pipeline.py",
        "scripts/generate_q2_paper_figures.py",
    ]
    return {relative: sha256(root / relative) for relative in files}


def write_data_manifest(
    root: Path,
    table_paths: dict[str, Path],
    source_hashes: dict[str, str],
) -> Path:
    records: dict[str, Any] = {}
    for name, path in table_paths.items():
        function, ordering, inclusion = TABLE_RULES[name]
        records[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "derivation_function": function,
            "ordering_rule": ordering,
            "inclusion_rule": inclusion,
            "sources": {source: source_hashes[source] for source in TABLE_SOURCES[name]},
        }
    payload = {
        "schema_version": "1.0",
        "scientific_scope": "Q2_V4_1_ONLY",
        "classification": "Q2_V4_1_G2",
        "radial_classifications": ["RS+", "RT+"],
        "contains_raw_outputs": False,
        "q1_livecodebench_sources": 0,
        "q3_results": 0,
        "code_commit": _git_head(root),
        "code_state": "WORKTREE_PINNED_BY_CODE_HASHES",
        "code_hashes": code_hashes(root),
        "tables": records,
    }
    path = root / "manuscript/data/paper1_q2/FIGURE_DATA_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_source_manifest(
    root: Path,
    *,
    figure_paths: dict[str, dict[str, Path]],
    table_manifest: Path,
    source_hashes: dict[str, str],
    spec: dict[str, Any],
) -> Path:
    table_payload = json.loads(table_manifest.read_text(encoding="utf-8"))
    figure_specs = {entry["id"]: entry for entry in spec["figures"]}
    figures: dict[str, Any] = {}
    for figure_id, outputs in figure_paths.items():
        table_names = FIGURE_TABLES[figure_id]
        source_paths = set(FIGURE_DIRECT_SOURCES[figure_id])
        for table_name in table_names:
            source_paths.update(TABLE_SOURCES[table_name])
        entry = figure_specs[figure_id]
        figures[figure_id] = {
            "title": entry["title"],
            "classification": entry["classification"],
            "dimensions_inches": entry["final_size_inches"],
            "supports": entry["supports"],
            "does_not_support": entry["does_not_support"],
            "source_artifacts": {source: source_hashes[source] for source in sorted(source_paths)},
            "derived_tables": {
                name: {
                    "path": table_payload["tables"][name]["path"],
                    "sha256": table_payload["tables"][name]["sha256"],
                }
                for name in table_names
            },
            "outputs": {
                suffix: {
                    "path": str(output.relative_to(root)),
                    "sha256": sha256(output),
                }
                for suffix, output in outputs.items()
            },
        }
    payload = {
        "schema_version": "1.0",
        "scientific_scope": "Q2_V4_1_ONLY",
        "classification": "Q2_V4_1_G2",
        "radial_classifications": ["RS+", "RT+"],
        "contains_raw_outputs": False,
        "public_clone_reproducible": True,
        "private_artifacts_required": [],
        "q1_livecodebench_sources": 0,
        "q3_results": 0,
        "figure_spec": {
            "path": str((root / "manuscript/figures/paper1_q2/FIGURE_SPEC.json").relative_to(root)),
            "sha256": sha256(root / "manuscript/figures/paper1_q2/FIGURE_SPEC.json"),
        },
        "figure_data_manifest": {
            "path": str(table_manifest.relative_to(root)),
            "sha256": sha256(table_manifest),
        },
        "code_commit": _git_head(root),
        "code_state": "WORKTREE_PINNED_BY_CODE_HASHES",
        "code_hashes": code_hashes(root),
        "figures": figures,
    }
    path = root / "manuscript/figures/paper1_q2/SOURCE_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = ["write_data_manifest", "write_source_manifest"]
