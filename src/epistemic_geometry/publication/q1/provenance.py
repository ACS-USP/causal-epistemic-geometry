"""Provenance manifests for deterministic Q1 figure tables and outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from epistemic_geometry.publication.q1.loaders import sha256

TABLE_SOURCES = {
    "figure2_genealogy": ["manuscript/figures/paper1/FIGURE_SPEC.json"],
    "confirmatory_item_profiles": [
        "review/q1_confirmatory_fixed_controllers/HOLDOUT_CONTENT_MANIFEST.json",
        "review/q1_confirmatory_fixed_controllers/journal_qwen.jsonl",
        "review/q1_confirmatory_fixed_controllers/journal_ministral.jsonl",
    ],
    "confirmatory_transition_decomposition": [
        "review/q1_confirmatory_fixed_controllers/HOLDOUT_CONTENT_MANIFEST.json",
        "review/q1_confirmatory_fixed_controllers/journal_qwen.jsonl",
        "review/q1_confirmatory_fixed_controllers/journal_ministral.jsonl",
        "review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json",
    ],
    "confirmatory_effects": ["review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json"],
    "confirmatory_safety": [
        "review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json",
        "review/q1_confirmatory_fixed_controllers/ANALYSIS_LOCK.json",
    ],
    "cross_domain_effects": [
        "review/gate9_selected_d75_evaluation/ESTIMANDS.json",
        "review/gate10_cross_domain_charcount/ESTIMANDS.json",
    ],
    "s1_duration_history": [
        "review/micro_q1/ESTIMANDS.json",
        "review/gate5_source_duration/ESTIMANDS.json",
    ],
    "s2_dose_calibration": [
        "review/gate7_fresh_l27_replication/ESTIMANDS.json",
        "review/gate8_l27_dose_calibration/DOSE_SUMMARY.csv",
    ],
    "s3_development_confirmation_controls": [
        "review/gate9_selected_d75_evaluation/ESTIMANDS.json",
        "review/gate13_1_all_layer_causal_atlas/ESTIMANDS.json",
        "review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json",
    ],
    "s5_ministral_invalidity": ["manuscript/data/posthoc_ministral_invalidity_aggregate.json"],
    "s7_loo_sensitivity": ["review/q1_confirmatory_fixed_controllers/LOO_SENSITIVITY.csv"],
    "s8_token_regimes": ["review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json"],
}

TABLE_RULES = {
    "figure2_genealogy": (
        "genealogy",
        "frozen chronological gate order",
        "direct ancestral Qwen stages",
    ),
    "confirmatory_item_profiles": (
        "confirmatory_item_profiles",
        "holdout manifest order",
        "all 57 items; invalid retained as e=1",
    ),
    "confirmatory_transition_decomposition": (
        "transition_decomposition",
        "component order fixed in spec",
        "all four rollout cross-products",
    ),
    "confirmatory_effects": (
        "confirmatory_effects",
        "Qwen then Ministral; meaningful then R0-R3",
        "all prospective confirmatory controls",
    ),
    "confirmatory_safety": (
        "confirmatory_safety",
        "Qwen then Ministral",
        "baseline and meaningful; no complete-case filtering",
    ),
    "cross_domain_effects": (
        "cross_domain_effects",
        "CRUXEval then long character count",
        "meaningful and all four domain-specific random controls",
    ),
    "s1_duration_history": (
        "duration_history",
        "Gate 4 then Gate 5 frozen condition order",
        "all frozen one-shot/sustained conditions represented",
    ),
    "s2_dose_calibration": (
        "dose_calibration",
        "D25,D50,D75,D100",
        "all Gate 8 doses plus Gate 7 full-dose validity",
    ),
    "s3_development_confirmation_controls": (
        "development_confirmation_controls",
        "model then stage then meaningful/R0-R3",
        "all four controls at each stage",
    ),
    "s5_ministral_invalidity": (
        "invalidity_taxonomy",
        "frozen taxonomy artifact order",
        "13 invalid meaningful rows; post-hoc only",
    ),
    "s7_loo_sensitivity": (
        "loo_sensitivity",
        "model then holdout manifest order",
        "all 57 leave-one-item-out rows per model",
    ),
    "s8_token_regimes": (
        "token_regimes",
        "model then frozen condition order",
        "all confirmatory conditions; descriptive only",
    ),
}

FIGURE_TABLES = {
    "figure1": [],
    "figure2": ["figure2_genealogy"],
    "figure3": [
        "confirmatory_item_profiles",
        "confirmatory_transition_decomposition",
        "confirmatory_effects",
    ],
    "figure4": ["confirmatory_effects", "confirmatory_safety"],
    "figure5": ["cross_domain_effects"],
    "s1": ["s1_duration_history"],
    "s2": ["s2_dose_calibration"],
    "s3": ["s3_development_confirmation_controls"],
    "s4": ["confirmatory_item_profiles"],
    "s5": ["s5_ministral_invalidity"],
    "s7": ["s7_loo_sensitivity"],
    "s8": ["s8_token_regimes"],
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
        "src/epistemic_geometry/publication/q1/loaders.py",
        "src/epistemic_geometry/publication/q1/figure_tables.py",
        "src/epistemic_geometry/publication/q1/plotting.py",
        "src/epistemic_geometry/publication/q1/provenance.py",
        "src/epistemic_geometry/publication/q1/pipeline.py",
        "scripts/generate_q1_paper_figures.py",
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
        sources = TABLE_SOURCES[name]
        records[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "derivation_function": function,
            "ordering_rule": ordering,
            "inclusion_rule": inclusion,
            "sources": {
                source: source_hashes.get(source, sha256(root / source)) for source in sources
            },
        }
    payload = {
        "schema_version": "1.0",
        "scientific_scope": "Q1_ONLY",
        "q2_semantic_sources": 0,
        "code_commit": _git_head(root),
        "code_state": "WORKTREE_PINNED_BY_CODE_HASHES",
        "code_hashes": code_hashes(root),
        "tables": records,
    }
    path = root / "manuscript/data/paper1/FIGURE_DATA_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def write_source_manifest(
    root: Path,
    *,
    figure_paths: dict[str, dict[str, Path]],
    table_manifest: Path,
    source_hashes: dict[str, str],
    spec: dict[str, Any],
) -> Path:
    path = root / "manuscript/figures/paper1/SOURCE_MANIFEST.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    historical = previous.get("historical_package") if isinstance(previous, dict) else None
    if historical is None and previous:
        historical = previous
    figure_specs = {entry["id"]: entry for entry in spec["figures"]}
    table_payload = json.loads(table_manifest.read_text())
    implemented: dict[str, Any] = {}
    for figure_id, outputs in figure_paths.items():
        entry = figure_specs[figure_id]
        table_names = FIGURE_TABLES[figure_id]
        tables = {name: table_payload["tables"][name] for name in table_names}
        direct_sources = entry.get("sources", [])
        source_paths = set(direct_sources)
        for table in tables.values():
            source_paths.update(table["sources"])
        implemented[figure_id] = {
            "title": entry["title"],
            "scientific_status": entry["scientific_status"],
            "item_ordering_rule": entry.get(
                "ordering", spec["global_rules"]["confirmatory_item_order"]
            ),
            "inclusion_policy": entry.get("inclusion", "AS_FROZEN_IN_FIGURE_SPEC"),
            "source_artifacts": {
                source: source_hashes.get(source, sha256(root / source))
                for source in sorted(source_paths)
            },
            "derived_tables": {
                name: {
                    "path": record["path"],
                    "sha256": record["sha256"],
                }
                for name, record in tables.items()
            },
            "notes": (
                "POST_HOC_DESCRIPTIVE_ONLY; recovered answers never enter frozen outcomes"
                if figure_id == "s5"
                else "AS_FROZEN_IN_FIGURE_SPEC"
            ),
            "outputs": {
                suffix: {
                    "path": str(output.relative_to(root)),
                    "sha256": sha256(output),
                }
                for suffix, output in outputs.items()
            },
        }
    payload = {
        "schema_version": "2.0",
        "scientific_scope": "Q1_ONLY",
        "contains_raw_outputs": False,
        "q2_semantic_sources": 0,
        "figure_spec": {
            "path": "manuscript/figures/paper1/FIGURE_SPEC.json",
            "sha256": sha256(root / "manuscript/figures/paper1/FIGURE_SPEC.json"),
        },
        "figure_data_manifest": {
            "path": str(table_manifest.relative_to(root)),
            "sha256": sha256(table_manifest),
        },
        "plotting_code_commit": _git_head(root),
        "plotting_code_state": "WORKTREE_PINNED_BY_CODE_HASHES",
        "plotting_code_hashes": code_hashes(root),
        "validated_frozen_source_artifacts": source_hashes,
        "implemented_figures": implemented,
        "omitted_supplements": {
            "s6": "heterogeneous stages; no comparable frozen regression target",
            "s9": "cross-controller/layer sign comparison risk",
        },
        "historical_package": historical,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = ["code_hashes", "write_data_manifest", "write_source_manifest"]
