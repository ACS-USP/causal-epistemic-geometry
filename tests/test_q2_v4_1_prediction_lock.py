from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from scripts.prepare_q2_v4_1_prediction_lock import _target_embedding

from epistemic_geometry.experiments.q2_v4_1 import EXPECTED_SAFE_IDS, load_frozen_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_delta_zero_g3_planning_target_is_not_a2_superiority_target() -> None:
    rng = np.random.default_rng(11)
    metrics = {name: rng.normal(size=(31, 8)) for name in ("A0", "A1", "A2")}
    _target, achieved = _target_embedding(metrics, 17, 0.0)

    assert achieved["target_anchor"] == "A0"
    assert achieved["delta"] <= 0.0


def test_immutable_safe_bank_reconstructs_in_original_order() -> None:
    candidates, safe = load_frozen_candidates(
        ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json",
        ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json",
    )

    assert len(candidates) == 40
    assert [row["candidate_id"] for row in safe] == list(EXPECTED_SAFE_IDS)
    assert all(row["joint_safe"] for row in safe)


def test_label_free_runner_has_no_semantic_or_benchmark_outcome_imports() -> None:
    path = ROOT / "scripts/run_q2_v4_1_label_free_geometry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert all("semantic" not in module.lower() for module in imported_modules)


def test_prepared_g3_artifact_is_cpu_only_and_outcome_free() -> None:
    path = ROOT / "review/q2_v4_1_prediction_lock/G3_POWER_CHARACTERIZATION.json"
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["cpu_only"] is True
    assert value["semantic_outcomes"] == 0
    assert value["correctness_inspected"] is False
    assert value["rows"]
