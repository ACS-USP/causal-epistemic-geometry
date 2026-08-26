from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_collector_has_no_semantic_scoring_import_or_call() -> None:
    source = (ROOT / "scripts/run_q2_v4_1_semantic.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "epistemic_geometry.benchmarks.external.semantic_v3" not in imported_from
    assert not any("evaluate_external_answer_v3" in name for name in imported_names)
    assert "DEFERRED_UNTIL_COMPLETE" in source


def test_frozen_semantic_schedule_contract_is_37800_rows() -> None:
    import scripts.analyze_q2_v4_1_semantic as analysis

    rows, conditions = analysis.load_schedule()
    keys = {(row["item_id"], row["condition"], int(row["rollout_index"])) for row in rows}
    assert len(rows) == 37_800
    assert len(keys) == 37_800
    assert len(conditions) == 63
    assert conditions[0] == "BASELINE"


def test_finite_shape_matches_explicit_two_rollout_formula() -> None:
    import scripts.analyze_q2_v4_1_semantic as analysis

    errors = np.asarray(
        [
            [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    observed = analysis.finite_shape(errors, np.arange(3))[0, 1]
    delta0 = errors[0, :, 0] - errors[1, :, 0]
    delta1 = errors[0, :, 1] - errors[1, :, 1]
    expected = 3.0 / 2.0 * (np.mean(delta0 * delta1) - np.mean(delta0) * np.mean(delta1))
    assert observed == expected
