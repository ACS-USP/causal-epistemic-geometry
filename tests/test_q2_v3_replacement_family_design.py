from __future__ import annotations

import json
from pathlib import Path

from scripts.design_q2_v3_replacement_family import (
    build_inventory,
    read_static_records,
    static_features,
)

ROOT = Path(__file__).resolve().parents[1]


def test_static_reader_is_invariant_to_benchmark_output(tmp_path: Path) -> None:
    base = {
        "id": "sample_x",
        "official_index": 1,
        "code": "def f(x):\n    return str(int(x))",
        "input": "'7'",
        "dataset_repo": "fixture",
        "dataset_revision": "fixed",
    }
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    path_a.write_text(json.dumps({**base, "output": "'7'"}) + "\n", encoding="utf-8")
    path_b.write_text(json.dumps({**base, "output": "WRONG-SENTINEL"}) + "\n", encoding="utf-8")
    assert read_static_records(path_a) == read_static_records(path_b)
    assert "output" not in read_static_records(path_a)[0]


def test_static_feature_taxonomy_on_synthetic_programs() -> None:
    nested = {
        "id": "nested",
        "official_index": 0,
        "code": "def f(x):\n    return str(int(x.strip()))",
        "input": "'7'",
    }
    feature = static_features(nested)
    assert feature["INTERMEDIATE_DATAFLOW_COMPOSITION"]
    assert feature["TYPE_COERCION_SEMANTICS"]
    assert not feature["EXCEPTION_ERROR_PATH"]

    exceptional = {
        "id": "exceptional",
        "official_index": 1,
        "code": (
            "def f(x):\n    try:\n        return x\n"
            "    except ValueError:\n        return None"
        ),
        "input": "1",
    }
    assert static_features(exceptional)["EXCEPTION_ERROR_PATH"]


def test_repository_inventory_is_disjoint_and_outcome_free() -> None:
    inventory = build_inventory(ROOT)
    assert inventory["input_records"] == 336
    assert inventory["excluded_primary_panel_ids"] == 200
    assert inventory["excluded_prior_source_construction_ids"] == 24
    assert inventory["excluded_prior_source_validation_ids"] == 24
    assert inventory["available_disjoint_records"] == 88
    assert len(inventory["available_item_ids"]) == len(set(inventory["available_item_ids"]))
    assert inventory["output_field_used"] is False
    assert inventory["model_behavior_used"] is False
    assert inventory["correctness_used"] is False
