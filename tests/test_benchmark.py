import json

import pytest

from epistemic_geometry.benchmarks.jsonl import JsonlBenchmark
from epistemic_geometry.benchmarks.mock import MockBenchmark


def test_mock_benchmark_is_deterministic() -> None:
    first = MockBenchmark(5, seed=9).items()
    second = MockBenchmark(5, seed=9).items()
    assert first == second


def test_jsonl_validation_and_exact_normalization(tmp_path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        json.dumps({"id": "x", "prompt": "Answer A", "target": "A", "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    benchmark = JsonlBenchmark(path, allowed_targets=["A", "B"])
    assert benchmark.items()[0].id == "x"
    assert benchmark.parser.normalize("A\nexplanation") == "A"


def test_jsonl_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    record = {"id": "x", "prompt": "Answer A", "target": "A"}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        JsonlBenchmark(path, allowed_targets=["A"])


def test_jsonl_rejects_malformed_target(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "x", "prompt": "Answer", "target": "C"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in allowed targets"):
        JsonlBenchmark(path, allowed_targets=["A", "B"])
