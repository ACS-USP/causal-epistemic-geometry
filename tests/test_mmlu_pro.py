from __future__ import annotations

import json

import pytest

import epistemic_geometry.reproducibility as reproducibility
from epistemic_geometry.benchmarks.mmlu_pro import MMLUProBenchmark, row_to_item
from epistemic_geometry.benchmarks.splits import create_mmlu_pro_split_manifest
from epistemic_geometry.config import (
    BackendConfig,
    BenchmarkConfig,
    ConfigError,
    ExperimentConfig,
    OutputConfig,
    RunConfig,
    SteeringConfig,
)
from epistemic_geometry.reproducibility import require_remote_hf_execution


def _row(index: int, category: str = "category-a") -> dict:
    return {
        "question_id": f"q-{index:04d}",
        "question": "Which option is correct?",
        "options": ["wrong", "correct", "other", "last"],
        "answer_index": 1,
        "category": category,
        "src": "fixture",
        "cot_content": "must never enter the prompt",
    }


def test_mmlu_row_is_exact_label_and_ignores_cot() -> None:
    item = row_to_item(_row(1), "validation")
    assert item.id == "validation:q-0001"
    assert item.target == "B"
    assert "must never enter" not in item.prompt
    assert item.metadata["candidate_labels"] == ["A", "B", "C", "D"]


def test_mmlu_fixture_adapter_and_strict_prompt() -> None:
    benchmark = MMLUProBenchmark.from_rows([_row(1), _row(2, "category-b")])
    assert len(benchmark) == 2
    assert benchmark.parser.parse("B").status == "OK"
    assert benchmark.parser.parse("I think B").status == "AMBIGUOUS"
    assert benchmark.provenance()["dataset_revision"] == "fixture"


def test_split_manifest_is_deterministic_and_label_free(tmp_path) -> None:
    rows = [_row(index, "a" if index % 2 else "b") for index in range(1024)]
    benchmark = MMLUProBenchmark.from_rows(rows, split="test")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = create_mmlu_pro_split_manifest(benchmark, first_path, seed=20260816)
    second = create_mmlu_pro_split_manifest(benchmark, second_path, seed=20260816)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["sizes"] == {
        "dev_calibration": 512,
        "dev_evaluation": 512,
        "confirmatory_holdout": 0,
    }
    ids = [item_id for values in first["splits"].values() for item_id in values]
    assert len(ids) == len(set(ids))
    raw = json.dumps(first)
    assert "answer_index" not in raw
    assert "correct" not in raw


def test_development_config_cannot_access_confirmatory_holdout() -> None:
    with pytest.raises(ConfigError, match="CONFIRMATORY_HOLDOUT"):
        RunConfig(
            experiment=ExperimentConfig("blocked", "development", 1),
            backend=BackendConfig("mock"),
            benchmark=BenchmarkConfig(
                type="mmlu_pro",
                split="confirmatory_holdout",
                dataset_id="TIGER-Lab/MMLU-Pro",
            ),
            steering=SteeringConfig(),
            output=OutputConfig(),
        )


def test_real_hf_operations_are_refused_outside_runpod(monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("CEG_EXECUTION_PROFILE", raising=False)
    with pytest.raises(RuntimeError, match="remote-only"):
        require_remote_hf_execution("test operation")


def test_real_hf_operations_allow_explicit_spark1_profile(monkeypatch) -> None:
    monkeypatch.setenv("CEG_EXECUTION_PROFILE", "SPARK1")
    monkeypatch.setenv("HF_HOME", "/srv/shared/hf-cache")
    monkeypatch.setattr(
        reproducibility,
        "remote_execution_context",
        lambda: {
            "hostname": "spark1",
            "pwd": "/home/gabriel.alexandre/projects/ceg-q2-v4-presemantic",
            "HF_HOME": "/srv/shared/hf-cache",
        },
    )
    require_remote_hf_execution("test operation")
