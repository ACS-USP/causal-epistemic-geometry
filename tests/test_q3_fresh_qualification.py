from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "scripts")

import execute_q3_fresh_qualification as executor  # noqa: E402
from run_q2_oos_v2_semantic import frozen_terminal_metadata  # noqa: E402

from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402
from epistemic_geometry.types import BackendOutput  # noqa: E402


def _router() -> dict[str, np.ndarray]:
    return {
        "pca_mean": np.zeros(4096),
        "pca_components": np.eye(8, 4096),
        "pca_scale": np.ones(8),
        "router_u": np.zeros((8, 2)),
        "router_v": np.zeros((8, 2)),
        "router_a": np.zeros(8),
        "router_b": np.arange(8, dtype=float),
    }


def test_router_selection_is_deterministic_argmax() -> None:
    order = [f"P{i}" for i in range(8)]
    assert executor.select_policy(np.zeros(4096), _router(), order) == "P7"


def test_router_tie_breaks_by_frozen_order() -> None:
    router = _router()
    router["router_b"][:] = 0
    order = [f"P{i}" for i in range(8)]
    assert executor.select_policy(np.zeros(4096), router, order) == "P0"


def test_router_nonfinite_falls_back_to_frozen_champion() -> None:
    router = _router()
    router["router_b"][0] = np.nan
    assert (
        executor.select_policy(np.zeros(4096), router, [f"P{i}" for i in range(8)])
        == "V4_DIRECTION_02_MEDIUM"
    )


def test_private_prompt_loader_has_reference_firewall(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    prompt = "predict"
    rows = [
        {
            "family_id": f"f-{index}",
            "prompt": prompt,
            "prompt_sha256": executor.sha256_text(prompt),
            "reference_repr": "1",
        }
        for index in range(300)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFERENCE_FIREWALL"):
        executor.load_prompts(path)


def test_collection_source_does_not_import_parser_or_evaluator() -> None:
    source = inspect.getsource(executor)
    assert "semantic_v3" not in source
    assert "evaluate_external_answer" not in source
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("semantic_v3" in name for name in imported)


def test_collect_refuses_without_preopen_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(executor, "build_backend", lambda _: called.append("model"))
    monkeypatch.setattr(executor, "verify_environment", lambda _: called.append("environment"))
    with pytest.raises(RuntimeError, match="PREOPEN_REQUIRED"):
        executor.collect(tmp_path, "model", tmp_path / "p", tmp_path / "r")
    assert called == []


def test_crash_safe_resume_skips_identical_completed_key(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    identity = {"experiment": "synthetic"}
    row = {"family_id": "f", "condition": "p", "rollout_index": 0, "seed": 7}
    journal = CrashSafeJournal(path, identity=identity, key_fields=executor.KEY_FIELDS)
    journal.append(row)
    journal.append(row)
    reopened = CrashSafeJournal(path, identity=identity, key_fields=executor.KEY_FIELDS)
    assert len(reopened.rows) == 1


def test_crash_safe_conflicting_duplicate_fails(tmp_path: Path) -> None:
    journal = CrashSafeJournal(
        tmp_path / "journal.jsonl",
        identity={"experiment": "synthetic"},
        key_fields=executor.KEY_FIELDS,
    )
    row = {"family_id": "f", "condition": "p", "rollout_index": 0, "seed": 7}
    journal.append(row)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        journal.append({**row, "seed": 8})


def test_operational_retry_set_is_narrow() -> None:
    assert executor._retryable(TimeoutError())
    assert executor._retryable(ConnectionError())
    assert not executor._retryable(RuntimeError("model outcome"))
    assert not executor._retryable(ValueError("scientific row"))


def test_terminal_repetition_is_persisted_failure() -> None:
    output = BackendOutput(
        raw_output="",
        metadata={
            "generated_token_count": 256,
            "terminal_policy": {"triggered": True},
        },
    )
    terminal = frozen_terminal_metadata(output)
    assert terminal["terminal_reason"] == "EXTREME_MECHANICAL_REPETITION_V1"
    assert terminal["commitment_valid_if_terminal_failure"] is False
    assert terminal["semantic_evaluable_if_terminal_failure"] is False
    assert terminal["binary_error_e_if_terminal_failure"] == 1


def test_terminal_hard_cap_is_persisted_failure() -> None:
    output = BackendOutput(
        raw_output="",
        metadata={
            "generated_token_count": 4096,
            "terminal_policy": {"triggered": False},
        },
    )
    terminal = frozen_terminal_metadata(output)
    assert terminal["terminal_reason"] == "max_new_tokens"
    assert terminal["binary_error_e_if_terminal_failure"] == 1


def test_generation_context_binds_layer_duration_and_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class Backend:
        device = torch.device("cpu")
        torch = torch

        @staticmethod
        def layer_module(layer: int):
            assert layer == 27
            return torch.nn.Identity()

        @staticmethod
        def _encode_item(_item):
            return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}, "x", "hash"

    meta = {
        "P": {
            "alpha": 2.0,
            "vector_sha256": "v",
            "layer": 27,
            "duration": "sustained_current_token",
        }
    }
    context, observed = executor.generation_context(
        Backend(),
        executor.model_item("f", "p"),
        "P",
        {"P": np.ones(4096)},
        meta,
        ["P"],
        _router(),
    )
    assert isinstance(context, executor.Gate6HookTrace)
    assert observed["layer"] == 27
    assert observed["duration"] == "sustained_current_token"
    assert observed["alpha"] == 2.0


def test_frozen_schedule_is_complete_after_materialization() -> None:
    if not executor.SCHEDULE.exists():
        pytest.skip("schedule is materialized after executor tests are committed")
    rows = executor.load_schedule()
    assert len(rows) == 6000
    assert len({(r["family_id"], r["condition"], r["rollout_index"]) for r in rows}) == 6000
    assert len({r["seed"] for r in rows}) == 6000


def test_collection_never_loads_confirmation_or_reserve() -> None:
    source = inspect.getsource(executor.collect)
    assert "load_prompts(private_prompts)" in source
    assert "confirmation.jsonl" not in source
    assert "reserve.jsonl" not in source
