"""Crash-safe Q1 V3 physical journal and resume tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from epistemic_geometry.benchmarks.reasoning import runner
from epistemic_geometry.benchmarks.reasoning.calibration import generate_stage_a_manifests
from epistemic_geometry.benchmarks.reasoning.journal import PhysicalGenerationJournal
from epistemic_geometry.config import load_config
from epistemic_geometry.types import BackendOutput


def _identity() -> dict[str, object]:
    return {
        "phase": "stage_a_screen",
        "manifest_hash": "manifest",
        "config_hash": "config",
        "source_commit": "commit",
    }


def _trajectory(latent_id: str, rollout_index: int, value: int = 1) -> dict[str, object]:
    return {
        "latent_id": latent_id,
        "view_id": f"{latent_id}:canonical",
        "family": "FSM-R",
        "cell": "length_4",
        "target": value,
        "rollout_index": rollout_index,
        "sampling_seed": 17,
        "physical_generation_id": f"physical-{latent_id}-{rollout_index}",
        "source_max_budget": 2048,
        "source_raw_text": "FINAL: 1",
        "source_token_ids": [1, 2, 3],
        "source_metadata": {},
        "derived_records": {},
    }


def test_journal_roundtrip_conflict_and_truncated_tail(tmp_path) -> None:
    path = tmp_path / "physical_journal.jsonl"
    journal = PhysicalGenerationJournal(path, identity=_identity())
    row = _trajectory("latent-a", 0)
    journal.append(row)
    journal.append(row)
    assert list(journal.rows) == [("latent-a", 0)]

    path.write_bytes(path.read_bytes() + b'{"journal_version":"broken"')
    recovered = PhysicalGenerationJournal(path, identity=_identity())
    assert list(recovered.rows) == [("latent-a", 0)]
    assert recovered.quarantined_tail is not None
    assert json.loads(path.read_text().splitlines()[0])["physical_key"] == ["latent-a", 0]

    with pytest.raises(ValueError, match="conflicting duplicate"):
        recovered.append({**row, "target": 2})


def test_valid_final_json_without_newline_is_normalized(tmp_path) -> None:
    path = tmp_path / "physical_journal.jsonl"
    journal = PhysicalGenerationJournal(path, identity=_identity())
    journal.append(_trajectory("latent-a", 0))
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    recovered = PhysicalGenerationJournal(path, identity=_identity())
    assert recovered.quarantined_tail is None
    assert path.read_bytes().endswith(b"\n")


def test_prefix_runner_resume_matches_uninterrupted_run(tmp_path, monkeypatch) -> None:
    manifests = generate_stage_a_manifests([("FSM-R", "length_4")], seed=31)
    payload = {
        "manifests": {
            f"{split.family}/{split.cell}/{split.reasoning_budget}": split.to_record()
            for split in manifests
        }
    }
    manifest_path = tmp_path / "stage_a.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config("configs/q1_v3_reasoning_instrument.example.yaml")

    @dataclass
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "<think>done</think>\nFINAL: 4" if ids else ""

    class FakeBackend:
        tokenizer = FakeTokenizer()

        def __init__(self, fail_after: int | None = None):
            self.calls = 0
            self.fail_after = fail_after

        def generate_reasoning_view(self, view, *, sampling_seed, max_new_tokens):
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("simulated process interruption")
            return BackendOutput(
                raw_output="<think>done</think>\nFINAL: 4",
                metadata={
                    "generated_token_ids": list(range(max_new_tokens)),
                    "generation_seed": sampling_seed,
                },
            )

        def provenance(self):
            return {"model": "fake"}

    interrupted_backend = FakeBackend(fail_after=2)
    monkeypatch.setattr(runner, "build_backend", lambda _config: interrupted_backend)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.run_baseline_calibration(
            config, manifest_path, tmp_path / "resumed", max_items=2,
        )
    journal_path = tmp_path / "resumed" / "physical_journal.jsonl"
    assert len(journal_path.read_text().splitlines()) == 2

    resume_backend = FakeBackend()
    monkeypatch.setattr(runner, "build_backend", lambda _config: resume_backend)
    resumed = runner.run_baseline_calibration(
        config, manifest_path, tmp_path / "resumed", max_items=2,
    )
    clean_backend = FakeBackend()
    monkeypatch.setattr(runner, "build_backend", lambda _config: clean_backend)
    clean = runner.run_baseline_calibration(
        config, manifest_path, tmp_path / "clean", max_items=2,
    )
    assert (resumed / "rollouts.jsonl").read_text() == (clean / "rollouts.jsonl").read_text()
    assert (resumed / "outcomes.json").read_text() == (clean / "outcomes.json").read_text()
    assert resume_backend.calls == 2
    assert json.loads((resumed / "manifest.json").read_text())["physical_generation_count"] == 4


def test_resume_refuses_changed_config(tmp_path, monkeypatch) -> None:
    manifests = generate_stage_a_manifests([("FSM-R", "length_4")], seed=31)
    payload = {
        "manifests": {
            f"{split.family}/{split.cell}/{split.reasoning_budget}": split.to_record()
            for split in manifests
        }
    }
    manifest_path = tmp_path / "stage_a.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config("configs/q1_v3_reasoning_instrument.example.yaml")

    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "FINAL: 4" if ids else ""

    class FakeBackend:
        tokenizer = FakeTokenizer()

        def generate_reasoning_view(self, view, *, sampling_seed, max_new_tokens):
            del view, sampling_seed
            return BackendOutput(
                raw_output="FINAL: 4",
                metadata={"generated_token_ids": list(range(max_new_tokens))},
            )

        def provenance(self):
            return {"model": "fake"}

    monkeypatch.setattr(runner, "build_backend", lambda _config: FakeBackend())
    runner.run_baseline_calibration(config, manifest_path, tmp_path / "run", max_items=1)
    changed_backend = replace(config.backend, temperature=0.7)
    changed = replace(config, backend=changed_backend)
    with pytest.raises(ValueError, match="provenance|identity"):
        runner.run_baseline_calibration(changed, manifest_path, tmp_path / "run", max_items=1)
