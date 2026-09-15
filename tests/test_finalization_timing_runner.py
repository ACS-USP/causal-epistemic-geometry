from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from epistemic_geometry.benchmarks.reasoning.finalization_timing import (
    build_manifest,
    build_schedule,
)
from epistemic_geometry.benchmarks.reasoning.finalization_timing_journal import (
    FinalizationTimingJournal,
)
from epistemic_geometry.benchmarks.reasoning.finalization_timing_runner import (
    candidate_identity_hash,
    run_serial_finalization_timing,
)
from epistemic_geometry.steering.vector import vector_hash
from epistemic_geometry.types import BackendOutput, Intervention, SteeringVector

_FAKE_VECTOR_PATH = Path(__file__).resolve()
_FAKE_VECTOR_FILE_SHA256 = hashlib.sha256(_FAKE_VECTOR_PATH.read_bytes()).hexdigest()
_FAKE_VECTOR_CANONICAL_SHA256 = vector_hash(np.ones(3))


def _intervention() -> Intervention:
    vector = SteeringVector(
        np.ones(3), 27, "fake", "none", hash=_FAKE_VECTOR_CANONICAL_SHA256
    )
    return Intervention(27, 0.75, "fake-vector", "last_token", vector)


def _candidate_identity() -> dict[str, object]:
    return {
        "model_repo": "fake/model",
        "model_revision": "fake-revision",
        "tokenizer_repo": "fake/tokenizer",
        "tokenizer_revision": "fake-tokenizer-revision",
        "dtype": "bf16",
        "attention_backend": "sdpa",
        "vector_path": str(_FAKE_VECTOR_PATH),
        "vector_file_sha256": _FAKE_VECTOR_FILE_SHA256,
        "vector_canonical_sha256": _FAKE_VECTOR_CANONICAL_SHA256,
        "layer": 27,
        "eta": 0.75,
        "hook_scope": "sustained_current_token",
        "decoding_config": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "enable_thinking": True,
            "prompt_mode": "plain",
            "inference_mode": "generation",
            "execution_mode": "serial_reference",
        },
    }


def _temp_candidate_identity(tmp_path, payload: bytes = b"frozen-vector") -> dict[str, object]:
    path = tmp_path / "vector.bin"
    path.write_bytes(payload)
    identity = _candidate_identity()
    identity["vector_path"] = str(path)
    identity["vector_file_sha256"] = hashlib.sha256(payload).hexdigest()
    return identity


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, bool]] = []
        self.contexts: list[tuple[str, bool]] = []
        self.active = False
        self.config = _candidate_identity()["decoding_config"]
        self.provenance_values = {
            "model_identifier": "fake/model",
            "model_revision": "fake-revision",
            "tokenizer_identifier": "fake/tokenizer",
            "tokenizer_revision": "fake-tokenizer-revision",
            "dtype": "bf16",
            "attention_backend": "sdpa",
        }

    def provenance(self) -> dict[str, object]:
        return dict(self.provenance_values)

    def steer_sustained_current_token(self, intervention):
        class Context:
            def __enter__(_self):
                assert not self.active
                self.active = True
                self.contexts.append((intervention.vector_id, True))

            def __exit__(_self, *_args):
                assert self.active
                self.active = False

        return Context()

    def generate_reasoning_view(self, view, *, sampling_seed, max_new_tokens):
        assert not self.active or view is not None
        self.calls.append((view.view_id, sampling_seed, max_new_tokens, self.active))
        return BackendOutput(
            raw_output="partial",
            metadata={
                "stop_reason": "max_new_tokens",
                "generation_seed": sampling_seed,
                "generated_token_ids": [151668],
            },
        )


def test_timing_runner_is_serial_and_scopes_d75_contexts() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    backend = _FakeBackend()
    records = run_serial_finalization_timing(
        backend,
        manifest,
        schedule,
        intervention=_intervention(),
        candidate_identity=_candidate_identity(),
        controller_provenance={
            "controller": "frozen-d75",
            "vector_file_sha256": _FAKE_VECTOR_FILE_SHA256,
        },
    )

    assert len(records) == len(schedule)
    assert [call[0] for call in backend.calls] == [
        f"{row['latent_id']}:canonical" for row in schedule
    ]
    assert [call[2] for call in backend.calls] == [row["cap"] for row in schedule]
    assert [call[1] for call in backend.calls] == [row["sampling_seed"] for row in schedule]
    assert [call[3] for call in backend.calls] == [row["condition"] == "D75" for row in schedule]
    assert len(backend.contexts) == sum(row["condition"] == "D75" for row in schedule)
    assert not backend.active
    assert all(record.parse_status == "TRUNCATED_NO_FINAL" for record in records)
    assert all(record.token_ids == (151668,) for record in records)
    assert all(record.metadata["condition"] in {"BASELINE", "D75"} for record in records)
    assert all(
        record.metadata["cap"] == record.generation_config["max_new_tokens"]
        for record in records
    )
    assert all(record.metadata["seed_regime"] == "MATCHED" for record in records)
    assert all(record.metadata["physical_generation_id"] for record in records)
    assert len({record.metadata["physical_generation_id"] for record in records}) == len(records)
    assert all(
        record.metadata["controller_provenance"]
        == {"controller": "frozen-d75", "vector_file_sha256": _FAKE_VECTOR_FILE_SHA256}
        for record in records
        if record.metadata["condition"] == "D75"
    )


def test_runner_rejects_tampered_schedule_identity_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    schedule[0] = {**schedule[0], "family": "FSM-R"}
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="schedule family"):
        run_serial_finalization_timing(
            backend,
            manifest,
            schedule,
            intervention=_intervention(),
            candidate_identity=_candidate_identity(),
            controller_provenance={"vector_file_sha256": _FAKE_VECTOR_FILE_SHA256},
        )
    assert backend.calls == []


def test_runner_requires_d75_intervention_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="require an Intervention"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            candidate_identity=_candidate_identity(),
        )
    assert backend.calls == []


def test_runner_requires_candidate_identity_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="candidate_identity is required"):
        run_serial_finalization_timing(backend, manifest, build_schedule(manifest))
    assert backend.calls == []


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda identity: identity.pop("dtype"), "missing fields"),
        (lambda identity: identity.update({"layer": True}), "layer"),
        (lambda identity: identity.update({"decoding_config": {}}), "decoding_config"),
    ],
)
def test_runner_rejects_malformed_candidate_identity_before_generation(mutator, message) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    identity = _candidate_identity()
    mutator(identity)
    with pytest.raises((TypeError, ValueError), match=message):
        run_serial_finalization_timing(
            backend, manifest, build_schedule(manifest), candidate_identity=identity
        )
    assert backend.calls == []


def test_runner_preserves_candidate_identity_on_baseline_and_d75_records() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    backend = _FakeBackend()
    identity = _candidate_identity()
    expected_hash = candidate_identity_hash(identity)
    records = run_serial_finalization_timing(
        backend,
        manifest,
        schedule,
        intervention=_intervention(),
        candidate_identity=identity,
        controller_provenance={
            "controller": "frozen-d75",
            "vector_file_sha256": _FAKE_VECTOR_FILE_SHA256,
        },
    )
    assert all(record.metadata["candidate_identity"] == identity for record in records)
    assert all(record.metadata["candidate_identity_hash"] == expected_hash for record in records)
    assert all(
        record.metadata["controller_provenance"] is None
        for record in records
        if record.metadata["condition"] == "BASELINE"
    )
    assert all(
        record.metadata["controller_provenance"]
        == {"controller": "frozen-d75", "vector_file_sha256": _FAKE_VECTOR_FILE_SHA256}
        for record in records
        if record.metadata["condition"] == "D75"
    )


def test_controller_provenance_cannot_substitute_for_candidate_identity() -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="candidate_identity is required"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            controller_provenance=_candidate_identity(),
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "journal_identity", [{"run": "fake"}, {"candidate_identity_hash": "wrong"}]
)
def test_runner_rejects_journal_identity_candidate_hash_before_generation(
    tmp_path, journal_identity
) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    journal = FinalizationTimingJournal(tmp_path / "journal.jsonl", identity=journal_identity)
    with pytest.raises(ValueError, match="candidate_identity_hash"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            candidate_identity=_candidate_identity(),
            controller_provenance={"vector_file_sha256": _FAKE_VECTOR_FILE_SHA256},
            journal=journal,
            journal_identity=journal_identity,
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_identifier", "other/model"),
        ("model_revision", "other-revision"),
        ("tokenizer_identifier", "other/tokenizer"),
        ("tokenizer_revision", "other-tokenizer-revision"),
        ("dtype", "float32"),
        ("attention_backend", "eager"),
    ],
)
def test_runner_rejects_backend_provenance_mismatch_before_generation(field, value) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    backend.provenance_values[field] = value
    with pytest.raises(ValueError, match="backend provenance"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            candidate_identity=_candidate_identity(),
            controller_provenance={"vector_file_sha256": _FAKE_VECTOR_FILE_SHA256},
        )
    assert backend.calls == []


def test_runner_rejects_decoding_config_mismatch_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    backend.config["top_p"] = 0.9
    with pytest.raises(ValueError, match="backend.config top_p"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            candidate_identity=_candidate_identity(),
            controller_provenance={"vector_file_sha256": _FAKE_VECTOR_FILE_SHA256},
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "kind",
    ["layer", "vector_layer", "alpha", "vector_hash", "token_scope"],
)
def test_runner_rejects_intervention_mismatch_before_generation(kind) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    intervention = _intervention()
    if kind == "layer":
        intervention = replace(intervention, layer=26)
    elif kind == "vector_layer":
        intervention = replace(intervention, vector=replace(intervention.vector, layer=26))
    elif kind == "alpha":
        intervention = replace(intervention, alpha=0.5)
    elif kind == "vector_hash":
        intervention = replace(intervention, vector=replace(intervention.vector, hash="wrong"))
    else:
        intervention = replace(intervention, token_scope="all_tokens")
    with pytest.raises(ValueError, match="intervention"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=intervention,
            candidate_identity=_candidate_identity(),
            controller_provenance={"vector_file_sha256": _FAKE_VECTOR_FILE_SHA256},
        )
    assert backend.calls == []


def test_runner_rejects_forged_declared_vector_hash_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    forged_vector = SteeringVector(
        np.zeros(3), 27, "fake", "none", hash=_FAKE_VECTOR_CANONICAL_SHA256
    )
    forged_intervention = Intervention(27, 0.75, "fake-vector", "last_token", forged_vector)
    with pytest.raises(ValueError, match="recomputed intervention vector hash"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=forged_intervention,
            candidate_identity=_candidate_identity(),
            controller_provenance={"vector_file_sha256": _FAKE_VECTOR_FILE_SHA256},
        )
    assert backend.calls == []


@pytest.mark.parametrize("file_sha", [None, "wrong-file-sha"])
def test_runner_requires_matching_controller_vector_file_hash_before_generation(file_sha) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    provenance = {} if file_sha is None else {"vector_file_sha256": file_sha}
    with pytest.raises(ValueError, match="vector_file_sha256"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            candidate_identity=_candidate_identity(),
            controller_provenance=provenance,
        )
    assert backend.calls == []


def test_runner_accepts_matching_local_vector_file_before_generation(tmp_path) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    identity = _temp_candidate_identity(tmp_path)
    records = run_serial_finalization_timing(
        backend,
        manifest,
        build_schedule(manifest),
        intervention=_intervention(),
        candidate_identity=identity,
        controller_provenance={"vector_file_sha256": identity["vector_file_sha256"]},
    )
    assert len(records) > 0
    assert backend.calls


def test_runner_rejects_tampered_local_vector_file_before_generation(tmp_path) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    identity = _temp_candidate_identity(tmp_path)
    Path(identity["vector_path"]).write_bytes(b"tampered-vector")
    with pytest.raises(ValueError, match="vector file SHA-256"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            candidate_identity=identity,
            controller_provenance={"vector_file_sha256": identity["vector_file_sha256"]},
        )
    assert backend.calls == []


def test_runner_rejects_missing_local_vector_file_before_generation(tmp_path) -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    identity = _temp_candidate_identity(tmp_path)
    Path(identity["vector_path"]).unlink()
    with pytest.raises(ValueError, match="vector_path"):
        run_serial_finalization_timing(
            backend,
            manifest,
            build_schedule(manifest),
            intervention=_intervention(),
            candidate_identity=identity,
            controller_provenance={"vector_file_sha256": identity["vector_file_sha256"]},
        )
    assert backend.calls == []
