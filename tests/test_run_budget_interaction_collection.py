from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.run_budget_interaction_collection import (
    LockValidationError,
    load_and_validate_lock,
)

from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import identity_hash
from epistemic_geometry.benchmarks.reasoning.budget_interaction_runner import (
    candidate_identity_hash,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest
from epistemic_geometry.steering.vector import vector_hash

NAMESPACE = "Q1-V3-BUDGET-CAUSAL-INTERACTION-V1"


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _fixture(tmp_path: Path) -> Path:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    vector_path = tmp_path / "vector.npy"
    np.save(vector_path, values)
    candidate = {
        "model_repo": "fake/model",
        "model_revision": "revision",
        "tokenizer_repo": "fake/model",
        "tokenizer_revision": "revision",
        "dtype": "torch.bfloat16",
        "attention_backend": "sdpa",
        "vector_path": "vector.npy",
        "vector_file_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "vector_canonical_sha256": vector_hash(values),
        "layer": 27,
        "eta": 1.0,
        "hook_scope": "sustained_current_token",
        "decoding_config": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "enable_thinking": True,
            "prompt_mode": "chat",
            "inference_mode": "generation",
            "execution_mode": "serial_reference",
        },
    }
    manifest = [{"identity_hash": "fixture-latent"}]
    schedule = [{"schedule": "fixture"}]
    manifest_hash = stable_digest(NAMESPACE, "MANIFEST", "fixture-latent")
    schedule_digest = stable_digest(NAMESPACE, "SCHEDULE", canonical_json(schedule))
    manifest_raw = _write_json(tmp_path / "MANIFEST.json", manifest)
    schedule_raw = _write_json(tmp_path / "SCHEDULE.json", schedule)
    provenance = {"manifest_hash": manifest_hash, "schedule_digest": schedule_digest}
    _write_json(tmp_path / "PROVENANCE.json", provenance)
    artifact_sha256 = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("MANIFEST.json", "SCHEDULE.json", "PROVENANCE.json")
    }
    candidate_hash = candidate_identity_hash(candidate)
    journal = {
        "candidate_identity_hash": candidate_hash,
        "manifest_hash": manifest_hash,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "schedule_digest": schedule_digest,
        "schedule_sha256": hashlib.sha256(schedule_raw).hexdigest(),
    }
    lock = {
        "artifact_sha256": artifact_sha256,
        "candidate_identity": candidate,
        "candidate_identity_hash": candidate_hash,
        "journal_identity": journal,
        "journal_identity_hash": identity_hash(journal),
    }
    lock_path = tmp_path / "LOCK.json"
    _write_json(lock_path, lock)
    return lock_path


def test_load_and_validate_lock_checks_all_identities(tmp_path: Path) -> None:
    lock_path = _fixture(tmp_path)

    loaded = load_and_validate_lock(lock_path)

    assert loaded["candidate_identity"]["dtype"] == "torch.bfloat16"
    assert loaded["vector_path"] == tmp_path / "vector.npy"
    np.testing.assert_array_equal(loaded["vector_values"], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("target", ["MANIFEST.json", "SCHEDULE.json", "PROVENANCE.json"])
def test_bad_artifact_hash_blocks_loading(tmp_path: Path, target: str) -> None:
    lock_path = _fixture(tmp_path)
    lock = json.loads(lock_path.read_text())
    lock["artifact_sha256"][target] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(LockValidationError, match="SHA-256 mismatch"):
        load_and_validate_lock(lock_path)


def test_bad_vector_hash_blocks_loading(tmp_path: Path) -> None:
    lock_path = _fixture(tmp_path)
    lock = json.loads(lock_path.read_text())
    lock["candidate_identity"]["vector_canonical_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(LockValidationError, match="candidate identity hash"):
        load_and_validate_lock(lock_path)
