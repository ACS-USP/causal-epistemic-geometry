from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.freeze_finalization_timing_lock import freeze
from scripts.materialize_finalization_timing_prelock import materialize

from epistemic_geometry.steering.vector import vector_hash


def put(p: Path, x):
    p.write_text(json.dumps(x))
    return p


def test_freeze_is_outcome_free_and_refuses_overwrite(tmp_path: Path):
    source = put(tmp_path / "source.json", [{"latent_id": "old"}])
    out = tmp_path / "a"
    materialize({source: hashlib.sha256(source.read_bytes()).hexdigest()}, out)
    v = np.array([1.0, 2.0])
    np.save(tmp_path / "v.npy", v)
    candidate = {
        "model_repo": "x",
        "model_revision": "r",
        "tokenizer_repo": "x",
        "tokenizer_revision": "r",
        "dtype": "torch.bfloat16",
        "attention_backend": "sdpa",
        "vector_path": "v.npy",
        "vector_file_sha256": hashlib.sha256((tmp_path / "v.npy").read_bytes()).hexdigest(),
        "vector_canonical_sha256": vector_hash(v),
        "layer": 27,
        "eta": 1.0,
        "hook_scope": "sustained_current_token",
        "think_close_token_id": 151668,
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
    cp = put(tmp_path / "candidate.json", candidate)
    ctrl = put(tmp_path / "controller.json", {"layer": 27})
    draft = tmp_path / "draft.md"
    draft.write_text("draft")
    lock = freeze(out, cp, ctrl, draft, source_commit="abc")
    assert lock["status"] == "FROZEN_NOT_RUN" and lock["design"]["calls"] == 768
    assert set(json.loads((out / "LOCK.json").read_text())) >= {
        "journal_identity",
        "candidate_identity",
    }
    with pytest.raises(FileExistsError):
        freeze(out, cp, ctrl, draft)
