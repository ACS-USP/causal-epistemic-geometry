from __future__ import annotations

import hashlib
import json

import pytest
from scripts.materialize_finalization_timing_prelock import materialize


def _write(path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materializer_uses_only_hashed_manifest_sources(tmp_path) -> None:
    flat = tmp_path / "flat.json"
    stage = tmp_path / "stage.json"
    flat_hash = _write(flat, [{"latent_id": "old-flat"}])
    stage_hash = _write(
        stage,
        {"manifests": [{"items": [{"latent_id": "old-stage"}]}]},
    )
    result = materialize({flat: flat_hash, stage: stage_hash}, tmp_path / "output")
    assert result["provenance"]["N"] == 192
    assert result["provenance"]["call_count"] == 768
    assert result["provenance"]["excluded_id_count"] == 2
    baseline_seeds = {
        row["sampling_seed"]
        for row in result["schedule"]
        if row["condition"] == "BASELINE"
    }
    d75_seeds = {
        row["sampling_seed"]
        for row in result["schedule"]
        if row["condition"] == "D75"
    }
    assert baseline_seeds == d75_seeds


def test_materializer_refuses_mismatched_or_existing_sources(tmp_path) -> None:
    source = tmp_path / "flat.json"
    digest = _write(source, [{"latent_id": "old"}])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize({source: "0" * 64}, output)
    materialize({source: digest}, output)
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize({source: digest}, output)
