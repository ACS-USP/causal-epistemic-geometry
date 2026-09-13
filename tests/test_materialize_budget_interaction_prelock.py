from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from epistemic_geometry.benchmarks.reasoning.budget_interaction import NAMESPACE

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_budget_interaction_prelock.py"


def _script_module():
    spec = importlib.util.spec_from_file_location("budget_prelock_materializer", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load prelock materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_stage_a(path: Path) -> bytes:
    payload = {
        "manifests": {
            "synthetic/2048": {
                "items": [
                    {"latent_id": "synthetic-historical-a"},
                    {"latent_id": "synthetic-historical-b"},
                ]
            },
            "synthetic/4096": {
                "items": [
                    {"latent_id": "synthetic-historical-a"},
                    {"latent_id": "synthetic-historical-b"},
                ]
            },
        },
        "paired_budget_groups": {
            "synthetic": {
                "latent_ids": ["synthetic-historical-a", "synthetic-historical-b"]
            }
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_materializer_writes_deterministic_identity_and_provenance(tmp_path: Path) -> None:
    module = _script_module()
    source = tmp_path / "synthetic_stage_a.json"
    source_bytes = _synthetic_stage_a(source)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = module.materialize(source, first_dir)
    second = module.materialize(source, second_dir)

    names = module.OUTPUT_FILENAMES
    assert all(
        (first_dir / name).read_bytes() == (second_dir / name).read_bytes() for name in names
    )
    manifest = json.loads((first_dir / module.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    schedule = json.loads((first_dir / module.SCHEDULE_FILENAME).read_text(encoding="utf-8"))
    provenance = json.loads((first_dir / module.PROVENANCE_FILENAME).read_text(encoding="utf-8"))

    assert len(manifest) == provenance["N"] == 96
    assert len(schedule) == provenance["call_count"] == 768
    assert provenance["calls_per_cap"] == 384
    assert provenance["namespace"] == NAMESPACE
    assert provenance["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert provenance["extracted_id_count"] == 2
    assert {row["manifest_hash"] for row in manifest} == {provenance["manifest_hash"]}
    assert module._schedule_digest(schedule) == provenance["schedule_digest"]
    assert {"synthetic-historical-a", "synthetic-historical-b"}.isdisjoint(
        {row["latent_id"] for row in manifest}
    )
    assert first["provenance"] == second["provenance"] == provenance


def test_cli_reports_counts_without_printing_latent_ids(tmp_path: Path, capsys) -> None:
    module = _script_module()
    source = tmp_path / "synthetic_stage_a.json"
    _synthetic_stage_a(source)

    assert module.main([str(source), str(tmp_path / "prelock")]) == 0
    stdout = capsys.readouterr().out
    assert "synthetic-historical-a" not in stdout
    assert "synthetic-historical-b" not in stdout
    assert '"N": 96' in stdout
    assert '"call_count": 768' in stdout


def test_materializer_refuses_existing_artifacts_without_overwrite(tmp_path: Path) -> None:
    module = _script_module()
    source = tmp_path / "synthetic_stage_a.json"
    _synthetic_stage_a(source)
    output = tmp_path / "prelock"
    module.materialize(source, output)
    before = {
        name: (output / name).read_bytes()
        for name in module.OUTPUT_FILENAMES
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.materialize(source, output)

    after = {name: (output / name).read_bytes() for name in module.OUTPUT_FILENAMES}
    assert after == before
