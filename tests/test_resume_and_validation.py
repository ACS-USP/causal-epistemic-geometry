import json
from dataclasses import replace
from pathlib import Path

import pytest

from epistemic_geometry.config import OutputConfig, load_config
from epistemic_geometry.experiments.baseline_vs_steering import run_experiment
from epistemic_geometry.io.artifacts import RunInterrupted, RunSession, validate_run_directory
from epistemic_geometry.types import Prediction


def _config(root: Path):
    config = load_config(Path("configs/mock_smoke.yaml"))
    return replace(config, output=OutputConfig(root=str(root), save_figures=False))


def test_interrupted_resume_matches_uninterrupted_run(tmp_path) -> None:
    config = _config(tmp_path / "resumable")
    with pytest.raises(RunInterrupted) as interruption:
        run_experiment(config, stop_after_items=2)
    resumed = run_experiment(config, resume_dir=interruption.value.run_dir)
    uninterrupted = run_experiment(_config(tmp_path / "uninterrupted"))
    assert (resumed / "predictions.jsonl").read_text() == (
        uninterrupted / "predictions.jsonl"
    ).read_text()
    assert json.loads((resumed / "metrics.json").read_text()) == json.loads(
        (uninterrupted / "metrics.json").read_text()
    )
    assert validate_run_directory(resumed)["valid"] is True


def test_resume_refuses_changed_alpha(tmp_path) -> None:
    config = _config(tmp_path / "alpha")
    with pytest.raises(RunInterrupted) as interruption:
        run_experiment(config, stop_after_items=1)
    changed = replace(config, steering=replace(config.steering, alpha=0.75))
    with pytest.raises(ValueError, match="config hash"):
        run_experiment(changed, resume_dir=interruption.value.run_dir)


def test_resume_refuses_identity_vector_change(tmp_path) -> None:
    config = _config(tmp_path / "vector")
    identity = {
        "config_hash": "same",
        "experiment_seed": 1,
        "backend_type": "mock",
        "model_identifier": "mock",
        "vector_hash": "vector-a",
        "intervention": {"alpha": 1.0},
        "backend": {"model_fingerprint": "fingerprint-a"},
    }
    session = RunSession.create(config, identity)
    changed = dict(identity, vector_hash="vector-b")
    with pytest.raises(ValueError, match="provenance"):
        RunSession.resume(session.run_dir, config, changed)


def test_duplicate_prediction_and_truncated_tail_are_safe(tmp_path) -> None:
    config = _config(tmp_path / "rows")
    identity = {
        "config_hash": "same",
        "experiment_seed": 1,
        "backend_type": "mock",
        "model_identifier": "mock",
        "vector_hash": "vector-a",
        "intervention": {"alpha": 1.0},
        "backend": {"model_fingerprint": "fingerprint-a"},
    }
    session = RunSession.create(config, identity)
    prediction = Prediction("item-1", "baseline", "A", "A", "A", True)
    session.append_prediction(prediction)
    with pytest.raises(ValueError, match="overwrite"):
        session.append_prediction(prediction)
    with session.predictions_path.open("ab") as handle:
        handle.write(b'{"item_id":"broken"')
    rows = session.existing_predictions()
    assert list(rows) == [("item-1", "baseline")]
    assert list(session.run_dir.glob("predictions.quarantine*.jsonl"))

