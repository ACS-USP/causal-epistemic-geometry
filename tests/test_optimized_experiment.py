"""End-to-end optimized tiny-transformer artifact and resume checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from epistemic_geometry.config import OutputConfig, load_config  # noqa: E402
from epistemic_geometry.experiments.baseline_vs_steering import run_experiment  # noqa: E402
from epistemic_geometry.io.artifacts import RunInterrupted, validate_run_directory  # noqa: E402


def _config(root: Path):
    config = load_config(Path("configs/tiny_transformer_smoke.yaml"))
    return replace(config, output=OutputConfig(root=str(root), save_figures=False))


def test_optimized_tiny_resume_matches_uninterrupted_run(tmp_path) -> None:
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


def test_optimized_resume_refuses_changed_alpha(tmp_path) -> None:
    config = _config(tmp_path / "alpha")
    with pytest.raises(RunInterrupted) as interruption:
        run_experiment(config, stop_after_items=1)
    changed = replace(config, steering=replace(config.steering, alpha=0.75))
    with pytest.raises(ValueError, match="config hash"):
        run_experiment(changed, resume_dir=interruption.value.run_dir)
