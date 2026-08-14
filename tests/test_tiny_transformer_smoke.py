import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from epistemic_geometry.config import OutputConfig, load_config  # noqa: E402
from epistemic_geometry.experiments.baseline_vs_steering import run_experiment  # noqa: E402
from epistemic_geometry.io.artifacts import validate_run_directory  # noqa: E402


def test_tiny_random_transformer_uses_real_pipeline(tmp_path) -> None:
    config = load_config(Path("configs/tiny_transformer_smoke.yaml"))
    config = config.__class__(
        experiment=config.experiment,
        backend=config.backend,
        benchmark=config.benchmark,
        steering=config.steering,
        output=OutputConfig(root=str(tmp_path), save_figures=False),
    )
    run_dir = run_experiment(config)
    assert isinstance(run_dir, Path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["backend_type"] == "tiny_transformer"
    assert manifest["identity"]["backend"]["injected_test_model"] is True
    assert "TINY RANDOM TRANSFORMER RESULTS ARE SOFTWARE VALIDATION ONLY" in (
        run_dir / "summary.md"
    ).read_text()
    assert validate_run_directory(run_dir)["valid"] is True

