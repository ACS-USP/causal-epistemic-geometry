import json
from dataclasses import replace
from pathlib import Path

from epistemic_geometry.config import OutputConfig, load_config
from epistemic_geometry.experiments.baseline_vs_steering import run_experiment


def _config(output_root: Path):
    config = load_config(Path("configs/mock_smoke.yaml"))
    return replace(config, output=OutputConfig(root=str(output_root), save_figures=False))


def test_full_mock_experiment_writes_paired_artifacts(tmp_path) -> None:
    run_dir = run_experiment(_config(tmp_path / "runs"))
    assert isinstance(run_dir, Path)
    for filename in (
        "config_resolved.yaml",
        "manifest.json",
        "predictions.jsonl",
        "metrics.json",
        "summary.md",
    ):
        assert (run_dir / filename).exists()
    records = [
        json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()
    ]
    baseline_ids = [record["item_id"] for record in records if record["condition"] == "baseline"]
    treatment_ids = [record["item_id"] for record in records if record["condition"] == "steered"]
    assert baseline_ids == treatment_ids
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "error_correlation_phi" in metrics
    assert "COMPLEMENTARITY HEADROOM" in (run_dir / "summary.md").read_text()
    assert "MOCK RESULTS ARE SOFTWARE VALIDATION ONLY" in (run_dir / "summary.md").read_text()
