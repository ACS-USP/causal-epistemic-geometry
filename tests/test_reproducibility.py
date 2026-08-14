import json
from dataclasses import replace
from pathlib import Path

from epistemic_geometry.config import OutputConfig, load_config
from epistemic_geometry.experiments.baseline_vs_steering import run_experiment


def test_same_mock_config_has_identical_predictions_and_metrics(tmp_path) -> None:
    config = load_config(Path("configs/mock_smoke.yaml"))
    first = run_experiment(replace(config, output=OutputConfig(root=str(tmp_path / "first"))))
    second = run_experiment(replace(config, output=OutputConfig(root=str(tmp_path / "second"))))
    first_predictions = (first / "predictions.jsonl").read_text()
    second_predictions = (second / "predictions.jsonl").read_text()
    first_metrics = json.loads((first / "metrics.json").read_text())
    second_metrics = json.loads((second / "metrics.json").read_text())
    assert first_predictions == second_predictions
    assert first_metrics == second_metrics

