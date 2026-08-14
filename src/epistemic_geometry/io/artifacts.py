"""Human-readable and machine-readable run artifacts."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import yaml

from epistemic_geometry import __version__
from epistemic_geometry.config import RunConfig
from epistemic_geometry.reproducibility import (
    canonical_json,
    git_metadata,
    runtime_metadata,
    stable_digest,
)
from epistemic_geometry.types import ExperimentResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def render_summary(config: RunConfig, result: ExperimentResult) -> str:
    """Render the deliberately non-celebratory summary used in every run."""

    metrics = result.metrics
    lines = [
        "# Baseline vs Steering",
        "",
        "## EXPERIMENT",
        "baseline vs steering",
        "",
        "## STATUS",
        f"{config.experiment.stage.upper()} / {config.backend.type.upper()}",
        "",
        "## ITEMS",
        str(int(metrics["n_items"])),
        "",
        "## PRIMARY DESCRIPTIVE METRICS",
        f"BASELINE ACCURACY: {_format_metric(metrics['baseline_accuracy'])}",
        f"STEERED ACCURACY: {_format_metric(metrics['treatment_accuracy'])}",
        f"DELTA ACCURACY: {_format_metric(metrics['delta_accuracy'])}",
        f"ERROR CORRELATION (PHI): {_format_metric(metrics['error_correlation_phi'])}",
        f"ERROR JACCARD: {_format_metric(metrics['error_jaccard'])}",
        f"DISAGREEMENT RATE: {_format_metric(metrics['disagreement_rate'])}",
        f"DOUBLE FAULT: {_format_metric(metrics['double_fault'])}",
        f"RESCUE RATE: {_format_metric(metrics['rescue_rate'])}",
        f"DAMAGE RATE: {_format_metric(metrics['damage_rate'])}",
        f"PAIR ORACLE ACCURACY: {_format_metric(metrics['pair_oracle_accuracy'])}",
        f"COMPLEMENTARITY HEADROOM: {_format_metric(metrics['complementarity_headroom'])}",
        "",
        "## 2×2 PAIRED OUTCOMES",
        "| Baseline | Treatment | Count |",
        "|---|---|---:|",
        "| correct | correct | "
        f"{metrics['pair_counts'].get('baseline_correct__treatment_correct', 0)} |",
        "| correct | wrong | "
        f"{metrics['pair_counts'].get('baseline_correct__treatment_wrong', 0)} |",
        "| wrong | correct | "
        f"{metrics['pair_counts'].get('baseline_wrong__treatment_correct', 0)} |",
        "| wrong | wrong | "
        f"{metrics['pair_counts'].get('baseline_wrong__treatment_wrong', 0)} |",
        "",
        "## SCIENTIFIC CAUTION",
        "A useful intervention must be discussed as an accuracy/complementarity trade-off.",
        "Complementarity headroom is not an implementable ensemble result.",
        "",
        "IMPORTANT: MOCK RESULTS ARE SOFTWARE VALIDATION ONLY.",
        "They are not scientific evidence and do not establish the research question.",
        "",
    ]
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_run_artifacts(config: RunConfig, result: ExperimentResult) -> Path:
    """Write the standard run directory and return its absolute path."""

    config_hash = stable_digest(canonical_json(config.as_dict()))[:10]
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(config.output.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    base_name = f"{timestamp}_{config.experiment.name}_{config_hash}"
    run_dir = root / base_name
    collision_index = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{collision_index:02d}"
        collision_index += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "artifact_schema_version": 1,
        "package_version": __version__,
        "timestamp_utc": timestamp,
        "experiment_seed": config.experiment.seed,
        "backend_type": config.backend.type,
        "model_identifier": config.backend.model_path or config.backend.model_id,
        "benchmark": config.benchmark.type,
        "benchmark_path": config.benchmark.path,
        "config_hash": config_hash,
        "vector_hash": result.provenance.get("vector_hash"),
        "generation": {
            "do_sample": config.backend.do_sample,
            "temperature": config.backend.temperature,
            "max_new_tokens": config.backend.max_new_tokens,
        },
        **git_metadata(_repo_root()),
        **runtime_metadata(),
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8"
    )
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in result.predictions:
            handle.write(
                json.dumps(
                    {
                        "item_id": prediction.item_id,
                        "condition": prediction.condition,
                        "raw_output": prediction.raw_output,
                        "normalized_output": prediction.normalized_output,
                        "target": prediction.target,
                        "correct": prediction.correct,
                        "metadata": prediction.metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(result.metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render_summary(config, result), encoding="utf-8")
    return run_dir
