"""Human-readable and machine-readable run artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from epistemic_geometry import __version__
from epistemic_geometry.config import RunConfig, load_config
from epistemic_geometry.metrics import bootstrap_paired_metrics, compute_paired_metrics
from epistemic_geometry.reproducibility import (
    canonical_json,
    git_metadata,
    runtime_metadata,
    stable_digest,
)
from epistemic_geometry.types import ExperimentResult, Prediction


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


def resolved_config_hash(config: RunConfig) -> str:
    """Hash resolved scientific settings, excluding operational output paths."""

    scientific_config = config.as_dict()
    scientific_config.pop("output", None)
    return stable_digest(canonical_json(scientific_config))[:10]


def _prediction_payload(prediction: Prediction, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": prediction.item_id,
        "condition": prediction.condition,
        "raw_output": prediction.raw_output,
        "normalized_output": prediction.normalized_output,
        "target": prediction.target,
        "correct": prediction.correct,
        "parse_status": prediction.parse_status,
        "metadata": prediction.metadata,
        "provenance": identity,
    }


def _prediction_from_payload(payload: dict[str, Any]) -> Prediction:
    return Prediction(
        item_id=str(payload["item_id"]),
        condition=str(payload["condition"]),
        raw_output=str(payload["raw_output"]),
        normalized_output=str(payload["normalized_output"]),
        target=str(payload["target"]),
        correct=bool(payload["correct"]),
        parse_status=str(payload.get("parse_status", "OK")),
        metadata=dict(payload.get("metadata", {})),
    )


def _prediction_key(payload: dict[str, Any] | Prediction) -> tuple[str, str]:
    if isinstance(payload, Prediction):
        return payload.item_id, payload.condition
    return str(payload["item_id"]), str(payload["condition"])


def _canonical_prediction_records(predictions: list[Prediction]) -> list[dict[str, Any]]:
    records = [_prediction_payload(prediction, {}) for prediction in predictions]
    return sorted(records, key=lambda row: (row["item_id"], row["condition"]))


def _canonical_prediction_hash(predictions: list[Prediction]) -> str:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in _canonical_prediction_records(predictions)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class RunInterrupted(RuntimeError):
    """Raised by a deliberate test interruption after preserving the run dir."""

    def __init__(self, run_dir: Path) -> None:
        super().__init__(f"Run interrupted after partial progress: {run_dir}")
        self.run_dir = run_dir


class RunSession:
    """Single-process append-only run session with conservative resume checks."""

    def __init__(self, run_dir: Path, config: RunConfig, identity: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.config = config
        self.identity = identity
        self.config_hash = resolved_config_hash(config)
        self.predictions_path = run_dir / "predictions.jsonl"
        self.manifest_path = run_dir / "manifest.json"

    @staticmethod
    def _new_run_dir(config: RunConfig) -> Path:
        root = Path(config.output.root)
        if not root.is_absolute():
            root = Path.cwd() / root
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{timestamp}_{config.experiment.name}_{resolved_config_hash(config)}"
        run_dir = root / base_name
        collision_index = 1
        while run_dir.exists():
            run_dir = root / f"{base_name}_{collision_index:02d}"
            collision_index += 1
        return run_dir

    @staticmethod
    def _manifest_base(
        config: RunConfig, identity: dict[str, Any], timestamp: str
    ) -> dict[str, Any]:
        return {
            "artifact_schema_version": 2,
            "package_version": __version__,
            "timestamp_utc": timestamp,
            "status": "CREATED",
            "experiment_seed": config.experiment.seed,
            "backend_type": config.backend.type,
            "model_identifier": config.backend.model_path or config.backend.model_id,
            "benchmark": config.benchmark.type,
            "benchmark_path": config.benchmark.path,
            "config_hash": resolved_config_hash(config),
            "identity": identity,
            "model_provenance": identity.get("backend"),
            "vector_provenance": identity.get("vector"),
            "generation": {
                "do_sample": config.backend.do_sample,
                "temperature": config.backend.temperature,
                "max_new_tokens": config.backend.max_new_tokens,
            },
            **git_metadata(_repo_root()),
            **runtime_metadata(),
        }

    @classmethod
    def create(cls, config: RunConfig, identity: dict[str, Any]) -> RunSession:
        run_dir = cls._new_run_dir(config)
        run_dir.mkdir(parents=True, exist_ok=False)
        session = cls(run_dir, config, identity)
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        _atomic_write_text(
            run_dir / "config_resolved.yaml",
            yaml.safe_dump(config.as_dict(), sort_keys=False),
        )
        manifest = cls._manifest_base(config, identity, timestamp)
        _atomic_write_text(
            session.manifest_path,
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        )
        session.predictions_path.touch()
        session.set_status("RUNNING")
        return session

    @classmethod
    def resume(cls, run_dir: str | Path, config: RunConfig, identity: dict[str, Any]) -> RunSession:
        path = Path(run_dir)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETE":
            raise ValueError(
                "Run is already COMPLETE; use validate-run or choose an interrupted run"
            )
        if manifest.get("config_hash") != resolved_config_hash(config):
            raise ValueError("Resume refused: resolved config hash does not match the existing run")
        if manifest.get("identity") != identity:
            raise ValueError("Resume refused: model/vector/intervention provenance does not match")
        session = cls(path, config, identity)
        session.set_status("RUNNING")
        return session

    def set_status(self, status: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        _atomic_write_text(
            self.manifest_path,
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        )

    def _recover_prediction_file(self) -> None:
        if not self.predictions_path.exists():
            self.predictions_path.touch()
            return
        raw = self.predictions_path.read_bytes()
        if not raw:
            return
        lines = raw.splitlines(keepends=True)
        if lines[-1].endswith((b"\n", b"\r")):
            return
        try:
            json.loads(lines[-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            quarantine = self.predictions_path.with_suffix(".quarantine.jsonl")
            suffix = 1
            while quarantine.exists():
                quarantine = self.predictions_path.with_suffix(f".quarantine.{suffix}.jsonl")
                suffix += 1
            quarantine.write_bytes(lines[-1])
            _atomic_write_text(self.predictions_path, b"".join(lines[:-1]).decode("utf-8"))

    def existing_predictions(self) -> dict[tuple[str, str], Prediction]:
        self._recover_prediction_file()
        rows: dict[tuple[str, str], Prediction] = {}
        lines = self.predictions_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid complete prediction row at line {line_number}") from exc
            key = _prediction_key(payload)
            if key in rows:
                raise ValueError(f"Duplicate prediction key: {key}")
            if payload.get("provenance") != self.identity:
                raise ValueError(f"Prediction provenance mismatch at line {line_number}")
            rows[key] = _prediction_from_payload(payload)
        return rows

    def append_prediction(self, prediction: Prediction) -> None:
        existing = self.existing_predictions()
        key = _prediction_key(prediction)
        if key in existing:
            raise ValueError(f"Refusing to overwrite completed prediction key: {key}")
        row = _prediction_payload(prediction, self.identity)
        with self.predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(self, result: ExperimentResult) -> Path:
        predictions = result.predictions
        metrics = _json_safe(result.metrics)
        metrics_text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.run_dir / "metrics.json", metrics_text)
        _atomic_write_text(self.run_dir / "summary.md", render_summary(self.config, result))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "COMPLETE",
                "prediction_count": len(predictions),
                "prediction_sha256": _canonical_prediction_hash(predictions),
                "metrics_sha256": hashlib.sha256(metrics_text.encode("utf-8")).hexdigest(),
            }
        )
        _atomic_write_text(
            self.manifest_path,
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        )
        return self.run_dir


def render_summary(config: RunConfig, result: ExperimentResult) -> str:
    """Render the deliberately non-celebratory summary used in every run."""

    metrics = result.metrics
    parse_failures = sum(
        1 for prediction in result.predictions if prediction.parse_status != "OK"
    )
    if config.backend.type == "mock":
        caution = "IMPORTANT: MOCK RESULTS ARE SOFTWARE VALIDATION ONLY."
    elif config.backend.type == "tiny_transformer":
        caution = "IMPORTANT: TINY RANDOM TRANSFORMER RESULTS ARE SOFTWARE VALIDATION ONLY."
    else:
        caution = "IMPORTANT: THIS IS DEVELOPMENT INFRASTRUCTURE, NOT A CONFIRMATORY RESULT."
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
        f"PARSE FAILURES (NOT MODEL ERRORS BY THEMSELVES): {parse_failures}",
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
        caution,
        "These outputs are not scientific evidence and do not establish the research question.",
        "",
    ]
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_run_artifacts(config: RunConfig, result: ExperimentResult) -> Path:
    """Write the standard run directory through the resumable session writer."""

    identity = {
        "config_hash": resolved_config_hash(config),
        "experiment_seed": config.experiment.seed,
        "backend_type": config.backend.type,
        "model_identifier": config.backend.model_path or config.backend.model_id,
        "vector_hash": result.provenance.get("vector_hash"),
        "vector": result.provenance.get("vector"),
        "intervention": result.provenance.get("intervention"),
        "backend": result.provenance.get("backend", {"backend_type": config.backend.type}),
        "prompt_mode": config.backend.prompt_mode,
        "decoding": {
            "do_sample": config.backend.do_sample,
            "temperature": config.backend.temperature,
            "max_new_tokens": config.backend.max_new_tokens,
        },
        "benchmark_type": config.benchmark.type,
    }
    session = RunSession.create(config, identity)
    for prediction in result.predictions:
        session.append_prediction(prediction)
    return session.finalize(result)


def validate_run_directory(run_dir: str | Path) -> dict[str, Any]:
    """Validate a completed run against its resolved config and canonical metrics."""

    path = Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(path / "config_resolved.yaml")
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"Run status is {manifest.get('status')!r}, not COMPLETE")
    if manifest.get("config_hash") != resolved_config_hash(config):
        raise ValueError("Manifest config hash does not match config_resolved.yaml")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Manifest lacks identity provenance")
    session = RunSession(path, config, identity)
    rows = session.existing_predictions()
    predictions = sorted(
        rows.values(), key=lambda prediction: (prediction.item_id, prediction.condition)
    )
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    recomputed = compute_paired_metrics(predictions)
    if "bootstrap" in metrics:
        recomputed["bootstrap"] = bootstrap_paired_metrics(
            predictions, config.experiment.seed
        )
    if _json_safe(metrics) != _json_safe(recomputed):
        raise ValueError("Stored metrics do not match deterministic recomputation")
    if int(manifest.get("prediction_count", -1)) != len(predictions):
        raise ValueError("Manifest prediction_count does not match prediction rows")
    if manifest.get("prediction_sha256") != _canonical_prediction_hash(predictions):
        raise ValueError("Manifest prediction hash does not match canonical rows")
    metrics_text = (path / "metrics.json").read_text(encoding="utf-8")
    if manifest.get("metrics_sha256") != hashlib.sha256(metrics_text.encode("utf-8")).hexdigest():
        raise ValueError("Manifest metrics hash does not match metrics.json")
    return {
        "valid": True,
        "status": manifest["status"],
        "prediction_count": len(predictions),
        "config_hash": manifest["config_hash"],
        "prediction_sha256": manifest["prediction_sha256"],
        "metrics_sha256": manifest["metrics_sha256"],
    }
