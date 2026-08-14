"""The first development experiment: paired baseline versus one vector."""

from __future__ import annotations

import json
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any

from epistemic_geometry.backends import ModelBackend, build_backend
from epistemic_geometry.backends.base import validate_vector_dimension
from epistemic_geometry.benchmarks import Benchmark, JsonlBenchmark, MockBenchmark
from epistemic_geometry.config import RunConfig
from epistemic_geometry.io.artifacts import RunInterrupted, RunSession, resolved_config_hash
from epistemic_geometry.metrics import bootstrap_paired_metrics, compute_paired_metrics
from epistemic_geometry.reproducibility import seed_everything
from epistemic_geometry.steering import (
    difference_of_means,
    load_vector,
    random_unit_vector,
)
from epistemic_geometry.types import ExperimentResult, Intervention, Prediction, SteeringVector


def build_benchmark(config: RunConfig) -> Benchmark:
    """Construct a benchmark adapter without coupling it to a model backend."""

    benchmark_config = config.benchmark
    if benchmark_config.type == "mock":
        return MockBenchmark(
            n_items=benchmark_config.n_items,
            seed=config.experiment.seed,
            allowed_targets=benchmark_config.allowed_targets,
        )
    path = Path(benchmark_config.path or "")
    if not path.is_absolute():
        path = Path.cwd() / path
    return JsonlBenchmark(
        path=path,
        allowed_targets=benchmark_config.allowed_targets,
        max_items=benchmark_config.max_items,
    )


def build_vector(config: RunConfig, backend: ModelBackend, benchmark: Benchmark) -> SteeringVector:
    """Load or construct exactly one vector for the current development run."""

    steering = config.steering
    if steering.vector_path:
        path = Path(steering.vector_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        vector = load_vector(path)
    elif steering.constructor == "random_unit":
        vector = random_unit_vector(
            dimension=steering.vector_dimension or backend.hidden_size,
            seed=(
                steering.vector_seed
                if steering.vector_seed is not None
                else config.experiment.seed
            ),
            layer=steering.layer,
            metadata={"source": "config_generated_random_control"},
        )
    elif steering.constructor == "difference_of_means":
        items = benchmark.items()
        midpoint = len(items) // 2
        if midpoint == 0:
            raise ValueError("Difference-of-means construction needs at least two benchmark items")
        vector = difference_of_means(
            backend,
            items[:midpoint],
            items[midpoint:],
            layer=steering.layer,
            seed=config.experiment.seed,
        )
    elif steering.constructor.startswith("mock_fixture:"):
        fixture_kind = steering.constructor.split(":", 1)[1]
        if not hasattr(backend, "fixture_vector"):
            raise ValueError("mock_fixture constructors are available only with the mock backend")
        vector = backend.fixture_vector(fixture_kind, steering.vector_seed)  # type: ignore[attr-defined]
        vector = SteeringVector(
            values=vector.values,
            layer=steering.layer,
            constructor=vector.constructor,
            normalization=vector.normalization,
            metadata=vector.metadata,
            hash=vector.hash,
        )
    else:
        raise ValueError(f"Unsupported steering constructor: {steering.constructor}")
    validate_vector_dimension(vector, backend)
    return vector


def _prediction(
    item: Any,
    condition: str,
    output: Any,
    parser: Any,
) -> Prediction:
    parsed = parser.parse(output.raw_output)
    return Prediction(
        item_id=item.id,
        condition=condition,
        raw_output=output.raw_output,
        normalized_output=parsed.normalized,
        target=item.target,
        correct=parsed.status == "OK" and parsed.normalized == item.target.upper(),
        parse_status=parsed.status,
        metadata={**output.metadata, "parse_status": parsed.status},
    )


def evaluate_paired(
    backend: ModelBackend,
    benchmark: Benchmark,
    intervention: Intervention,
    bootstrap_seed: int | None = None,
) -> ExperimentResult:
    """Run paired in-order inference while preserving baseline state."""

    predictions: list[Prediction] = []
    for item in benchmark:
        baseline_output = backend.predict(item)
        predictions.append(_prediction(item, "baseline", baseline_output, benchmark.parser))
        with backend.steer(intervention):
            steered_output = backend.predict(item)
        predictions.append(_prediction(item, "steered", steered_output, benchmark.parser))
    metrics = compute_paired_metrics(predictions)
    if bootstrap_seed is not None:
        metrics["bootstrap"] = bootstrap_paired_metrics(predictions, bootstrap_seed)
    return ExperimentResult(
        predictions=predictions,
        metrics=metrics,
        provenance={
            "n_items": len(benchmark),
            "vector_hash": intervention.vector.hash,
            "intervention": {
                "layer": intervention.layer,
                "alpha": intervention.alpha,
                "vector_id": intervention.vector_id,
                "token_scope": intervention.token_scope,
            },
        },
    )


def _scalar_run(
    config: RunConfig,
    resume_dir: Path | None = None,
    stop_after_items: int | None = None,
) -> Path:
    seed_everything(config.experiment.seed)
    benchmark = build_benchmark(config)
    backend = build_backend(config)
    vector = build_vector(config, backend, benchmark)
    alpha = config.steering.alpha_values()[0]
    if not config.steering.enabled:
        alpha = 0.0
    intervention = Intervention(
        layer=config.steering.layer,
        alpha=alpha,
        vector_id=vector.hash,
        token_scope=config.steering.token_scope,
        vector=vector,
    )
    backend_provenance = backend.provenance() if hasattr(backend, "provenance") else {
        "backend_type": config.backend.type
    }
    identity = {
        "config_hash": resolved_config_hash(config),
        "experiment_seed": config.experiment.seed,
        "backend_type": config.backend.type,
        "model_identifier": config.backend.model_path or config.backend.model_id,
        "vector_hash": vector.hash,
        "vector": {
            "hash": vector.hash,
            "layer": vector.layer,
            "constructor": vector.constructor,
            "normalization": vector.normalization,
            "metadata": vector.metadata,
        },
        "intervention": {
            "layer": intervention.layer,
            "alpha": intervention.alpha,
            "vector_id": intervention.vector_id,
            "token_scope": intervention.token_scope,
        },
        "backend": backend_provenance,
        "prompt_mode": config.backend.prompt_mode,
        "decoding": {
            "do_sample": config.backend.do_sample,
            "temperature": config.backend.temperature,
            "max_new_tokens": config.backend.max_new_tokens,
        },
        "benchmark_type": config.benchmark.type,
    }
    session = (
        RunSession.resume(resume_dir, config, identity)
        if resume_dir
        else RunSession.create(config, identity)
    )
    existing = session.existing_predictions()
    try:
        for item_index, item in enumerate(benchmark, start=1):
            baseline_key = (item.id, "baseline")
            if baseline_key not in existing:
                baseline_output = backend.predict(item)
                prediction = _prediction(item, "baseline", baseline_output, benchmark.parser)
                session.append_prediction(prediction)
                existing[baseline_key] = prediction
            steered_key = (item.id, "steered")
            if steered_key not in existing:
                with backend.steer(intervention):
                    steered_output = backend.predict(item)
                prediction = _prediction(item, "steered", steered_output, benchmark.parser)
                session.append_prediction(prediction)
                existing[steered_key] = prediction
            if stop_after_items is not None and item_index >= stop_after_items:
                session.set_status("INTERRUPTED")
                raise RunInterrupted(session.run_dir)
    except RunInterrupted:
        raise
    except Exception:
        session.set_status("FAILED")
        raise

    predictions = [
        existing[(item.id, condition)]
        for item in benchmark
        for condition in ("baseline", "steered")
    ]
    metrics = compute_paired_metrics(predictions)
    metrics["bootstrap"] = bootstrap_paired_metrics(predictions, config.experiment.seed)
    result = ExperimentResult(
        predictions=predictions,
        metrics=metrics,
        provenance={
            "n_items": len(benchmark),
            "vector_hash": vector.hash,
            "vector": identity["vector"],
            "intervention": identity["intervention"],
            "backend": backend_provenance,
            "prompt_mode": config.backend.prompt_mode,
            "decoding": identity["decoding"],
            "benchmark_type": config.benchmark.type,
        },
    )
    return session.finalize(result)


def run_experiment(
    config: RunConfig,
    resume_dir: str | Path | None = None,
    stop_after_items: int | None = None,
) -> Path | list[Path]:
    """Run one scalar experiment or an explicitly configured development sweep."""

    alpha_values = config.steering.alpha_values()
    if resume_dir is not None and len(alpha_values) > 1:
        raise ValueError("Resume is supported for scalar runs, not alpha sweeps")
    if len(alpha_values) > 1:
        if config.experiment.stage != "development":
            raise ValueError("Alpha sweeps are development-only; set experiment.stage: development")
        paths = [
            _scalar_run(replace(config, steering=replace(config.steering, alpha=alpha)))
            for alpha in alpha_values
        ]
        _write_alpha_sweep(config, alpha_values, paths)
        return paths
    return _scalar_run(
        config,
        resume_dir=Path(resume_dir) if resume_dir is not None else None,
        stop_after_items=stop_after_items,
    )


def _write_alpha_sweep(config: RunConfig, alpha_values: list[float], paths: list[Path]) -> Path:
    """Write development-only alpha plots separately from individual run artifacts."""

    sweep_dir = paths[0].parent / "alpha-sweep"
    figures_dir = sweep_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha, path in zip(alpha_values, paths, strict=True):
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "alpha": alpha,
                "accuracy": metrics["treatment_accuracy"],
                "error_similarity_vs_baseline": metrics["error_correlation_phi"],
                "run_dir": str(path),
            }
        )
    (sweep_dir / "sweep_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not config.output.save_figures:
        return sweep_dir

    alphas = [row["alpha"] for row in rows]
    accuracy_values = [row["accuracy"] for row in rows]
    similarity_values = [row["error_similarity_vs_baseline"] for row in rows]
    for filename, ylabel, values in (
        ("accuracy_vs_alpha.svg", "Treatment accuracy", accuracy_values),
        ("error_similarity_vs_alpha.svg", "Phi error similarity vs baseline", similarity_values),
    ):
        _write_svg_line_plot(figures_dir / filename, alphas, values, ylabel)
    return sweep_dir


def _write_svg_line_plot(
    path: Path, x_values: list[float], y_values: list[float], ylabel: str
) -> None:
    """Write a tiny dependency-free figure for portable development artifacts."""

    width, height = 720, 460
    left, right, top, bottom = 80, 30, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    y_min, y_max = min(0.0, min(y_values)), max(1.0, max(y_values))
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - y_value) / (y_max - y_min) * plot_height
        return x, y

    points = [point(x_value, y_value) for x_value, y_value in zip(x_values, y_values, strict=True)]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" />' for x, y in points)
    x_label = f"alpha ({x_min:g} to {x_max:g})"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        '  <rect width="100%" height="100%" fill="white" />\n'
        f'  <text x="{width / 2:.0f}" y="24" text-anchor="middle" '
        'font-family="sans-serif" font-size="18">Development-only alpha sweep</text>\n'
        f'  <line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" />\n'
        f'  <line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}" stroke="black" />\n'
        f'  <polyline points="{polyline}" fill="none" stroke="#1769aa" '
        f'stroke-width="3" />\n'
        f'  {circles}\n'
        f'  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14">{escape(x_label)}</text>\n'
        f'  <text x="18" y="{height / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {height / 2:.0f})" font-family="sans-serif" '
        f'font-size="14">{escape(ylabel)}</text>\n'
        '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")
