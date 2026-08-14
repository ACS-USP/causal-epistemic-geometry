"""Command-line entry points for local and future RunPod use."""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path

import typer

from epistemic_geometry import __version__
from epistemic_geometry.analysis.summarize import read_summary
from epistemic_geometry.config import ConfigError, load_config
from epistemic_geometry.experiments.baseline_vs_steering import (
    build_benchmark,
    build_vector,
)
from epistemic_geometry.experiments.baseline_vs_steering import run_experiment as execute_experiment
from epistemic_geometry.reproducibility import git_metadata
from epistemic_geometry.steering import load_vector, save_vector

app = typer.Typer(help="Causal Geometry of Epistemic Complementarity research CLI")


def _dependency_status(name: str) -> str:
    return "yes" if importlib.util.find_spec(name) else "no"


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", help="Optional YAML config to validate.")
) -> None:
    """Report local runtime/GPU readiness without downloading anything."""

    typer.echo(f"Python: {platform.python_version()}")
    typer.echo(f"Package: {__version__}")
    torch_status = _dependency_status("torch")
    transformers_status = _dependency_status("transformers")
    typer.echo(f"torch installed: {torch_status}")
    typer.echo(f"transformers installed: {transformers_status}")
    if torch_status == "yes":
        import torch

        typer.echo(f"CUDA available: {torch.cuda.is_available()}")
        typer.echo(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            typer.echo(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        typer.echo(
            "bf16 support: "
            f"{torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}"
        )
    else:
        typer.echo("CUDA available: unknown (torch is not installed)")
        typer.echo("GPU names: unavailable")
        typer.echo("bf16 support: unknown")
    typer.echo(f"HF_HOME: {os.environ.get('HF_HOME', '(default)')}")
    typer.echo(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '(default)')}")

    if config is None:
        run_root = Path.cwd() / "runs"
    else:
        try:
            loaded = load_config(config)
        except (ConfigError, TypeError) as exc:
            typer.echo(f"Config: INVALID ({exc})")
            raise typer.Exit(code=1) from exc
        run_root = Path(loaded.output.root)
        if not run_root.is_absolute():
            run_root = Path.cwd() / run_root
        typer.echo(f"Config: valid ({loaded.backend.type} backend)")
        layer_plausible = loaded.backend.layer >= 0 and loaded.steering.layer >= 0
        typer.echo(f"Layer config plausible: {layer_plausible}")
        if loaded.backend.type == "huggingface":
            typer.echo(
                "HF backend dependencies ready: "
                f"{torch_status == 'yes' and transformers_status == 'yes'}"
            )
        if loaded.benchmark.type == "jsonl":
            benchmark_path = Path(loaded.benchmark.path or "")
            if not benchmark_path.is_absolute():
                benchmark_path = Path.cwd() / benchmark_path
            typer.echo(f"Benchmark path exists: {benchmark_path.exists()}")
        if loaded.steering.vector_path:
            vector_path = Path(loaded.steering.vector_path)
            if not vector_path.is_absolute():
                vector_path = Path.cwd() / vector_path
            typer.echo(f"Steering vector exists: {vector_path.exists()}")
        if loaded.backend.type == "huggingface":
            typer.echo(
                "HF config note: model/layer plausibility is checked only when the optional "
                "backend is explicitly constructed; doctor does not download models."
            )

    run_root.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Writable run directory: {run_root} ({os.access(run_root, os.W_OK)})")


@app.command("run")
def run(config: Path = typer.Argument(..., help="YAML experiment configuration.")) -> None:
    """Run baseline versus one steering vector, or an explicit dev alpha sweep."""

    try:
        loaded = load_config(config)
        paths = execute_experiment(loaded)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if isinstance(paths, list):
        typer.echo("Development alpha sweep complete:")
        for path in paths:
            typer.echo(f"  {path}")
    else:
        typer.echo(f"Run complete: {paths}")


@app.command("build-vector")
def build_vector_command(
    config: Path = typer.Argument(..., help="YAML config selecting backend and constructor."),
    output: Path = typer.Argument(..., help="Output .npz path; JSON metadata is adjacent."),
) -> None:
    """Build and save one reproducible steering vector."""

    try:
        loaded = load_config(config)
        benchmark = build_benchmark(loaded)
        from epistemic_geometry.backends import build_backend

        backend = build_backend(loaded)
        vector = build_vector(loaded, backend, benchmark)
        vector_path, metadata_path = save_vector(
            vector,
            output,
            git_commit=git_metadata(Path.cwd()).get("git_commit"),
        )
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Vector build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Vector: {vector_path}")
    typer.echo(f"Metadata: {metadata_path}")
    typer.echo(f"Hash: {vector.hash}")


@app.command("inspect-vector")
def inspect_vector(path: Path = typer.Argument(..., help="Vector .npz path.")) -> None:
    """Print safe vector metadata without loading opaque objects."""

    try:
        vector = load_vector(path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        typer.echo(f"Vector inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Hash: {vector.hash}")
    typer.echo(f"Dimension: {vector.dimension}")
    typer.echo(f"Layer: {vector.layer}")
    typer.echo(f"Constructor: {vector.constructor}")
    typer.echo(f"Normalization: {vector.normalization}")
    typer.echo(f"Metadata: {vector.metadata}")


@app.command()
def summarize(run_dir: Path = typer.Argument(..., help="Generated run directory.")) -> None:
    """Print a stored run summary."""

    try:
        typer.echo(read_summary(run_dir))
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
