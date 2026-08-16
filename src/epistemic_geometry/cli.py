"""Command-line entry points for local and future RunPod use."""

from __future__ import annotations

import importlib.util
import json
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
from epistemic_geometry.experiments.q1_v1 import (
    audit_q1_v1_repeat,
    build_split_manifest,
    run_q1_v1,
    validate_q1_v1_run,
)
from epistemic_geometry.experiments.q1_v1_1 import (
    audit_q1_v1_1_repeat,
    estimate_v1_v1,
    run_q1_v1_1,
    validate_q1_v1_1_run,
)
from epistemic_geometry.io.artifacts import validate_run_directory
from epistemic_geometry.reproducibility import git_metadata, runtime_metadata
from epistemic_geometry.steering import load_vector, save_vector
from epistemic_geometry.storage import storage_report

app = typer.Typer(help="Causal Geometry of Epistemic Complementarity research CLI")


def _dependency_status(name: str) -> str:
    return "yes" if importlib.util.find_spec(name) else "no"


def _model_cache_status(model_ref: str) -> str:
    """Inspect the local HF cache only; never resolve or download a model."""

    if Path(model_ref).exists():
        return "CACHED"
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_ref and repo.revisions:
                return "CACHED"
    except Exception:
        # Cache inspection is informational; a missing cache directory must
        # become NOT CACHED rather than aborting a no-download preflight.
        pass
    return "NOT CACHED"


def _path_is_inside(path_value: str | None, parent: str) -> bool:
    if not path_value:
        return False
    try:
        Path(path_value).expanduser().resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


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
        typer.echo(
            "MPS available: "
            f"{hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
        )
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
    hf_home = os.environ.get("HF_HOME")
    if hf_home and Path("/workspace").exists() and not _path_is_inside(hf_home, "/workspace"):
        typer.echo("WARNING: HF_HOME is outside /workspace; model cache may be ephemeral.")
    elif Path("/workspace").exists() and not hf_home:
        typer.echo("WARNING: HF_HOME is unset on a /workspace machine; cache location is implicit.")

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
        if loaded.backend.type in {"huggingface", "tiny_transformer"}:
            typer.echo(
                "HF backend dependencies ready: "
                f"{torch_status == 'yes' and transformers_status == 'yes'}"
            )
        if loaded.benchmark.type == "jsonl":
            benchmark_path = Path(loaded.benchmark.path or "")
            if not benchmark_path.is_absolute():
                benchmark_path = Path.cwd() / benchmark_path
            typer.echo(f"Benchmark path exists: {benchmark_path.exists()}")
        if loaded.benchmark.type == "mmlu_pro":
            typer.echo(
                "MMLU-Pro dataset: remote-only; local doctor performs no dataset load/download"
            )
        if loaded.steering.vector_path:
            vector_path = Path(loaded.steering.vector_path)
            if not vector_path.is_absolute():
                vector_path = Path.cwd() / vector_path
            typer.echo(f"Steering vector exists: {vector_path.exists()}")
        if loaded.backend.type in {"huggingface", "tiny_transformer"}:
            typer.echo(
                "HF config note: model/layer plausibility is checked only when the optional "
                "backend is explicitly constructed; doctor does not download models."
            )

    run_root.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Writable run directory: {run_root} ({os.access(run_root, os.W_OK)})")


@app.command("run")
def run(
    config: Path = typer.Argument(..., help="YAML experiment configuration."),
    resume: Path | None = typer.Option(
        None, "--resume", help="Resume an interrupted run directory."
    ),
) -> None:
    """Run baseline versus one steering vector, or an explicit dev alpha sweep."""

    try:
        loaded = load_config(config)
        paths = execute_experiment(loaded, resume_dir=resume)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if isinstance(paths, list):
        typer.echo("Development alpha sweep complete:")
        for path in paths:
            typer.echo(f"  {path}")
    else:
        typer.echo(f"Run complete: {paths}")


@app.command()
def preflight(
    config: Path = typer.Argument(..., help="YAML configuration to inspect without inference."),
) -> None:
    """Report what a run would do without loading weights or generating tokens."""

    try:
        loaded = load_config(config)
        benchmark_count: int | str
        if loaded.benchmark.type == "mock":
            benchmark_count = loaded.benchmark.n_items
        elif loaded.benchmark.type == "mmlu_pro":
            benchmark_count = "configured split (not loaded by preflight)"
            if loaded.benchmark.split_manifest:
                manifest_path = Path(loaded.benchmark.split_manifest)
                if not manifest_path.is_absolute():
                    manifest_path = Path.cwd() / manifest_path
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    split_ids = manifest.get("splits", {}).get(loaded.benchmark.split or "")
                    if isinstance(split_ids, list):
                        benchmark_count = len(split_ids)
        else:
            benchmark_path = Path(loaded.benchmark.path or "")
            if not benchmark_path.is_absolute():
                benchmark_path = Path.cwd() / benchmark_path
            if not benchmark_path.exists():
                benchmark_count = "UNKNOWN (benchmark path missing)"
            else:
                benchmark = build_benchmark(loaded)
                benchmark_count = len(benchmark)
        blockers: list[str] = []
        if loaded.benchmark.type == "jsonl":
            benchmark_path = Path(loaded.benchmark.path or "")
            if not benchmark_path.is_absolute():
                benchmark_path = Path.cwd() / benchmark_path
            if not benchmark_path.exists():
                blockers.append(f"benchmark path missing: {benchmark_path}")
        if loaded.benchmark.type == "mmlu_pro":
            typer.echo(
                "Dataset contents: NOT LOADED (preflight is offline; real MMLU-Pro "
                "operations are RunPod-only)"
            )
            if _dependency_status("datasets") != "yes":
                blockers.append(
                    "dataset dependency unavailable locally; install the HF extra only on RunPod"
                )
            else:
                typer.echo(
                    "Dataset cache: NOT INSPECTED LOCALLY (remote dataset unresolved locally)"
                )
            if loaded.benchmark.split in {"dev_calibration", "dev_evaluation"}:
                manifest_path = Path(loaded.benchmark.split_manifest or "")
                if not manifest_path.is_absolute():
                    manifest_path = Path.cwd() / manifest_path
                if not manifest_path.exists():
                    blockers.append(f"split manifest missing: {manifest_path}")
        if loaded.backend.type in {"huggingface", "tiny_transformer"}:
            if _dependency_status("torch") != "yes" or _dependency_status("transformers") != "yes":
                blockers.append("Torch/Transformers dependencies are missing")
        if loaded.backend.type == "huggingface":
            model_ref = loaded.backend.model_path or loaded.backend.model_id
            if not model_ref or model_ref == "REPLACE_ME":
                blockers.append("model id/path is REPLACE_ME or missing")
            elif loaded.backend.model_path and not Path(model_ref).exists():
                blockers.append(f"local model path does not exist: {model_ref}")
            else:
                cache_status = _model_cache_status(str(model_ref))
                typer.echo(f"Model cache: {cache_status}")
                if cache_status != "CACHED":
                    blockers.append("model is NOT CACHED/NOT VERIFIED; no download was attempted")
            if loaded.backend.layer_path == "REPLACE_ME":
                blockers.append("backend.layer_path is REPLACE_ME")
            for field_name, field_value in (
                ("backend.model_revision", loaded.backend.model_revision),
                ("backend.tokenizer_id", loaded.backend.tokenizer_id),
                ("backend.tokenizer_revision", loaded.backend.tokenizer_revision),
            ):
                if field_value == "REPLACE_ME":
                    blockers.append(f"{field_name} is REPLACE_ME")
        if loaded.steering.vector_path:
            vector_path = Path(loaded.steering.vector_path)
            if not vector_path.is_absolute():
                vector_path = Path.cwd() / vector_path
            if not vector_path.exists():
                blockers.append(f"steering vector missing: {vector_path}")
            else:
                typer.echo(f"Vector path: {vector_path}")
        elif loaded.steering.constructor == "REPLACE_ME":
            blockers.append("steering constructor is REPLACE_ME")
        output_root = Path(loaded.output.root)
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root
        typer.echo(f"Backend: {loaded.backend.type}")
        model_label = loaded.backend.model_path or loaded.backend.model_id or "local fixture"
        typer.echo(f"Model: {model_label}")
        typer.echo(f"Benchmark: {loaded.benchmark.type} ({benchmark_count} items)")
        if loaded.benchmark.type == "mmlu_pro":
            typer.echo(f"Dataset revision: {loaded.benchmark.dataset_revision or 'UNKNOWN'}")
            typer.echo(f"Dataset split: {loaded.benchmark.split or 'UNKNOWN'}")
        typer.echo(f"Prompt mode: {loaded.backend.prompt_mode}")
        typer.echo(f"Layer: {loaded.steering.layer}")
        typer.echo(f"Alpha: {loaded.steering.alpha_values()}")
        typer.echo(f"Token scope: {loaded.steering.token_scope}")
        typer.echo(
            f"Decoding: do_sample={loaded.backend.do_sample}, "
            f"max_new_tokens={loaded.backend.max_new_tokens}"
        )
        if isinstance(benchmark_count, int):
            calls = benchmark_count * 2
            extraction_calls = (
                benchmark_count if loaded.steering.constructor == "difference_of_means" else 0
            )
            typer.echo(
                f"Estimated generation calls: {calls}; "
                f"activation extractions: {extraction_calls}"
            )
        typer.echo(f"Expected artifact root: {output_root}")
        typer.echo("Resume compatibility: config/vector/model identity is checked by hash")
        if blockers:
            typer.echo("PREFLIGHT: NOT READY")
            for blocker in blockers:
                typer.echo(f"  BLOCKER: {blocker}")
            raise typer.Exit(code=1)
        typer.echo("PREFLIGHT: READY (no inference or downloads performed)")
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"Preflight failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-run")
def validate_run(run_dir: Path = typer.Argument(..., help="Completed run directory.")) -> None:
    """Recompute metrics and hashes for a completed run."""

    try:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_type") == "q1_v1_fixed_15_condition_pilot":
            report = validate_q1_v1_run(run_dir)
        elif manifest.get("experiment_type") == "q1_v1_1_controlled_followup":
            split_manifest = Path("data/splits/mmlu_pro_q1_v1.json")
            if not split_manifest.exists():
                split_manifest = Path(manifest["split_manifest"])
            report = validate_q1_v1_1_run(run_dir, split_manifest)
        else:
            report = validate_run_directory(run_dir)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Run invalid: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("build-splits")
def build_splits(
    config: Path = typer.Argument(..., help="Pinned Q1 YAML config."),
    output: Path = typer.Argument(..., help="JSON split manifest output path."),
) -> None:
    """Create the fixed 512/512 development split from official MMLU-Pro test."""

    try:
        loaded = load_config(config)
        if loaded.benchmark.type != "mmlu_pro":
            raise ConfigError("build-splits requires benchmark.type: mmlu_pro")
        manifest = build_split_manifest(loaded, output)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Split construction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Split manifest: {output}")
    typer.echo(f"Manifest SHA-256: {manifest['manifest_sha256']}")
    typer.echo(json.dumps(manifest["sizes"], sort_keys=True))


@app.command("q1-v1")
def q1_v1(
    config: Path = typer.Argument(..., help="Pinned Q1 V1 base YAML config."),
    split_manifest: Path = typer.Argument(..., help="Fixed 512/512/holdout split manifest."),
) -> None:
    """Run the fixed Q1 V1 development pilot; the confirmatory holdout is forbidden."""

    try:
        loaded = load_config(config)
        if loaded.experiment.stage != "development":
            raise ConfigError("q1-v1 is development-only")
        if loaded.benchmark.type != "mmlu_pro":
            raise ConfigError("q1-v1 requires benchmark.type: mmlu_pro")
        path = run_q1_v1(loaded, split_manifest)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Q1 V1 failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Q1 V1 complete: {path}")


@app.command("audit-q1-v1-repeat")
def audit_q1_v1_repeat_command(
    config: Path = typer.Argument(..., help="Pinned Q1 V1 base YAML config."),
    split_manifest: Path = typer.Argument(..., help="Fixed Q1 split manifest."),
    run_dir: Path = typer.Argument(..., help="Completed Q1 V1 run directory."),
    repeat_items: int = typer.Option(32, "--items", min=1, max=512),
) -> None:
    """Repeat fixed Q1 conditions on a deterministic prefix and record tolerance."""

    try:
        loaded = load_config(config)
        result = audit_q1_v1_repeat(loaded, split_manifest, run_dir, repeat_items)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Q1 repeat audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("preflight-q1-v1-1")
def preflight_q1_v1_1(config: Path = typer.Argument(..., help="Frozen V1.1 YAML config.")) -> None:
    """Estimate V1.1 workload and cost without model/data inference."""

    try:
        loaded = load_config(config)
        estimate = estimate_v1_v1(loaded)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"V1.1 preflight failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(estimate, indent=2, sort_keys=True))
    if not estimate["cost_gate_pass"]:
        typer.echo("V1.1 PREFLIGHT: STOP — projected cost exceeds $2.00", err=True)
        raise typer.Exit(code=1)
    typer.echo("V1.1 PREFLIGHT: COST GATE PASS (no inference or downloads performed)")


@app.command("q1-v1-1")
def q1_v1_1(
    config: Path = typer.Argument(..., help="Frozen Q1 V1.1 YAML config."),
    split_manifest: Path = typer.Argument(..., help="Frozen V1 split manifest."),
) -> None:
    """Run the controlled V1.1 follow-up; the holdout is forbidden."""

    try:
        loaded = load_config(config)
        path = run_q1_v1_1(loaded, split_manifest)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Q1 V1.1 failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Q1 V1.1 complete: {path}")


@app.command("audit-q1-v1-1-repeat")
def audit_q1_v1_1_repeat_command(
    config: Path = typer.Argument(..., help="Frozen Q1 V1.1 YAML config."),
    split_manifest: Path = typer.Argument(..., help="Frozen V1 split manifest."),
    run_dir: Path = typer.Argument(..., help="Completed Q1 V1.1 run directory."),
    repeat_items: int = typer.Option(16, "--items", min=1, max=512),
) -> None:
    """Repeat the fixed V1.1 reproducibility subset."""

    try:
        loaded = load_config(config)
        result = audit_q1_v1_1_repeat(loaded, split_manifest, run_dir, repeat_items)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Q1 V1.1 repeat audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def environment() -> None:
    """Print a non-secret runtime snapshot."""

    typer.echo(json.dumps(runtime_metadata(), indent=2, sort_keys=True))


@app.command("storage-check")
def storage_check(
    workspace: Path = typer.Option(Path("/workspace"), "--workspace"),
    threshold_gib: float = typer.Option(10.0, "--threshold-gib", min=0.0),
) -> None:
    """Report disk usage and warn about low persistent workspace capacity."""

    report = storage_report(workspace, threshold_gib)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if report["warning"]:
        typer.echo(f"STORAGE CHECK: WARNING: {report['warning']}", err=True)
    else:
        typer.echo("STORAGE CHECK: OK")


@app.command("estimate-memory")
def estimate_memory(
    parameters: int = typer.Option(..., "--parameters", help="Parameter count, e.g. 8000000000."),
    dtype: str = typer.Option("bf16", "--dtype", help="float32, fp16, bf16, or int8."),
) -> None:
    """Give a rough weight-memory estimate; KV cache/runtime overhead is extra."""

    bytes_per_parameter = {"float32": 4, "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
    if parameters <= 0 or dtype not in bytes_per_parameter:
        typer.echo(
            "parameters must be positive and dtype must be float32, fp16, bf16, or int8",
            err=True,
        )
        raise typer.Exit(code=1)
    weight_bytes = parameters * bytes_per_parameter[dtype]
    gib = weight_bytes / 1024**3
    typer.echo(f"Approximate weights: {gib:.2f} GiB ({dtype})")
    typer.echo(
        "Warning: generation KV cache, activations, allocator overhead, and device "
        "mapping are additional."
    )


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
            git_dirty=git_metadata(Path.cwd()).get("git_dirty"),
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
