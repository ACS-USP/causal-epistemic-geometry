"""No-inference remote environment validation with injectable local probes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    required: bool
    observed: Any
    expected: Any


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.passed or not check.required for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "checks": [asdict(check) for check in self.checks]}


class SystemProbe:
    """Read runtime facts without loading a model, allocating a dataset, or using a network."""

    def python_version(self) -> str:
        return platform.python_version()

    def package_version(self, distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    def torch_facts(self) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError:
            return {
                "installed": False,
                "cuda_available": False,
                "cuda_runtime": None,
                "gpu_models": [],
            }
        available = bool(torch.cuda.is_available())
        return {
            "installed": True,
            "cuda_available": available,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "gpu_models": (
                [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                if available
                else []
            ),
        }

    def disk_free_gb(self, path: Path) -> float:
        return shutil.disk_usage(path).free / (1024**3)

    def git_facts(self, root: Path) -> Mapping[str, Any]:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}


def _version_matches(version: str | None, constraint: str) -> bool:
    if version is None:
        return False
    return constraint == "*" or Version(version) in SpecifierSet(constraint)


def load_environment_spec(path: str | Path, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("remote environment spec must have schema_version: 1")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"unknown remote environment profile: {profile}")
    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"remote environment profile {profile!r} must be a mapping")
    return payload, selected


def run_preflight(
    spec: Mapping[str, Any],
    *,
    root: Path,
    probe: SystemProbe | None = None,
    hf_cache_path: Path | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    expected_source_commit: str | None = None,
) -> PreflightReport:
    """Validate an environment contract using only local metadata and filesystem state."""

    probe = probe or SystemProbe()
    checks: list[PreflightCheck] = []
    python_constraint = str(spec["python"])
    python_observed = probe.python_version()
    checks.append(
        PreflightCheck(
            "python_version",
            _version_matches(python_observed, python_constraint),
            True,
            python_observed,
            python_constraint,
        )
    )
    for import_name, package in dict(spec.get("packages", {})).items():
        distribution = str(package.get("distribution", import_name.replace("_", "-")))
        constraint = str(package.get("version", "*"))
        observed = probe.package_version(distribution)
        required = bool(package.get("required", True))
        checks.append(
            PreflightCheck(
                f"package:{import_name}",
                _version_matches(observed, constraint),
                required,
                observed,
                constraint,
            )
        )

    torch = dict(probe.torch_facts())
    require_cuda = bool(spec.get("require_cuda", True))
    checks.append(
        PreflightCheck(
            "torch_cuda_available",
            bool(torch.get("cuda_available")),
            require_cuda,
            bool(torch.get("cuda_available")),
            True,
        )
    )
    cuda_constraint = str(spec.get("cuda_runtime", "*"))
    cuda_runtime = torch.get("cuda_runtime")
    checks.append(
        PreflightCheck(
            "cuda_runtime",
            _version_matches(str(cuda_runtime), cuda_constraint) if cuda_runtime else False,
            require_cuda,
            cuda_runtime,
            cuda_constraint,
        )
    )
    gpu_models = list(torch.get("gpu_models") or [])
    checks.append(
        PreflightCheck("gpu_model", bool(gpu_models), require_cuda, gpu_models, "at least one GPU")
    )

    minimum_disk = float(spec.get("minimum_disk_free_gb", 0))
    disk_free = probe.disk_free_gb(root)
    checks.append(
        PreflightCheck(
            "disk_free_gb",
            disk_free >= minimum_disk,
            True,
            round(disk_free, 3),
            minimum_disk,
        )
    )

    cache = hf_cache_path or (Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else None)
    cache_required = bool(spec.get("hf_cache", {}).get("required", True))
    cache_exists = bool(cache and cache.exists() and cache.is_dir())
    checks.append(
        PreflightCheck(
            "hf_cache_path",
            cache_exists,
            cache_required,
            str(cache) if cache else None,
            "existing directory",
        )
    )

    models = dict(spec.get("models", {}))
    if model_id is not None or model_revision is not None:
        expected_revision = dict(models.get(model_id or "", {})).get("revision")
        checks.append(
            PreflightCheck(
                "model_revision",
                bool(model_id and model_revision and expected_revision == model_revision),
                True,
                {"model_id": model_id, "revision": model_revision},
                {"model_id": model_id, "revision": expected_revision},
            )
        )

    git = dict(probe.git_facts(root))
    checks.append(
        PreflightCheck(
            "source_git_commit",
            expected_source_commit is None or git.get("commit") == expected_source_commit,
            True,
            git.get("commit"),
            expected_source_commit or "record current commit",
        )
    )
    checks.append(
        PreflightCheck(
            "source_worktree_clean",
            not bool(git.get("dirty")),
            True,
            git.get("dirty"),
            False,
        )
    )
    return PreflightReport(tuple(checks))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote-preflight",
        description=(
            "Validate a remote environment without inference, downloads, or dataset access."
        ),
    )
    parser.add_argument("--spec", type=Path, default=Path("remote_environment.yaml"))
    parser.add_argument("--profile", default="CORE_QWEN")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _payload, profile = load_environment_spec(args.spec, args.profile)
    report = run_preflight(
        profile,
        root=args.root.resolve(),
        hf_cache_path=args.hf_cache,
        model_id=args.model_id,
        model_revision=args.model_revision,
        expected_source_commit=args.expected_source_commit,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for check in report.checks:
            status = "PASS" if check.passed else ("WARN" if not check.required else "FAIL")
            print(
                f"{status:4} {check.name}: "
                f"observed={check.observed!r}; expected={check.expected!r}"
            )
        print(f"REMOTE PREFLIGHT: {'READY' if report.ready else 'NOT READY'}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
