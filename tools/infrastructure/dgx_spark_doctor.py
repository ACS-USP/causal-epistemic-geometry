#!/usr/bin/env python3
"""Collect a secret-free DGX Spark environment census and stable fingerprint.

This tool is diagnostic only. It does not install packages, contact the network,
load a model, allocate CUDA tensors, or mutate system configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ceg.dgx-spark-environment.v1"
FINGERPRINT_VERSION = "ceg.dgx-spark-fingerprint.v1"
RELEVANT_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "flash-attn",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
    "vllm",
)


def run_command(
    args: Sequence[str], *, cwd: Path | None = None, timeout: int = 15
) -> dict[str, Any]:
    """Run a bounded read-only command and return a JSON-safe result."""
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False, "returncode": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            [executable, *args[1:]],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": "command timed out",
        }
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_os_release(text: str) -> dict[str, str]:
    """Parse /etc/os-release without evaluating shell syntax."""
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        parsed[key] = value.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")
    return parsed


def parse_key_value_lines(text: str, delimiter: str = ":") -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if delimiter not in line:
            continue
        key, value = line.split(delimiter, 1)
        parsed[key.strip()] = value.strip()
    return parsed


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    """Return selected Linux memory values as bytes."""
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for key, raw_value in parse_key_value_lines(path.read_text(encoding="utf-8")).items():
        parts = raw_value.split()
        if not parts:
            continue
        multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
        try:
            values[key] = int(parts[0]) * multiplier
        except ValueError:
            continue
    return values


def installed_python_packages() -> list[dict[str, str]]:
    """Return a normalized package freeze without paths or credentials."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        packages[name.lower().replace("_", "-")] = distribution.version
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def package_versions(freeze: list[dict[str, str]]) -> dict[str, str | None]:
    versions = {entry["name"]: entry["version"] for entry in freeze}
    return {name: versions.get(name) for name in RELEVANT_PACKAGES}


def collect_torch() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"installed": False, "import_error": f"{type(exc).__name__}: {exc}"}

    cuda_available = bool(torch.cuda.is_available())
    data: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        data.update(
            device_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            reported_total_memory_bytes=int(properties.total_memory),
        )
    return data


def collect_nvidia() -> dict[str, Any]:
    query = run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    gpu_rows: list[dict[str, str]] = []
    if query["returncode"] == 0:
        for row in query["stdout"].splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 3:
                gpu_rows.append(
                    {
                        "name": fields[0],
                        "driver_version": fields[1],
                        "compute_capability": fields[2],
                    }
                )
    driver_path = Path("/proc/driver/nvidia/version")
    driver_text = driver_path.read_text(encoding="utf-8").strip() if driver_path.is_file() else None
    nvcc = run_command(("nvcc", "--version"))
    return {
        "gpus": gpu_rows,
        "nvidia_smi_available": query["available"],
        "nvidia_smi_error": query["stderr"] or None,
        "kernel_driver_version_text": driver_text,
        "nvcc_available": nvcc["available"],
        "nvcc_version_text": nvcc["stdout"] or nvcc["stderr"] or None,
    }


def collect_native_packages() -> list[dict[str, str]]:
    query = run_command(("dpkg-query", "-W", "-f=${Package}\t${Version}\n"), timeout=30)
    if query["returncode"] != 0:
        return []
    markers = ("cuda", "cudnn", "nccl", "nvidia")
    rows: list[dict[str, str]] = []
    for line in query["stdout"].splitlines():
        if "\t" not in line:
            continue
        name, version = line.split("\t", 1)
        if any(marker in name.lower() for marker in markers):
            rows.append({"name": name, "version": version})
    return sorted(rows, key=lambda row: row["name"])


def collect_git(repo: Path) -> dict[str, Any]:
    top = run_command(("git", "rev-parse", "--show-toplevel"), cwd=repo)
    if top["returncode"] != 0:
        return {"repository": False, "head": None, "clean": None}
    head = run_command(("git", "rev-parse", "HEAD"), cwd=repo)
    status = run_command(("git", "status", "--porcelain"), cwd=repo)
    return {
        "repository": True,
        "head": head["stdout"] or None,
        "clean": status["returncode"] == 0 and not status["stdout"],
    }


def collect_shared_storage(home: Path | None = None) -> dict[str, Any]:
    shared = (home or Path.home()) / "shared"
    expected = ["modelos", "datasets", "checkpoints"]
    findmnt = run_command(("findmnt", "-J", "-T", str(shared))) if shared.exists() else None
    service_active = run_command(("systemctl", "is-active", "spark-shared.service"))
    service_enabled = run_command(("systemctl", "is-enabled", "spark-shared.service"))
    result: dict[str, Any] = {
        "path": str(shared),
        "exists": shared.exists(),
        "expected_directories": {name: (shared / name).is_dir() for name in expected},
        "service_active": service_active["stdout"] or "unknown",
        "service_enabled": service_enabled["stdout"] or "unknown",
        "mount": None,
    }
    if findmnt and findmnt["returncode"] == 0:
        try:
            result["mount"] = json.loads(findmnt["stdout"])
        except json.JSONDecodeError:
            result["mount"] = {"raw": findmnt["stdout"]}
    if shared.exists():
        stats = os.statvfs(shared)
        result["filesystem_bytes"] = {
            "total": stats.f_blocks * stats.f_frsize,
            "available": stats.f_bavail * stats.f_frsize,
        }
    return result


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def build_fingerprint_metadata(report: dict[str, Any]) -> dict[str, Any]:
    """Select stable environment fields; exclude time and free-space fluctuations."""
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "architecture": report["identity"]["architecture"],
        "kernel": report["identity"]["kernel"],
        "os_release": report["os_release"],
        "cpu": report["cpu"],
        "memory_total_bytes": report["memory"].get("MemTotal"),
        "nvidia": report["nvidia"],
        "native_nvidia_packages": report["native_nvidia_packages"],
        "python": report["python"],
        "python_packages": report["python_packages"],
        "torch": report["torch"],
        "git_head": report["git"]["head"],
    }


def attach_fingerprint(report: dict[str, Any]) -> dict[str, Any]:
    metadata = build_fingerprint_metadata(report)
    report["fingerprint"] = {
        "algorithm": "sha256",
        "canonical_metadata": metadata,
        "sha256": hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest(),
    }
    return report


def collect_environment(repo: Path) -> dict[str, Any]:
    os_path = Path("/etc/os-release")
    lscpu = run_command(("lscpu", "-J"))
    cpu: Any
    if lscpu["returncode"] == 0:
        try:
            cpu = json.loads(lscpu["stdout"])
        except json.JSONDecodeError:
            cpu = parse_key_value_lines(run_command(("lscpu",))["stdout"])
    else:
        cpu = parse_key_value_lines(run_command(("lscpu",))["stdout"])

    freeze = installed_python_packages()
    ip_links = run_command(("ip", "-json", "link", "show"))["stdout"]
    infiniband_path = Path("/sys/class/infiniband")
    infiniband_devices = (
        sorted(path.name for path in infiniband_path.iterdir()) if infiniband_path.is_dir() else []
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "identity": {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "kernel": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
        },
        "os_release": parse_os_release(os_path.read_text(encoding="utf-8"))
        if os_path.is_file()
        else {},
        "cpu": cpu,
        "memory": read_meminfo(),
        "nvidia": collect_nvidia(),
        "native_nvidia_packages": collect_native_packages(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "python_packages": freeze,
        "relevant_package_versions": package_versions(freeze),
        "torch": collect_torch(),
        "git": collect_git(repo),
        "shared_storage": collect_shared_storage(),
        "network": {
            "interfaces": ip_links,
            "infiniband_devices": infiniband_devices,
            "expected_nccl_hca": "rocep1s0f1",
            "expected_nccl_hca_present": "rocep1s0f1" in infiniband_devices,
        },
    }
    return attach_fingerprint(report)


def render_summary(report: dict[str, Any]) -> str:
    identity = report["identity"]
    torch = report["torch"]
    gpu_names = [gpu["name"] for gpu in report["nvidia"]["gpus"]]
    lines = [
        "DGX Spark environment doctor",
        f"hostname: {identity['hostname']}",
        f"architecture: {identity['architecture']}",
        f"kernel: {identity['kernel']['release']}",
        f"os: {report['os_release'].get('PRETTY_NAME', 'unknown')}",
        f"memory total bytes: {report['memory'].get('MemTotal', 'unknown')}",
        f"GPU(s): {', '.join(gpu_names) if gpu_names else 'not detected'}",
        f"PyTorch: {torch.get('version', 'not importable')}",
        f"CUDA available: {torch.get('cuda_available', False)}",
        f"BF16 supported: {torch.get('bf16_supported', False)}",
        f"shared storage: {report['shared_storage']['exists']}",
        f"git HEAD: {report['git']['head'] or 'not a repository'}",
        f"environment SHA-256: {report['fingerprint']['sha256']}",
    ]
    return "\n".join(lines)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path.cwd(), help="repository used for git HEAD"
    )
    parser.add_argument("--json-out", type=Path, help="write the full report atomically")
    parser.add_argument(
        "--fingerprint-out", type=Path, help="write only canonical metadata and its SHA-256"
    )
    parser.add_argument("--summary-only", action="store_true", help="omit JSON from stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_environment(args.repo.resolve())
    if args.json_out:
        write_json_atomic(args.json_out, report)
    if args.fingerprint_out:
        write_json_atomic(args.fingerprint_out, report["fingerprint"])
    print(render_summary(report), file=sys.stderr if not args.summary_only else sys.stdout)
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
