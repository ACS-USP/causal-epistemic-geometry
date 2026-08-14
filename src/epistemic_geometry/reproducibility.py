"""Deterministic seeds and provenance helpers.

Scientific seeds must not use Python's process-randomized ``hash()``.  All
item-level randomness in this repository flows through SHA-256-derived seeds.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def stable_digest(*parts: object) -> str:
    """Return a stable SHA-256 digest for structured string-like parts."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: object, modulus: int = 2**32 - 1) -> int:
    """Map stable parts to a deterministic NumPy/Python-compatible seed."""

    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(stable_digest(*parts)[:16], 16) % modulus


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and optional Torch without requiring Torch."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_metadata(repo_root: Path) -> dict[str, Any]:
    """Return commit and dirty-state metadata without failing outside Git."""

    def run_git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status), "git_status": status or ""}


def package_versions(names: list[str]) -> dict[str, str | None]:
    """Collect installed package versions while tolerating optional packages."""

    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def runtime_metadata() -> dict[str, Any]:
    """Collect non-secret Python/device metadata for a manifest."""

    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ["causal-epistemic-geometry", "numpy", "pandas", "PyYAML", "torch", "transformers"]
        ),
    }
    try:
        import torch

        metadata["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "bf16_supported": bool(torch.cuda.is_bf16_supported())
            if torch.cuda.is_available()
            else False,
        }
    except (ImportError, RuntimeError):
        metadata["torch"] = {"installed": False}
    return metadata


def canonical_json(data: Any) -> str:
    """Serialize data for stable config/artifact hashing."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)

