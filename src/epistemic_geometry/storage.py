"""Non-destructive storage diagnostics for local machines and persistent Pods."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _usage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "exists": True,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / 1024**3,
    }


def _directory_size(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.split()[0] if result.stdout.split() else None


def storage_report(workspace: Path, threshold_gib: float) -> dict[str, Any]:
    """Return storage facts without deleting caches or artifacts."""

    cache = Path(os.environ.get("HF_HOME", str(workspace / "hf-cache")))
    runs = Path(
        os.environ.get("CEG_RUN_ROOT", str(workspace / "causal-epistemic-geometry" / "runs"))
    )
    report = {
        "root": _usage(Path("/")),
        "workspace": _usage(workspace),
        "hf_home": {"path": str(cache), "size": _directory_size(cache)},
        "runs": {"path": str(runs), "size": _directory_size(runs)},
        "threshold_gib": threshold_gib,
    }
    workspace_usage = report["workspace"]
    report["warning"] = (
        "workspace is missing"
        if not workspace_usage.get("exists")
        else (
            f"free space below {threshold_gib:.1f} GiB"
            if workspace_usage["free_gib"] < threshold_gib
            else None
        )
    )
    return report
