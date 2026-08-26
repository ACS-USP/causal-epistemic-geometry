#!/usr/bin/env python3
"""Tiny dstack container probe. It intentionally performs no model inference."""

from __future__ import annotations

import json
import platform
import socket
import sys
from pathlib import Path


def main() -> int:
    shared = Path("/shared")
    result = {
        "hostname": socket.gethostname(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "shared_visible": shared.is_dir(),
        "shared_expected_directories": {
            name: (shared / name).is_dir() for name in ("modelos", "datasets", "checkpoints")
        },
    }
    try:
        import torch
    except (ImportError, OSError) as exc:
        result.update(
            status="FAIL",
            torch_version=None,
            cuda_available=False,
            gpu_name=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    result.update(
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=bool(torch.cuda.is_available()),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    passed = (
        result["architecture"] in {"aarch64", "arm64"}
        and result["cuda_available"]
        and result["shared_visible"]
        and all(result["shared_expected_directories"].values())
    )
    result["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
