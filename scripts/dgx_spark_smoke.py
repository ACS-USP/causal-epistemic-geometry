#!/usr/bin/env python3
"""Tiny opt-in DGX Spark CUDA smoke; no models, downloads, or mutation."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print(json.dumps({"status": "FAIL", "reason": "torch_not_installed"}))
        return 1

    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if not torch.cuda.is_available():
        report.update(status="FAIL", reason="cuda_not_available")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report.update(
        device_name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        total_memory_bytes=properties.total_memory,
        bf16_supported=torch.cuda.is_bf16_supported(),
    )

    left = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8)
    right = torch.eye(8, device=device, dtype=torch.float32)
    result = left @ right
    torch.cuda.synchronize(device)
    exact = bool(torch.equal(result, left))
    report.update(status="PASS" if exact else "FAIL", tiny_matmul_identity=exact)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
