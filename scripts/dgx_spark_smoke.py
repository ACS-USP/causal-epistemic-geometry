#!/usr/bin/env python3
"""Bounded CPU/CUDA/BF16/unified-memory smoke; no models or downloads."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * (1024 if len(fields) > 1 else 1)
    return values


def torch_equal(value: Any) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def smoke(allocation_mib: int) -> tuple[dict[str, Any], int]:
    report: dict[str, Any] = {
        "schema_version": "ceg.dgx-spark-smoke.v1",
        "architecture": platform.machine(),
        "allocation_mib": allocation_mib,
        "memory_before": meminfo(),
    }
    try:
        import torch
    except (ImportError, OSError) as exc:
        report.update(
            status="FAIL", reason="torch_not_importable", error=f"{type(exc).__name__}: {exc}"
        )
        return report, 1

    report.update(
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()),
    )

    cpu_left = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    cpu_result = cpu_left @ torch.eye(8, dtype=torch.float32)
    report["cpu_fp32_identity_exact"] = torch_equal(torch.equal(cpu_result, cpu_left))

    torch.manual_seed(4319)
    cpu_seed_a = torch.rand((16, 16), dtype=torch.float32)
    torch.manual_seed(4319)
    cpu_seed_b = torch.rand((16, 16), dtype=torch.float32)
    report["cpu_seed_repeat_exact"] = torch_equal(torch.equal(cpu_seed_a, cpu_seed_b))

    if not torch.cuda.is_available():
        report.update(status="FAIL", reason="cuda_not_available")
        return report, 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report.update(
        device_name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        reported_total_device_memory_bytes=int(properties.total_memory),
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
    )

    left = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8)
    fp32_result = left @ torch.eye(8, device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    report["gpu_fp32_identity_exact"] = torch_equal(torch.equal(fp32_result, left))

    torch.manual_seed(9127)
    cuda_seed_a = torch.rand((16, 16), device=device, dtype=torch.float32)
    cuda_seed_product_a = cuda_seed_a @ cuda_seed_a.T
    torch.manual_seed(9127)
    cuda_seed_b = torch.rand((16, 16), device=device, dtype=torch.float32)
    cuda_seed_product_b = cuda_seed_b @ cuda_seed_b.T
    torch.cuda.synchronize(device)
    report["gpu_seed_repeat_exact"] = torch_equal(
        torch.equal(cuda_seed_product_a, cuda_seed_product_b)
    )

    if report["bf16_supported"]:
        bf16_left = torch.arange(64, device=device, dtype=torch.bfloat16).reshape(8, 8)
        bf16_result = bf16_left @ torch.eye(8, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        report["gpu_bf16_identity_exact"] = torch_equal(torch.equal(bf16_result, bf16_left))
        report["bf16_vs_fp32_max_abs_difference"] = float(
            (bf16_result.float() - fp32_result).abs().max().item()
        )
    else:
        report["gpu_bf16_identity_exact"] = False

    # 512 MiB by default is <0.5% of the stated 121 GiB unified-memory capacity.
    allocation_elements = allocation_mib * 1024 * 1024 // 4
    memory_probe = torch.empty(allocation_elements, device=device, dtype=torch.float32)
    memory_probe.fill_(1.0)
    torch.cuda.synchronize(device)
    report["memory_during"] = meminfo()
    report["cuda_memory_during"] = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    del memory_probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    report["memory_after"] = meminfo()

    checks = (
        "cpu_fp32_identity_exact",
        "cpu_seed_repeat_exact",
        "gpu_fp32_identity_exact",
        "gpu_seed_repeat_exact",
        "gpu_bf16_identity_exact",
    )
    passed = all(report.get(check) is True for check in checks)
    report.update(
        status="PASS" if passed else "FAIL",
        limitation=(
            "Seed repeatability covers only these simple operations; it does not establish "
            "deterministic Transformer generation. Memory observations are descriptive and "
            "may include unrelated host activity."
        ),
    )
    return report, 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-mib",
        type=int,
        default=512,
        help="bounded CUDA allocation used for the unified-memory observation (default: 512)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 64 <= args.allocation_mib <= 4096:
        print("--allocation-mib must be between 64 and 4096", file=sys.stderr)
        return 2
    report, exit_code = smoke(args.allocation_mib)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
