#!/usr/bin/env python3
"""Run baseline-only Q1 V3 Stage-A or Stage-B calibration on RunPod."""

from __future__ import annotations

import argparse
from pathlib import Path

from epistemic_geometry.benchmarks.reasoning.runner import run_baseline_calibration
from epistemic_geometry.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-key")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--inference-engine",
        choices=("serial_reasoning_reference", "max_budget_prefix_reuse", "batched_reasoning"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = run_baseline_calibration(
        config,
        args.manifest,
        args.output,
        manifest_key=args.manifest_key,
        max_items=args.max_items,
        inference_engine=args.inference_engine,
    )
    print(f"Q1 V3 baseline calibration artifacts: {output}")


if __name__ == "__main__":
    main()
