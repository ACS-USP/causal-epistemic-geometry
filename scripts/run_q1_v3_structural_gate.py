#!/usr/bin/env python3
"""Run the model-free Q1 V3 structural and answer-distribution gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epistemic_geometry.benchmarks.reasoning.validation import validate_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cell", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("review/q1_v3_reasoning_instrument/structural_gate_summary.json"),
    )
    args = parser.parse_args()
    if args.n_per_cell <= 0:
        raise ValueError("--n-per-cell must be positive")
    report = validate_suite(n_per_cell=args.n_per_cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
