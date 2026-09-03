#!/usr/bin/env python3
"""Write deterministic CPU-only power and precision planning artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q1_second_task_power import (  # noqa: E402
    HISTORICAL_C_INTERVAL,
    HISTORICAL_DELTA_C,
    HISTORICAL_DELTA_INTERVAL,
    HISTORICAL_MEANINGFUL_C,
    HISTORICAL_N,
    HISTORICAL_NULL_C,
    HISTORICAL_R,
    PLANNING_ICC,
    planning_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "review/q1_second_task_spark2_design",
    )
    parser.add_argument("--replicates", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = planning_grid(replicates=args.replicates)
    csv_path = args.output_dir / "POWER_PRECISION_GRID.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "classification": "CPU_ONLY_PROSPECTIVE_PLANNING",
        "replicates_per_cell": args.replicates,
        "seed": 2026082901,
        "historical_inputs": {
            "source": "frozen Qwen Q1 confirmatory artifacts",
            "n": HISTORICAL_N,
            "rollouts": HISTORICAL_R,
            "meaningful_C": HISTORICAL_MEANINGFUL_C,
            "null_C_values": HISTORICAL_NULL_C.tolist(),
            "delta_C_nullmean": HISTORICAL_DELTA_C,
            "C_interval": list(HISTORICAL_C_INTERVAL),
            "delta_C_interval": list(HISTORICAL_DELTA_INTERVAL),
            "planning_icc": PLANNING_ICC,
        },
        "assumptions": {
            "normal_calibration": "SE inferred from frozen percentile interval width",
            "item_scaling": "sqrt(57/N)",
            "rollout_scaling": "ICC variance decomposition; no claim of exact LiveCodeBench power",
            "test_correlation": 0.65,
            "null_controls": "empirical Qwen null C values sampled with replacement",
            "transfer_fractions": [0.0, 0.5, 0.75, 1.0],
        },
        "rows_file": csv_path.name,
    }
    (args.output_dir / "POWER_PRECISION_METHOD.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
