"""Reclassify preserved Gate 1 outputs under the corrected deterministic semantics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.reanalysis import reanalyze_gate1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-source-commit", required=True)
    parser.add_argument("--reanalysis-source-commit", required=True)
    args = parser.parse_args()
    result = reanalyze_gate1(
        args.journal,
        args.output,
        historical_source_commit=args.historical_source_commit,
        reanalysis_source_commit=args.reanalysis_source_commit,
    )
    print(result)


if __name__ == "__main__":
    main()
