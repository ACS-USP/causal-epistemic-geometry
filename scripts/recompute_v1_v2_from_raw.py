#!/usr/bin/env python3
"""CLI wrapper for the independent Q1 V1.2 aggregation audit."""

# The CLI prints long, human-readable audit labels.
# ruff: noqa: E501

from __future__ import annotations

import argparse
from pathlib import Path

from epistemic_geometry.analysis.v1_v2_audit import run_audit, validate_audit_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("review/q1_v1_2_aggregation_audit/raw_permutation_scores.jsonl"),
    )
    parser.add_argument(
        "--stored-sym",
        type=Path,
        default=Path("review/q1_v1_2_aggregation_audit/symmetrized_scores.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("review/q1_v1_2_principal_review/manifest.json"),
    )
    parser.add_argument(
        "--source-review-dir",
        type=Path,
        default=Path("review/q1_v1_2_principal_review"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/mmlu_pro_q1_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("review/q1_v1_2_principal_review_complete"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", type=Path, help="Validate an existing audit bundle and exit.")
    args = parser.parse_args()

    if args.validate:
        result = validate_audit_dir(args.validate)
        print(result)
        return 0
    result = run_audit(
        raw_path=args.raw,
        stored_sym_path=args.stored_sym,
        source_manifest_path=args.source_manifest,
        output_dir=args.output,
        repo_root=args.repo_root,
        source_review_dir=args.source_review_dir,
        split_manifest_path=args.split_manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        force=args.force,
    )
    print(f"Audit complete: {result['output_dir']}")
    print(
        f"Primary prediction mismatches: {result['comparison']['primary_prediction_mismatch_count']}"
    )
    print(
        f"Secondary prediction mismatches: {result['comparison']['secondary_prediction_mismatch_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
