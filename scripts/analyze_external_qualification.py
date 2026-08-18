#!/usr/bin/env python3
"""Aggregate completed Q1/Q2 external qualification runs without model access."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.base import ExternalResult  # noqa: E402
from epistemic_geometry.benchmarks.external.metrics import summarize_qualification  # noqa: E402


def _load_run(path: Path) -> tuple[dict[str, Any], list[ExternalResult]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    results_path = path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"completed results not found: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return manifest, [ExternalResult.from_record(row) for row in payload["rows"]]


def _decision(stage: str, summary: Any) -> tuple[bool, str]:
    if stage == "q1_smoke":
        checks = {
            "valid_completion_ge_90": summary.valid_completion >= 0.90,
            "at_least_two_correct": summary.correct_count >= 2,
            "at_least_two_genuine_wrong": summary.wrong_count >= 2,
            "wrong_not_dominated_by_failures": summary.wrong_count
            >= summary.invalid_count + summary.truncated_count + summary.runtime_error_count,
        }
    else:
        checks = {
            "valid_completion_ge_99": summary.valid_completion >= 0.99,
            "conditional_accuracy_30_to_80": 0.30 <= summary.conditional_accuracy <= 0.80,
            "genuine_wrong_dominates_failures": summary.wrong_count
            > summary.invalid_count + summary.truncated_count + summary.runtime_error_count,
            "seed_gap_le_10pp": summary.seed_accuracy_gap is not None
            and summary.seed_accuracy_gap <= 0.10,
        }
    return all(checks.values()), "; ".join(
        f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )


def _write_figures(output: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (output / "figures" / "FIGURES_SKIPPED.md").write_text(
            "Matplotlib unavailable; no decorative plots were generated.\n", encoding="utf-8"
        )
        return
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    labels = [str(row["candidate"]) for row in rows]
    valid = [float(row["valid_completion"]) for row in rows]
    accuracy = [float(row["conditional_accuracy"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(valid, accuracy)
    for x_value, y_value, label in zip(valid, accuracy, labels, strict=True):
        axis.annotate(label, (x_value, y_value))
    axis.set_xlabel("Valid completion")
    axis.set_ylabel("Conditional semantic accuracy")
    axis.set_title("External qualification: completion versus genuine accuracy")
    axis.set_xlim(0, 1.05)
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(figures / "completion_vs_accuracy.png", dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="candidate=/path/to/run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for argument in args.run:
        candidate, _, raw_path = argument.partition("=")
        if not candidate or not raw_path:
            raise ValueError("--run must have the form candidate=/path/to/run")
        manifest, results = _load_run(Path(raw_path))
        summary = summarize_qualification(results)
        passed, reasons = _decision(str(manifest["identity"]["stage"]), summary)
        rows.append(
            {
                "candidate": candidate,
                "stage": manifest["identity"]["stage"],
                "n_results": summary.n_results,
                "n_items": summary.n_items,
                "valid_completion": summary.valid_completion,
                "conditional_accuracy": summary.conditional_accuracy,
                "raw_accuracy": summary.raw_accuracy,
                "correct": summary.correct_count,
                "wrong": summary.wrong_count,
                "invalid_format": summary.invalid_count,
                "truncated_thinking": summary.truncated_count,
                "runtime_error": summary.runtime_error_count,
                "valid_ci95": summary.valid_interval,
                "conditional_accuracy_ci95": summary.conditional_accuracy_interval,
                "raw_accuracy_ci95": summary.raw_accuracy_interval,
                "seed_accuracy": summary.seed_accuracy,
                "seed_gap": summary.seed_accuracy_gap,
                "stable_hard": summary.stable_hard_count,
                "seed_sensitive": summary.seed_sensitive_count,
                "pair_oracle_accuracy": summary.pair_oracle_accuracy,
                "resampling_gain": summary.resampling_gain,
                "gate": "PASS" if passed else "FAIL",
                "gate_reasons": reasons,
            }
        )
        all_results.extend(row.to_record() for row in results)
    with (output / "QUALIFICATION_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "ERROR_MATRIX_DIAGNOSTICS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["candidate", "item_id", "rollout_seed", "status", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in all_results:
            writer.writerow({field: result.get(field) for field in fields})
    _write_figures(output, rows)
    (output / "COST_REPORT.md").write_text(
        "# Cost report\n\n"
        "This report is generated from bounded qualification artifacts only. GPU\n"
        "cost must be filled from the remote Pod billing/timing manifest; no\n"
        "qualification outcome is inferred from cost.\n\n"
        + "\n".join(
            f"- {row['candidate']} {row['stage']}: {row['n_results']} trajectories"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    qualified = [
        row["candidate"]
        for row in rows
        if row["stage"] == "q2_qualification" and row["gate"] == "PASS"
    ]
    sentinel = (
        "BENCHMARK_QUALIFIED_FOR_STEERING_PILOT"
        if qualified
        else "NO_EXTERNAL_BENCHMARK_QUALIFIED"
    )
    (output / "FINAL_REPORT.md").write_text(
        "# External benchmark qualification — final\n\n"
        + "\n".join(
            f"- **{row['candidate']} / {row['stage']}**: {row['gate']} — {row['gate_reasons']}"
            for row in rows
        )
        + f"\n\nFinal campaign decision: **{sentinel}**\n\n"
        "No steering pilot was run.\n",
        encoding="utf-8",
    )
    print(sentinel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
