#!/usr/bin/env python3
"""Produce descriptive, frozen-rule analysis for completed Q1 V3 Stage A."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epistemic_geometry.benchmarks.reasoning.calibration import (
    select_stage_b_cells,
    stage_a_qualifies,
)
from epistemic_geometry.metrics.errors import error_jaccard, phi_correlation


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _budget(row: dict[str, Any]) -> int:
    return int(row["generation_config"]["max_new_tokens"])


def _group(row: dict[str, Any]) -> str:
    return f"{row['family']}/{row['cell']}"


def _pair_metric(rows: list[dict[str, Any]], field: str) -> float:
    by_key = {
        (str(row["latent_id"]), int(row["rollout_index"])): row for row in rows
    }
    keys = sorted(by_key)
    left = [bool(by_key[key][field]) for key in keys if field == "correct"]
    if field != "correct":
        left = [by_key[key][field] for key in keys]
    return float(np.mean(left)) if left else float("nan")


def _outcome(group: str, budget: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rollout[int(row["rollout_index"])].append(row)
    seed_accuracy = [
        float(np.mean([bool(row["correct"]) for row in by_rollout[index]]))
        for index in sorted(by_rollout)
    ]
    parse_success = float(np.mean([row["parse_status"] == "OK" for row in rows]))
    mean_accuracy = float(np.mean([bool(row["correct"]) for row in rows]))
    truncation_statuses = {"THINKING_UNCLOSED", "TRUNCATED_NO_FINAL"}
    predictions = [row.get("parsed_answer") for row in rows]
    return {
        "manifest_key": f"{group}/{budget}",
        "family": group.split("/", 1)[0],
        "cell": group.split("/", 1)[1],
        "reasoning_budget": budget,
        "n_rollouts": len(rows),
        "n_latents": len({row["latent_id"] for row in rows}),
        "mean_accuracy": mean_accuracy,
        "seed_accuracy": seed_accuracy,
        "seed_accuracy_gap": max(seed_accuracy) - min(seed_accuracy)
        if seed_accuracy
        else float("nan"),
        "seed_accuracy_sd": float(np.std(seed_accuracy)) if seed_accuracy else float("nan"),
        "parse_success": parse_success,
        "parse_status_counts": {
            status: sum(row["parse_status"] == status for row in rows)
            for status in sorted({row["parse_status"] for row in rows})
        },
        "thinking_unclosed_rate": float(
            np.mean([row["parse_status"] == "THINKING_UNCLOSED" for row in rows])
        ),
        "truncation_rate": float(
            np.mean([row["parse_status"] in truncation_statuses for row in rows])
        ),
        "mean_prefix_tokens": float(np.mean([len(row["token_ids"]) for row in rows])),
        "median_prefix_tokens": float(np.median([len(row["token_ids"]) for row in rows])),
        "prediction_distribution": {
            str(value): predictions.count(value) for value in sorted(set(predictions), key=str)
        },
    }


def _transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["latent_id"]), int(row["rollout_index"]))][_budget(row)] = row
    result = []
    for left, right in ((512, 1024), (1024, 2048)):
        pairs = [values for values in by_key.values() if left in values and right in values]
        result.append(
            {
                "from_budget": left,
                "to_budget": right,
                "incorrect_to_correct": sum(
                    not values[left]["correct"] and values[right]["correct"] for values in pairs
                ),
                "correct_to_incorrect": sum(
                    values[left]["correct"] and not values[right]["correct"] for values in pairs
                ),
                "both_correct": sum(
                    values[left]["correct"] and values[right]["correct"] for values in pairs
                ),
                "both_incorrect": sum(
                    not values[left]["correct"] and not values[right]["correct"] for values in pairs
                ),
                "n_pairs": len(pairs),
            }
        )
    return result


def _seed_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_budget: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_budget[_budget(row)][str(row["latent_id"])].append(row)
    diagnostics: dict[str, Any] = {}
    for budget, by_latent in sorted(by_budget.items()):
        pairs = [
            sorted(values, key=lambda row: int(row["rollout_index"]))
            for values in by_latent.values()
        ]
        pairs = [pair for pair in pairs if len(pair) == 2]
        prediction_agreement = [
            pair[0].get("parsed_answer") == pair[1].get("parsed_answer") for pair in pairs
        ]
        error_left = [not pair[0]["correct"] for pair in pairs]
        error_right = [not pair[1]["correct"] for pair in pairs]
        diagnostics[str(budget)] = {
            "n_pairs": len(pairs),
            "prediction_agreement": float(np.mean(prediction_agreement)) if pairs else float("nan"),
            "hard_error_agreement": float(
                np.mean(
                    [
                        left == right
                        for left, right in zip(error_left, error_right, strict=True)
                    ]
                )
            )
            if pairs
            else float("nan"),
            "error_phi": phi_correlation(error_left, error_right) if pairs else float("nan"),
            "error_jaccard": error_jaccard(error_left, error_right) if pairs else float("nan"),
            "accuracy_rollout_0": float(np.mean([pair[0]["correct"] for pair in pairs]))
            if pairs
            else float("nan"),
            "accuracy_rollout_1": float(np.mean([pair[1]["correct"] for pair in pairs]))
            if pairs
            else float("nan"),
        }
    return diagnostics


def _write_figures(out: Path, outcomes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted({str(item["family"]) + "/" + str(item["cell"]) for item in outcomes})
    budgets = [512, 1024, 2048]

    def grouped(metric: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(groups))
        width = 0.24
        for index, budget in enumerate(budgets):
            values = [
                next(
                    item[metric]
                    for item in outcomes
                    if item["manifest_key"] == f"{group}/{budget}"
                )
                for group in groups
            ]
            ax.bar(x + (index - 1) * width, values, width, label=str(budget))
        ax.set_xticks(x, groups, rotation=35, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(title="budget")
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=160)
        plt.close(fig)

    grouped("mean_accuracy")
    grouped("parse_success")
    grouped("truncation_rate")

    fig, ax = plt.subplots(figsize=(8, 4))
    transition_rows = _transitions(rows)
    labels = [f"{row['from_budget']}→{row['to_budget']}" for row in transition_rows]
    rescues = [row["incorrect_to_correct"] for row in transition_rows]
    damages = [row["correct_to_incorrect"] for row in transition_rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, rescues, 0.36, label="incorrect → correct")
    ax.bar(x + 0.18, damages, 0.36, label="correct → incorrect")
    ax.set_xticks(x, labels)
    ax.set_ylabel("paired transitions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "budget_transitions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for family in sorted({item["family"] for item in outcomes}):
        family_rows = [item for item in outcomes if item["family"] == family]
        ax.plot(
            [item["reasoning_budget"] for item in family_rows],
            [item["mean_accuracy"] for item in family_rows],
            marker="o",
            label=family,
        )
    ax.axhspan(0.20, 0.90, color="green", alpha=0.08, label="qualification accuracy region")
    ax.set_xlabel("reasoning budget")
    ax.set_ylabel("mean accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "qualification_map.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_lengths = []
    labels = []
    for group in groups:
        for budget in budgets:
            values = [
                len(row["token_ids"])
                for row in rows
                if _group(row) == group and _budget(row) == budget
            ]
            grouped_lengths.append(values)
            labels.append(f"{group}\n{budget}")
    ax.boxplot(grouped_lengths, labels=labels, showfliers=False)
    ax.set_ylabel("derived prefix token count")
    ax.tick_params(axis="x", labelrotation=75)
    fig.tight_layout()
    fig.savefig(out / "reasoning_lengths.png", dpi=160)
    plt.close(fig)


def analyze(run_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    review = Path(review_dir)
    rows = _rows(run / "rollouts.jsonl")
    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(_group(row), _budget(row))].append(row)
    outcomes = [
        _outcome(group, budget, grouped_rows[(group, budget)])
        for group, budget in sorted(grouped_rows)
    ]
    for outcome in outcomes:
        outcome["qualified"] = stage_a_qualifies(outcome)
    selected = select_stage_b_cells(outcomes)
    transitions = _transitions(rows)
    seed_diagnostics = _seed_diagnostics(rows)
    report = {
        "status": "DEVELOPMENT_STAGE_A_DESCRIPTIVE_ONLY",
        "steering_performed": False,
        "geometry_performed": False,
        "confirmatory_accessed": False,
        "outcomes": outcomes,
        "transitions": transitions,
        "seed_diagnostics": seed_diagnostics,
        "selected_stage_b_plan": selected,
        "surviving_family_count": len(selected),
        "instrument_screen": "PASS" if len(selected) >= 2 else "FAIL",
    }
    review.mkdir(parents=True, exist_ok=True)
    (review / "qualification_table.json").write_text(
        json.dumps(outcomes, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    (review / "budget_diagnostics.json").write_text(
        json.dumps(
            {"transitions": transitions, "seed_diagnostics": seed_diagnostics},
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (review / "stage_b_deterministic_plan.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    (review / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    _write_figures(review / "figures", outcomes, rows)
    summary = [
        "# Q1 V3 Stage-A descriptive analysis",
        "",
        "This is baseline-only development calibration. No steering, geometry, "
        "Stage B, or confirmatory holdout was run.",
        "",
        f"Instrument screen: **{report['instrument_screen']}** "
        f"({len(selected)} surviving families).",
        "",
        "| Family | Cell | Budget | Accuracy | Parse | Seed gap | Qualified |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for outcome in outcomes:
        summary.append(
            f"| {outcome['family']} | {outcome['cell']} | {outcome['reasoning_budget']} | "
            f"{outcome['mean_accuracy']:.4f} | {outcome['parse_success']:.4f} | "
            f"{outcome['seed_accuracy_gap']:.4f} | {outcome['qualified']} |"
        )
    summary.extend(
        [
            "",
            "The selected Stage-B plan is deterministic metadata only. Stage B was not executed.",
            "",
            "Scientific result: none frozen.",
        ]
    )
    (review / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
