#!/usr/bin/env python3
"""Primary analysis for the frozen controller-held-out Q2 DEVELOPMENT pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_prediction import (  # noqa: E402
    classify_q2,
    edge_indices,
    edge_values,
    heldout_prediction,
    qap_permutation,
)
from epistemic_geometry.analysis.rank_statistics import spearman_correlation  # noqa: E402
from epistemic_geometry.experiments.q2_controller_heldout import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    CONTROLLER_IDS,
    N_ITEMS,
    QAP_PERMUTATIONS,
    QAP_SEED,
    error_arrays,
    pairwise_unbiased_distance_matrix,
)
from epistemic_geometry.research.reliability import CrashSafeJournal  # noqa: E402

REVIEW = ROOT / "review/q2_controller_heldout_geometry"
METRICS = ("M0_FLAT", "M1_WHITENED", "M2_FINITE_SECANT")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(review: Path, lock: dict[str, Any]) -> list[dict[str, Any]]:
    journal = CrashSafeJournal(
        review / "journal.jsonl",
        identity={
            "experiment_id": lock["experiment_id"],
            "phase": "COMMON_PANEL",
            "source_commit": lock["experiment_source_commit"],
            "protocol_lock_sha256": sha256(review / "PROTOCOL_LOCK.json"),
        },
        key_fields=("item_id", "condition", "rollout_index"),
    )
    rows = list(journal.rows.values())
    if len(rows) != 4080:
        raise RuntimeError(f"Q2 analysis requires 4,080 rows; found {len(rows)}")
    return rows


def condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    baseline_accuracy = float(
        np.mean([row["correct"] for row in rows if row["condition"] == "BASELINE"])
    )
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        tokens = np.asarray([row["generated_token_count"] for row in selected])
        output.append(
            {
                "condition": condition,
                "n": len(selected),
                "accuracy": float(np.mean([row["correct"] for row in selected])),
                "accuracy_change_from_baseline": float(
                    np.mean([row["correct"] for row in selected]) - baseline_accuracy
                ),
                "commitment_validity": float(
                    np.mean([row["commitment_valid"] for row in selected])
                ),
                "semantic_evaluability": float(
                    np.mean([row["semantic_evaluable"] for row in selected])
                ),
                "mean_tokens": float(np.mean(tokens)),
                "median_tokens": float(np.median(tokens)),
                "p90_tokens": float(np.quantile(tokens, 0.90)),
                "max_tokens": int(np.max(tokens)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(
    arrays: dict[str, np.ndarray],
    geometries: dict[str, np.ndarray],
    split: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_values = {
        name: {"rho": [], "srmse": [], "rmse_ratio": []} for name in METRICS
    }
    full_d = pairwise_unbiased_distance_matrix(arrays)
    full_edge = edge_values(
        full_d, edge_indices(CONTROLLER_IDS, split["train_controllers"])["heldout"]
    )
    reliability: list[float] = []
    edge_samples = np.empty(
        (BOOTSTRAP_RESAMPLES, len(CONTROLLER_IDS), len(CONTROLLER_IDS)), dtype=np.float32
    )
    for index in range(BOOTSTRAP_RESAMPLES):
        selected = rng.integers(0, N_ITEMS, size=N_ITEMS)
        sampled = {name: values[selected] for name, values in arrays.items()}
        distance = pairwise_unbiased_distance_matrix(sampled)
        edge_samples[index] = distance
        sampled_edge = edge_values(
            distance,
            edge_indices(CONTROLLER_IDS, split["train_controllers"])["heldout"],
        )
        correlation = spearman_correlation(sampled_edge, full_edge)
        if correlation is not None:
            reliability.append(correlation)
        for name in METRICS:
            score = heldout_prediction(
                geometries[name], distance, CONTROLLER_IDS, split["train_controllers"]
            )
            metric_values[name]["rho"].append(score["heldout_spearman_rho"])
            metric_values[name]["srmse"].append(score["heldout_standardized_rmse"])
            metric_values[name]["rmse_ratio"].append(score["rmse_ratio_to_constant"])
    intervals: dict[str, Any] = {}
    for name, values in metric_values.items():
        intervals[name] = {
            key: {
                "q025": float(np.nanquantile(records, 0.025)),
                "median": float(np.nanquantile(records, 0.5)),
                "q975": float(np.nanquantile(records, 0.975)),
            }
            for key, records in values.items()
        }
    edge_lower = np.quantile(edge_samples, 0.025, axis=0)
    edge_upper = np.quantile(edge_samples, 0.975, axis=0)
    all_edges = [
        (left, right)
        for left in range(len(CONTROLLER_IDS))
        for right in range(left + 1, len(CONTROLLER_IDS))
    ]
    widths = edge_values(edge_upper - edge_lower, all_edges)
    return intervals, {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "unit": "item; all controllers and both rollouts move together",
        "full_vs_bootstrap_D_rank_correlation": {
            "median": float(np.median(reliability)),
            "q025": float(np.quantile(reliability, 0.025)),
            "q975": float(np.quantile(reliability, 0.975)),
        },
        "edge_interval_width": {
            "mean": float(np.mean(widths)),
            "max": float(np.max(widths)),
        },
    }


def reliability_diagnostics(arrays: dict[str, np.ndarray], matrix: np.ndarray) -> dict[str, Any]:
    all_edges = [
        (left, right)
        for left in range(len(CONTROLLER_IDS))
        for right in range(left + 1, len(CONTROLLER_IDS))
    ]
    values = edge_values(matrix, all_edges)
    first_items = np.arange(N_ITEMS) % 2 == 0
    second_items = ~first_items
    first = pairwise_unbiased_distance_matrix(
        {name: array[first_items] for name, array in arrays.items()}
    )
    second = pairwise_unbiased_distance_matrix(
        {name: array[second_items] for name, array in arrays.items()}
    )
    return {
        "edge_count": len(values),
        "negative_edge_count": int(np.sum(values < 0)),
        "negative_edge_fraction": float(np.mean(values < 0)),
        "zero_edge_count": int(np.sum(values == 0)),
        "minimum_D": float(np.min(values)),
        "maximum_D": float(np.max(values)),
        "item_half_matrix_spearman": spearman_correlation(
            edge_values(first, all_edges), edge_values(second, all_edges)
        ),
        "two_rollout_low_resolution_warning": True,
        "plugin_squared_propensities_used": False,
    }


def plots(
    review: Path,
    geometries: dict[str, np.ndarray],
    distance: np.ndarray,
    split: dict,
) -> None:
    figure_dir = review / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name, matrix in {**geometries, "ERROR_D": distance}.items():
        fig, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="viridis")
        axis.set_title(name)
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name.lower()}_matrix.png", dpi=180)
        plt.close(fig)
    edges = edge_indices(CONTROLLER_IDS, split["train_controllers"])["heldout"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, name in zip(axes, METRICS, strict=True):
        axis.scatter(edge_values(geometries[name], edges), edge_values(distance, edges), s=18)
        axis.set_title(name)
        axis.set_xlabel("frozen geometry")
        axis.set_ylabel("unbiased D")
    fig.tight_layout()
    fig.savefig(figure_dir / "heldout_geometry_vs_error.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    lock = read_json(review / "PROTOCOL_LOCK.json")
    rows = load_rows(review, lock)
    panel = read_json(review / "DEVELOPMENT_PANEL_MANIFEST.json")
    item_ids = [row["item_id"] for row in panel]
    arrays = error_arrays(rows, item_ids)
    distance = pairwise_unbiased_distance_matrix(arrays)
    np.save(review / "ERROR_DISTANCE_MATRIX.npy", distance)
    with np.load(review / "GEOMETRY_MATRICES.npz", allow_pickle=False) as archive:
        geometries = {name: archive[name].astype(np.float64) for name in METRICS}
    split = read_json(review / "CONTROLLER_SPLIT_LOCK.json")
    metric_results: dict[str, Any] = {}
    for index, name in enumerate(METRICS):
        score = heldout_prediction(
            geometries[name], distance, CONTROLLER_IDS, split["train_controllers"]
        )
        qap = qap_permutation(
            geometries[name],
            distance,
            CONTROLLER_IDS,
            split["train_controllers"],
            permutations=QAP_PERMUTATIONS,
            seed=QAP_SEED + index,
        )
        metric_results[name] = {"score": score, "qap": qap}
    classification = classify_q2(metric_results)
    intervals, bootstrap_reliability = bootstrap(arrays, geometries, split)
    reliability = {
        **reliability_diagnostics(arrays, distance),
        **bootstrap_reliability,
    }
    summaries = condition_summary(rows)
    write_csv(review / "CONDITION_SUMMARY.csv", summaries)
    write_json(review / "PREDICTION_RESULTS.json", metric_results)
    write_json(review / "BOOTSTRAP_INTERVALS.json", intervals)
    write_json(review / "ERROR_GEOMETRY_RELIABILITY.json", reliability)
    write_json(
        review / "EVIDENCE_VECTOR.json",
        {
            "controller_bank": "QUALIFIED",
            "flat_geometry": classification["metric_signals"]["M0_FLAT"],
            "whitened_geometry": classification["metric_signals"]["M1_WHITENED"],
            "finite_secant": classification["metric_signals"]["M2_FINITE_SECANT"],
            "metric_ranking": sorted(
                METRICS,
                key=lambda name: (
                    -metric_results[name]["score"]["heldout_spearman_rho"],
                    metric_results[name]["score"]["heldout_rmse"],
                ),
            ),
            "prediction_failures": {
                name: [
                    check
                    for check, passed in classification["metric_signals"][name].items()
                    if check != "signal" and not passed
                ]
                for name in METRICS
            },
            "classification": classification["classification"],
            "development_only": True,
        },
    )
    write_json(review / "CLASSIFICATION.json", classification)
    plots(review, geometries, distance, split)
    best = max(METRICS, key=lambda name: metric_results[name]["score"]["heldout_spearman_rho"])
    if classification["classification"] in {
        "Q2_PILOT_SIMPLE_GEOMETRY_SIGNAL",
        "Q2_PILOT_HELDOUT_PREDICTION_SIGNAL",
        "Q2_PILOT_CONTROL_GEOMETRY_OUTPERFORMS_FLAT",
    }:
        next_title = "Geometry-guided prospective controller selection/replication draft"
    else:
        next_title = "Q2 measurement-reliability and local-geometry discriminator draft"
    (review / "NEXT_PROTOCOL_DRAFT.md").write_text(
        f"# {next_title}\n\nDRAFT ONLY. Principal review required. No inference authorized.\n",
        encoding="utf-8",
    )
    report = [
        "# Q2 controller-held-out epistemic geometry pilot",
        "",
        f"Classification: `{classification['classification']}` (DEVELOPMENT ONLY)",
        "",
        "The bank qualified before common-panel outcomes. The 57-item confirmatory pool was "
        "identity-excluded and no confirmatory outcome entered Q2.",
        "",
        "## Held-out prediction",
        "",
    ]
    for name in METRICS:
        score = metric_results[name]["score"]
        qap = metric_results[name]["qap"]
        report.extend(
            [
                f"- {name}: rho={score['heldout_spearman_rho']:.6f}, "
                f"QAP p={qap['p_value_one_sided']:.6f}, standardized "
                f"RMSE={score['heldout_standardized_rmse']:.6f}.",
            ]
        )
    report.extend(
        [
            "",
            f"Best predictive metric by held-out rho: `{best}`.",
            "",
            "M2 is a finite-displacement JS secant, not an exact JVP, Fisher, pullback, "
            "or manifold metric.",
            "",
            "Q1 remains `Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`. Q3 was not run.",
        ]
    )
    (review / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        review / "manifest_hashes.json",
        {
            path.name: sha256(path)
            for path in sorted(review.iterdir())
            if path.is_file() and path.name != "manifest_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                "classification": classification["classification"],
                "best_metric": best,
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
