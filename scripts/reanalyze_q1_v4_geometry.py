#!/usr/bin/env python3
"""Reanalyze preserved Q1 V4 geometry artifacts with average tied ranks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.rank_statistics import (  # noqa: E402
    label_permutation_test,
    pearson_correlation,
    spearman_correlation,
)
from epistemic_geometry.benchmarks.v4.geometry import conceptual_distance  # noqa: E402
from epistemic_geometry.reproducibility import stable_seed  # noqa: E402

DEFAULT_SOURCE = ROOT / "review" / "q1_v4_microbench"
DEFAULT_OUTPUT = ROOT / "review" / "q1_v4_microbench_reanalysis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairwise(values: list[np.ndarray], distance: Any) -> np.ndarray:
    return np.asarray(
        [
            distance(values[i], values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ],
        dtype=float,
    )


def _conceptual_pairs(domain: str, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        [
            conceptual_distance(domain, indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ],
        dtype=float,
    )


def _hellinger(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.sqrt(np.maximum(left, 0)) - np.sqrt(np.maximum(right, 0)))
        / math.sqrt(2)
    )


def reanalyze(source: Path) -> list[dict[str, Any]]:
    run = source / "geometry_qwen"
    rows_path = run / "rows.jsonl"
    arrays_path = run / "activations.npz"
    old_path = source / "GEOMETRY_RESULTS.csv"
    old_rows = {row["domain"]: row for row in csv.DictReader(old_path.open(encoding="utf-8"))}
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    arrays = np.load(arrays_path)
    results: list[dict[str, Any]] = []
    for domain in ("WEEKDAYS", "LETTERS"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        indices: dict[str, int] = {}
        for row in domain_rows:
            label = str(row["answer"])
            groups[label].append(row)
            indices[label] = int(row["conceptual_index"])
        labels = sorted(groups, key=indices.get)
        conceptual_indices = tuple(indices[label] for label in labels)
        centroids = [
            np.mean(
                np.stack([arrays[row["thinking_activation_key"]] for row in groups[label]]),
                axis=0,
            )
            for label in labels
        ]
        distributions = [
            np.mean(np.asarray([row["direct_probabilities"] for row in groups[label]]), axis=0)
            for label in labels
        ]
        activation = _pairwise(centroids, lambda x, y: np.linalg.norm(x - y))
        conceptual = _conceptual_pairs(domain, conceptual_indices)
        behavior = _pairwise(distributions, _hellinger)
        permutation = label_permutation_test(
            conceptual_indices,
            activation,
            lambda order, domain=domain: _conceptual_pairs(domain, tuple(order)),
            exact=domain == "WEEKDAYS",
            n_permutations=10_000,
            seed=stable_seed("V4-GEOMETRY-CORRECTED-PERMUTATION", domain),
        )
        old = old_rows[domain]
        results.append(
            {
                "domain": domain,
                "n_items": len(domain_rows),
                "n_concepts": len(labels),
                "old_tie_broken_spearman": float(old["thinking_activation_spearman"]),
                "corrected_average_rank_spearman": spearman_correlation(
                    conceptual, activation
                ),
                "activation_pearson": pearson_correlation(conceptual, activation),
                "old_permutation_p": float(old["permutation_p"]),
                "corrected_permutation_p": permutation["p_value"],
                "permutation_method": permutation["method"],
                "permutations": permutation["permutations"],
                "corrected_behavior_spearman": spearman_correlation(conceptual, behavior),
                "corrected_activation_behavior_spearman": spearman_correlation(
                    activation, behavior
                ),
                "qualitative_conclusion": (
                    "DESCRIPTIVE_ASSOCIATION_ONLY"
                    if permutation["p_value"] is not None
                    and float(permutation["p_value"]) <= 0.05
                    else "NO_CLEAR_DESCRIPTIVE_ASSOCIATION"
                ),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = reanalyze(args.source)
    source_files = [
        args.source / "GEOMETRY_RESULTS.csv",
        args.source / "GEOMETRY_MANIFEST.json",
        args.source / "geometry_qwen" / "rows.jsonl",
        args.source / "geometry_qwen" / "activations.npz",
    ]
    manifest = {
        "analysis": "Q1_V4_GEOMETRY_TIED_RANK_CORRECTION",
        "source_is_preserved_historical_artifact": True,
        "new_model_execution": False,
        "source_files": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_files
        },
        "results": results,
    }
    (args.output / "reanalysis.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "reanalysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = [
        "# Q1 V4 geometry corrected reanalysis",
        "",
        "This analysis reads preserved local artifacts only. It performs no model",
        "inference and does not overwrite the original V4 report.",
        "",
        "The original Spearman implementation assigned sequential ranks to ties.",
        "The corrected analysis uses average ranks. WEEKDAYS enumerates all 7!",
        "concept-label permutations; LETTERS uses a frozen 10,000-draw Monte Carlo",
        "label permutation with the plus-one correction.",
        "",
        "| domain | old rho | corrected rho | old p | corrected p | method | conclusion |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        report.append(
            f"| {row['domain']} | {row['old_tie_broken_spearman']:.6f} | "
            f"{row['corrected_average_rank_spearman']:.6f} | "
            f"{row['old_permutation_p']:.6g} | {row['corrected_permutation_p']:.6g} | "
            f"{row['permutation_method']} | {row['qualitative_conclusion']} |"
        )
    report += [
        "",
        "Even a corrected association is a tiny, descriptive, single-layer diagnostic.",
        "It is not a behavioral replication, causal intervention result, Q1 result, or",
        "evidence that intervention geometry predicts error covariance.",
    ]
    (args.output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
