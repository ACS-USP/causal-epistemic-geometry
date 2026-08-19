#!/usr/bin/env python3
"""Analyze completed V4 microbench artifacts without model access."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.v4.character_count import STRATA  # noqa: E402
from epistemic_geometry.benchmarks.v4.geometry import conceptual_distance  # noqa: E402
from epistemic_geometry.reproducibility import stable_seed  # noqa: E402

OUT = ROOT / "review" / "q1_v4_microbench"


def _wilson(successes: int, total: int) -> tuple[float, float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _charcount(run: Path) -> dict[str, Any]:
    journal_path = run / "journal.jsonl"
    if not journal_path.exists():
        fields = [
            "stratum",
            "n",
            "valid",
            "valid_rate",
            "correct",
            "wrong",
            "conditional_accuracy",
            "raw_accuracy",
            "invalid_format",
            "truncated",
            "runtime_error",
            "token_mean",
            "token_median",
            "token_max",
            "valid_ci95",
            "promising",
        ]
        with (OUT / "CHARCOUNT_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
        report = """# Character-count microbench report

The authorized character-count worker was interrupted by the V4 hard cost gate
before its first complete trajectory was journaled. This is an operational
non-result, not evidence that character counting is semantically good or bad.
No stratum was evaluated and no stratum was selected.

Decision sentinel for this bounded run: **CHARCOUNT_MICROBENCH_NOT_PROMISING**
(not evaluated; cost-gated).
"""
        (OUT / "CHARCOUNT_REPORT.md").write_text(report, encoding="utf-8")
        return {
            "status": "CHARCOUNT_MICROBENCH_NOT_PROMISING",
            "evaluation_status": "INTERRUPTED_COST_GATE",
            "strata": [],
        }
    rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line]
    complete_run = len(rows) == 30
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["stratum"])].append(row)
    summary: list[dict[str, Any]] = []
    for stratum in STRATA:
        group = by_stratum.get(stratum, [])
        if not group:
            summary.append(
                {
                    "stratum": stratum,
                    "n": 0,
                    "valid": 0,
                    "valid_rate": None,
                    "correct": 0,
                    "wrong": 0,
                    "conditional_accuracy": None,
                    "raw_accuracy": None,
                    "invalid_format": 0,
                    "truncated": 0,
                    "runtime_error": 0,
                    "token_mean": None,
                    "token_median": None,
                    "token_max": None,
                    "valid_ci95": None,
                    "promising": False,
                }
            )
            continue
        counts = Counter(row["status"] for row in group)
        valid = counts["VALID_CORRECT"] + counts["VALID_WRONG"]
        correct = counts["VALID_CORRECT"]
        tokens = [int(row["token_count"]) for row in group if row.get("token_count") is not None]
        summary.append(
            {
                "stratum": stratum,
                "n": len(group),
                "valid": valid,
                "valid_rate": valid / len(group),
                "correct": correct,
                "wrong": counts["VALID_WRONG"],
                "conditional_accuracy": correct / valid if valid else None,
                "raw_accuracy": correct / len(group),
                "invalid_format": counts["INVALID_FORMAT"],
                "truncated": counts["TRUNCATED_THINKING"],
                "runtime_error": counts["RUNTIME_ERROR"],
                "token_mean": float(np.mean(tokens)) if tokens else None,
                "token_median": float(np.median(tokens)) if tokens else None,
                "token_max": max(tokens) if tokens else None,
                "valid_ci95": _wilson(valid, len(group)),
                "promising": (
                    valid / len(group) >= 0.90
                    and counts["VALID_WRONG"] >= 2
                    and counts["VALID_CORRECT"] >= 2
                    and counts["VALID_WRONG"]
                    >= (
                        counts["INVALID_FORMAT"]
                        + counts["TRUNCATED_THINKING"]
                        + counts["RUNTIME_ERROR"]
                    )
                ),
            }
        )
    with (OUT / "CHARCOUNT_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    promising = [row for row in summary if row["promising"]]
    status = "CHARCOUNT_MICROBENCH_PROMISING" if promising else "CHARCOUNT_MICROBENCH_NOT_PROMISING"
    report = [
        "# Character-count microbench report",
        "",
        "This is a development-only screen; n=10 per stratum is not a scientific estimate.",
        "",
        (
            f"Only {len(rows)}/30 authorized trajectories were journaled before the hard "
            "cost gate. The short and medium strata are present; the long stratum was not "
            "run. These partial rows are retained as engineering diagnostics. The V4 "
            "qualification gate was not applied to this incomplete run."
            if not complete_run
            else (
                "All 30 authorized trajectories are present; the frozen V4 screen is "
                "reported below."
            )
        ),
        "",
        "| stratum | valid | correct | wrong | invalid | trunc. | conditional accuracy | "
        "mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        valid_rate = f"{row['valid_rate']:.1%}" if row["valid_rate"] is not None else "not run"
        token_mean = f"{row['token_mean']:.1f}" if row["token_mean"] is not None else "—"
        report.append(
            f"| {row['stratum']} | "
            f"{row['valid']}/{row['n']} "
            f"({valid_rate}) | "
            f"{row['correct']} | {row['wrong']} | {row['invalid_format']} | {row['truncated']} | "
            f"{row['conditional_accuracy']} | "
            f"{token_mean} |"
        )
    report += [
        "",
        (
            "Decision: **CHARCOUNT_MICROBENCH_NOT_PROMISING** (partial run; not evaluated)."
            if not complete_run
            else f"Decision: **{status}**"
        ),
        "",
        "No stratum was selected and no steering was run.",
    ]
    (OUT / "CHARCOUNT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        names = [row["stratum"] for row in summary]
        axis.bar(
            names,
            [row["valid_rate"] if row["valid_rate"] is not None else 0.0 for row in summary],
            label="valid completion",
        )
        axis.bar(
            names,
            [row["wrong"] / row["n"] if row["n"] else 0.0 for row in summary],
            label="genuine wrong",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("fraction of items")
        axis.set_title("V4 character-count development screen")
        axis.legend()
        figure.tight_layout()
        (OUT / "figures").mkdir(exist_ok=True)
        figure.savefig(OUT / "figures" / "charcount_gate.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    return {
        "status": "CHARCOUNT_MICROBENCH_NOT_PROMISING" if not complete_run else status,
        "evaluation_status": "PARTIAL_COST_GATE" if not complete_run else "COMPLETE",
        "complete": complete_run,
        "completed_rows": len(rows),
        "strata": summary,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _corr(left: np.ndarray, right: np.ndarray, *, spearman: bool) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    if spearman:
        left, right = _rank(left), _rank(right)
    return float(np.corrcoef(left, right)[0, 1])


def _pairwise(values: list[np.ndarray], distance: Any) -> np.ndarray:
    return np.asarray(
        [
            distance(values[i], values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
    )


def _hellinger(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.sqrt(np.maximum(left, 0)) - np.sqrt(np.maximum(right, 0))
    return float(np.linalg.norm(difference) / math.sqrt(2))


def _conceptual_pairs(domain: str, indices: list[int]) -> np.ndarray:
    return np.asarray(
        [
            conceptual_distance(domain, indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]
    )


def _geometry(run: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("instrument") != "GEOMETRY":
        raise ValueError("unexpected geometry manifest instrument")
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line]
    arrays = np.load(run / "activations.npz")
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(stable_seed("V4-GEOMETRY-PERMUTATION"))
    for domain in ("WEEKDAYS", "LETTERS"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        # Use the conceptual index stored in the manifest, not lexical label order.
        concept_groups: dict[str, list[int]] = defaultdict(list)
        concept_indices: dict[str, int] = {}
        for row in domain_rows:
            concept_groups[str(row["answer"])].append(int(row["conceptual_index"]))
            concept_indices[str(row["answer"])] = int(row["conceptual_index"])
        concepts = sorted(concept_groups, key=lambda label: concept_indices[label])
        centroids: dict[str, np.ndarray] = {}
        distributions: dict[str, np.ndarray] = {}
        for label in concepts:
            matching = [row for row in domain_rows if str(row["answer"]) == label]
            centroids[label] = np.mean(
                np.stack([arrays[row["thinking_activation_key"]] for row in matching]), axis=0
            )
            probs = [row["direct_probabilities"] for row in matching]
            if all(prob is not None for prob in probs):
                distributions[label] = np.mean(np.asarray(probs, dtype=float), axis=0)
        conceptual = _conceptual_pairs(domain, [concept_indices[label] for label in concepts])
        activation = _pairwise(
            [centroids[label] for label in concepts], lambda x, y: np.linalg.norm(x - y)
        )
        obs_spearman = _corr(conceptual, activation, spearman=True)
        obs_pearson = _corr(conceptual, activation, spearman=False)
        null = []
        for _ in range(10000):
            permuted = rng.permutation(len(concepts))
            permuted_indices = [concept_indices[concepts[index]] for index in permuted]
            permuted_conceptual = _conceptual_pairs(domain, permuted_indices)
            value = _corr(permuted_conceptual, activation, spearman=True)
            if value is not None:
                null.append(value)
        p_value = (
            (1 + sum(abs(value) >= abs(obs_spearman or 0.0) for value in null)) / (1 + len(null))
            if obs_spearman is not None
            else None
        )
        behavior_spearman = behavior_pearson = activation_behavior_spearman = None
        if len(distributions) == len(concepts):
            behavior = _pairwise([distributions[label] for label in concepts], _hellinger)
            behavior_spearman = _corr(conceptual, behavior, spearman=True)
            behavior_pearson = _corr(conceptual, behavior, spearman=False)
            activation_behavior_spearman = _corr(activation, behavior, spearman=True)
        results.append(
            {
                "domain": domain,
                "n_items": len(domain_rows),
                "n_concepts": len(concepts),
                "n_direct_concepts_represented": len(distributions),
                "thinking_activation_spearman": obs_spearman,
                "thinking_activation_pearson": obs_pearson,
                "permutation_p": p_value,
                "direct_behavior_spearman": behavior_spearman,
                "direct_behavior_pearson": behavior_pearson,
                "activation_behavior_spearman": activation_behavior_spearman,
                "permutations": len(null),
            }
        )
        try:
            import matplotlib.pyplot as plt

            matrix = np.stack([centroids[label] for label in concepts])
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            components = centered @ vh[:2].T
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter(components[:, 0], components[:, 1])
            for point, label in zip(components, concepts, strict=True):
                axis.annotate(label, (point[0], point[1]))
            axis.set_title(f"{domain} thinking activation centroids, block 31")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            figure.tight_layout()
            (OUT / "figures").mkdir(exist_ok=True)
            figure.savefig(OUT / "figures" / f"geometry_{domain.lower()}_pca.png", dpi=160)
            plt.close(figure)
        except ImportError:
            pass
    with (OUT / "GEOMETRY_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    signal = any(
        row["permutation_p"] is not None
        and row["permutation_p"] <= 0.05
        and row["thinking_activation_spearman"] is not None
        and abs(row["thinking_activation_spearman"]) >= 0.5
        for row in results
    )
    status = (
        "GEOMETRY_MICROBENCH_SIGNAL_PRESENT"
        if signal
        else "GEOMETRY_MICROBENCH_NO_CLEAR_SIGNAL"
    )
    report = [
        "# Geometry microbench report",
        "",
        "Development-only descriptive screen at the preselected zero-based block 31.",
        "No layer search, steering, manifold fitting, or causal claim is made.",
        "",
        "| domain | activation Spearman | activation Pearson | permutation p | "
        "behavior Spearman | concepts represented |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        report.append(
            f"| {row['domain']} | {row['thinking_activation_spearman']} | "
            f"{row['thinking_activation_pearson']} | {row['permutation_p']} | "
            f"{row['direct_behavior_spearman']} | "
            f"{row['n_direct_concepts_represented']}/{row['n_concepts']} |"
        )
    report += ["", f"Decision: **{status}**"]
    (OUT / "GEOMETRY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"status": status, "domains": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--char-run", type=Path)
    parser.add_argument("--geometry-run", type=Path)
    parser.add_argument("--geometry-manifest", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if args.char_run:
        payload["character_count"] = _charcount(args.char_run)
    if args.geometry_run and args.geometry_manifest:
        payload["geometry"] = _geometry(args.geometry_run, args.geometry_manifest)
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
