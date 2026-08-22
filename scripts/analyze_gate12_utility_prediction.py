#!/usr/bin/env python3
"""Reveal preserved historical outcomes after Gate-12 geometry freeze."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate12  # noqa: E402

REVIEW = ROOT / "review/gate12_utility_aligned_pullback"
GATE9 = ROOT / "review/gate9_selected_d75_evaluation"
GATE10 = ROOT / "review/gate10_cross_domain_charcount"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def outcome_map(path: Path) -> dict[tuple[str, str, int], bool]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
            if key in result:
                raise RuntimeError(f"duplicate historical outcome key: {key}")
            result[key] = row["status"] == "VALID_CORRECT"
    return result


def condition_for(label: str) -> str:
    return "MEANINGFUL_L27_D75" if label == "MEANINGFUL" else f"RANDOM_L27_D75_R{label[-1]}"


def utility_rows() -> list[dict[str, Any]]:
    freeze = read_json(REVIEW / "GEOMETRY_FREEZE.json")
    if not freeze.get("geometry_frozen") or freeze["historical_semantic_outcomes_read"]:
        raise RuntimeError("historical outcome reveal attempted before clean geometry freeze")
    geometry = [
        row
        for row in read_csv(REVIEW / "GEOMETRY_ONLY_TABLE.csv")
        if row["component"] == "UTILITY_PREDICTION"
    ]
    outcomes = {
        "CRUXEval": outcome_map(GATE9 / "journal.jsonl"),
        "CHARCOUNT": outcome_map(GATE10 / "journal.jsonl"),
    }
    result = []
    for row in geometry:
        domain = row["domain"]
        item_id = row["item_id"]
        condition = condition_for(row["direction"])
        baseline = {rollout: outcomes[domain][(item_id, "BASELINE", rollout)] for rollout in (0, 1)}
        treatment = {rollout: outcomes[domain][(item_id, condition, rollout)] for rollout in (0, 1)}
        result.append(
            {
                **row,
                "U_mean": float(row["U_mean"]),
                "U_sum": float(row["U_sum"]),
                "eta_utility": float(row["eta_utility"]),
                "Q_local": float(row["Q_local"]),
                "fisher_careful_alignment": float(row["fisher_careful_alignment"]),
                "historical_condition": condition,
                "Y": gate12.historical_utility_target(baseline, treatment),
            }
        )
    return result


def standardized_within(values: np.ndarray, domains: list[str]) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    labels = np.asarray(domains, dtype=object)
    for domain in sorted(set(domains)):
        mask = labels == domain
        centered = values[mask] - np.mean(values[mask])
        scale = float(np.std(centered, ddof=0))
        result[mask] = centered / scale if scale > 0 else 0.0
    return result


def slope(rows: list[dict[str, Any]]) -> float:
    domains = [row["domain"] for row in rows]
    x = standardized_within(np.asarray([row["U_mean"] for row in rows]), domains)
    y = np.asarray([row["Y"] for row in rows], dtype=np.float64)
    labels = np.asarray(domains, dtype=object)
    for domain in set(domains):
        y[labels == domain] -= np.mean(y[labels == domain])
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def auroc(rows: list[dict[str, Any]]) -> float | None:
    positive = np.asarray([row["U_mean"] for row in rows if row["Y"] > 0])
    negative = np.asarray([row["U_mean"] for row in rows if row["Y"] < 0])
    if len(positive) == 0 or len(negative) == 0:
        return None
    return float(
        np.mean(
            (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
        )
    )


def bootstrap(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    rng = np.random.default_rng(gate12.BOOTSTRAP_SEED)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["item_id"])].append(row)
    items = {
        domain: sorted({item for current_domain, item in grouped if current_domain == domain})
        for domain in ("CRUXEval", "CHARCOUNT")
    }
    rho_draws = []
    slope_draws = []
    contrast_draws = []
    for _ in range(gate12.BOOTSTRAP_RESAMPLES):
        sample = []
        sampled_meaningful: dict[str, list[float]] = defaultdict(list)
        for domain, domain_items in items.items():
            selected = rng.choice(domain_items, size=len(domain_items), replace=True)
            for item_id in selected:
                item_rows = grouped[(domain, str(item_id))]
                sample.extend(item_rows)
                sampled_meaningful[domain].extend(
                    row["U_mean"] for row in item_rows if row["direction"] == "MEANINGFUL"
                )
        rho = gate12.domain_centered_spearman(
            np.asarray([row["U_mean"] for row in sample]),
            np.asarray([row["Y"] for row in sample]),
            [row["domain"] for row in sample],
        )
        if rho is not None:
            rho_draws.append(rho)
        slope_draws.append(slope(sample))
        contrast_draws.append(
            float(
                np.mean(sampled_meaningful["CRUXEval"]) - np.mean(sampled_meaningful["CHARCOUNT"])
            )
        )
    return {
        "item_level_spearman": [float(value) for value in np.percentile(rho_draws, [2.5, 97.5])],
        "cluster_robust_slope": [float(value) for value in np.percentile(slope_draws, [2.5, 97.5])],
        "crux_minus_char_meaningful_U": [
            float(value) for value in np.percentile(contrast_draws, [2.5, 97.5])
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rho = gate12.domain_centered_spearman(
        np.asarray([row["U_mean"] for row in rows]),
        np.asarray([row["Y"] for row in rows]),
        [row["domain"] for row in rows],
    )
    intervals = bootstrap(rows)
    slope_value = slope(rows)
    within = {
        domain: gate12.spearman(
            [row["U_mean"] for row in rows if row["domain"] == domain],
            [row["Y"] for row in rows if row["domain"] == domain],
        )
        for domain in ("CRUXEval", "CHARCOUNT")
    }
    direction_means = {}
    for domain in ("CRUXEval", "CHARCOUNT"):
        direction_means[domain] = {}
        for label in ("MEANINGFUL", "RANDOM_R0", "RANDOM_R1", "RANDOM_R2", "RANDOM_R3"):
            selected = [
                row for row in rows if row["domain"] == domain and row["direction"] == label
            ]
            direction_means[domain][label] = {
                "U_mean": float(np.mean([row["U_mean"] for row in selected])),
                "Y_mean": float(np.mean([row["Y"] for row in selected])),
            }
    crux_meaningful = direction_means["CRUXEval"]["MEANINGFUL"]["U_mean"]
    char_meaningful = direction_means["CHARCOUNT"]["MEANINGFUL"]["U_mean"]
    crux_random = float(
        np.mean([direction_means["CRUXEval"][f"RANDOM_R{i}"]["U_mean"] for i in range(4)])
    )
    item_supported = bool(
        rho is not None
        and rho >= 0.20
        and intervals["item_level_spearman"][0] > 0
        and slope_value > 0
        and intervals["cluster_robust_slope"][0] > 0
    )
    domain_supported = bool(
        crux_meaningful > 0
        and crux_meaningful > crux_random
        and crux_meaningful - char_meaningful > 0
        and intervals["crux_minus_char_meaningful_U"][0] > 0
        and direction_means["CRUXEval"]["MEANINGFUL"]["Y_mean"] > 0
        and direction_means["CHARCOUNT"]["MEANINGFUL"]["Y_mean"] < 0
        and char_meaningful < 0
    )
    return {
        "pooled_domain_centered_spearman": rho,
        "within_domain_spearman": within,
        "cluster_robust_slope": slope_value,
        "auroc_positive_vs_negative_Y": auroc(rows),
        "direction_means": direction_means,
        "crux_minus_char_meaningful_U": crux_meaningful - char_meaningful,
        "bootstrap_intervals": intervals,
        "item_level_utility_prediction": "SUPPORTED" if item_supported else "NOT_SUPPORTED",
        "domain_level_utility_alignment": "SUPPORTED" if domain_supported else "NOT_SUPPORTED",
    }


def main() -> int:
    rows = utility_rows()
    write_csv(REVIEW / "UTILITY_PREDICTION.csv", rows)
    summary = summarize(rows)
    write_json(REVIEW / "UTILITY_PREDICTION_SUMMARY.json", summary)
    control = read_json(REVIEW / "CONTROL_PREDICTION_SUMMARY.json")
    control_supported = control["classification"] == "PULLBACK_CONTROL_PREDICTION_SUPPORTED"
    item_supported = summary["item_level_utility_prediction"] == "SUPPORTED"
    domain_supported = summary["domain_level_utility_alignment"] == "SUPPORTED"
    classification = gate12.classify(
        control_supported=control_supported,
        item_utility_supported=item_supported,
        domain_utility_supported=domain_supported,
    )
    write_json(
        REVIEW / "COMPONENT_DIAGNOSTICS.json",
        {
            "control_prediction": control["classification"],
            "item_utility_prediction": summary["item_level_utility_prediction"],
            "domain_utility_alignment": summary["domain_level_utility_alignment"],
            "primary_classification": classification,
        },
    )
    write_json(
        REVIEW / "BOOTSTRAP_INTERVALS.json",
        {
            "control": control["bootstrap_interval"],
            "utility": summary["bootstrap_intervals"],
            "unit": "item with every direction",
            "resamples": gate12.BOOTSTRAP_RESAMPLES,
        },
    )
    freeze = read_json(REVIEW / "GEOMETRY_FREEZE.json")
    freeze["transition_after_freeze"] = "GEOMETRY_FREEZE -> HISTORICAL_OUTCOME_REVEAL"
    freeze["historical_semantic_outcomes_read"] = True
    write_json(REVIEW / "GEOMETRY_FREEZE.json", freeze)
    print(json.dumps({"classification": classification, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
