#!/usr/bin/env python3
"""Freeze Gate-12 geometry before any historical semantic outcome reveal."""

from __future__ import annotations

import csv
import hashlib
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
HISTORICAL_RAW = ROOT / "review/gate11_1_artifact_complete_replication/raw_primitives"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_kl(domain: str, item_id: str, label: str, checkpoints: np.ndarray) -> float:
    path = HISTORICAL_RAW / f"{domain.lower()}__{item_id}.npz"
    with np.load(path, allow_pickle=False) as raw:
        baseline = raw["baseline_logits"]
        conditions = [str(value) for value in raw["condition_names"].tolist()]
        historical_checkpoints = raw["checkpoint_token_indices"].astype(np.int64)
        target_condition = (
            "TF_MEANINGFUL_L27_D75" if label == "MEANINGFUL" else f"TF_RANDOM_R{label[-1]}"
        )
        condition = raw["condition_logits"][conditions.index(target_condition)]
        indices = [int(np.where(historical_checkpoints == value)[0][0]) for value in checkpoints]
        return float(np.mean(gate12.categorical_kl(baseline[indices], condition[indices])))


def geometry_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(REVIEW / "RAW_GEOMETRY_MANIFEST.json")
    if manifest["status"] != "COMPLETE" or len(manifest["entries"]) != 112:
        raise RuntimeError("Gate-12 raw geometry is incomplete")
    rows: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"raw geometry hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as raw:
            baseline = raw["baseline_logits"]
            careful = raw["careful_logits"]
            derivatives = raw["jvp_vectors"]
            labels = [str(value) for value in raw["direction_labels"].tolist()]
            targets = raw["target_token_ids"]
            checkpoints = raw["checkpoint_indices"].astype(np.int64)
            for index, label in enumerate(labels):
                q_values = gate12.fisher_energy(baseline, derivatives[index])
                base = {
                    "component": entry["component"],
                    "domain": entry["domain"],
                    "item_id": entry["item_id"],
                    "direction": label,
                    "Q_local": float(np.mean(q_values)),
                    "Q_Hellinger": float(np.mean(q_values) / 4),
                }
                careful_delta = careful.astype(np.float64) - baseline.astype(np.float64)
                alignment = gate12.fisher_cosine(baseline, derivatives[index], careful_delta)
                base["fisher_careful_alignment"] = float(np.nanmean(alignment))
                if entry["component"] == "UTILITY_PREDICTION":
                    utility = gate12.utility_slope(baseline, derivatives[index], targets)
                    base.update(
                        {
                            "U_mean": float(np.mean(utility)),
                            "U_sum": float(np.sum(utility)),
                            "eta_utility": float(
                                np.mean(utility)
                                / np.sqrt(float(np.mean(q_values)) + gate12.EPSILON_Q)
                            ),
                        }
                    )
                else:
                    base.update({"U_mean": "", "U_sum": "", "eta_utility": ""})
                    control_row = {
                        **base,
                        "KL_D75": finite_kl(entry["domain"], entry["item_id"], label, checkpoints),
                    }
                    control.append(control_row)
                rows.append(base)
    return rows, control


def control_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    within = {}
    for domain, values in by_domain.items():
        within[domain] = gate12.spearman(
            np.log([row["Q_local"] + gate12.EPSILON_Q for row in values]),
            np.log([row["KL_D75"] + gate12.EPSILON_KL for row in values]),
        )
    q = np.asarray([row["Q_local"] for row in rows])
    kl = np.asarray([row["KL_D75"] for row in rows])
    domains = [row["domain"] for row in rows]
    pooled = gate12.domain_centered_spearman(
        np.log(q + gate12.EPSILON_Q), np.log(kl + gate12.EPSILON_KL), domains
    )
    rng = np.random.default_rng(gate12.CONTROL_BOOTSTRAP_SEED)
    item_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item_rows[(row["domain"], row["item_id"])].append(row)
    items = {
        domain: sorted({item for current_domain, item in item_rows if current_domain == domain})
        for domain in by_domain
    }
    draws = []
    for _ in range(gate12.BOOTSTRAP_RESAMPLES):
        sampled = []
        for domain, domain_items in items.items():
            selected = rng.choice(domain_items, size=len(domain_items), replace=True)
            for ordinal, item_id in enumerate(selected):
                for row in item_rows[(domain, str(item_id))]:
                    sampled.append({**row, "bootstrap_item": f"{domain}|{ordinal}"})
        value = gate12.domain_centered_spearman(
            np.log(np.asarray([row["Q_local"] for row in sampled]) + gate12.EPSILON_Q),
            np.log(np.asarray([row["KL_D75"] for row in sampled]) + gate12.EPSILON_KL),
            [row["domain"] for row in sampled],
        )
        if value is not None:
            draws.append(value)
    interval = [float(value) for value in np.percentile(draws, [2.5, 97.5])]
    q_contrasts = {}
    for domain, values in by_domain.items():
        meaningful = np.mean([row["Q_local"] for row in values if row["direction"] == "MEANINGFUL"])
        random = np.mean([row["Q_local"] for row in values if row["direction"] != "MEANINGFUL"])
        q_contrasts[domain] = {
            "meaningful": float(meaningful),
            "random_mean": float(random),
            "contrast": float(meaningful - random),
        }
    numerical = read_json(REVIEW / "NUMERICAL_VALIDATION.json")
    supported = bool(
        numerical["classification"] == "GATE12_DIFFERENTIABLE_ENGINEERING_PASS"
        and pooled is not None
        and pooled >= 0.50
        and interval[0] > 0.25
        and all(value > 0 for value in within.values())
        and all(value["contrast"] > 0 for value in q_contrasts.values())
        and numerical["median_local_kl_quadratic_difference"] <= 0.10
    )
    return {
        "pooled_domain_centered_spearman": pooled,
        "bootstrap_interval": interval,
        "within_domain_spearman": within,
        "meaningful_random_Q": q_contrasts,
        "classification": (
            "PULLBACK_CONTROL_PREDICTION_SUPPORTED"
            if supported
            else "PULLBACK_CONTROL_PREDICTION_NOT_ESTABLISHED"
        ),
    }


def main() -> int:
    rows, control = geometry_rows()
    write_csv(REVIEW / "GEOMETRY_ONLY_TABLE.csv", rows)
    write_csv(REVIEW / "CONTROL_PREDICTION.csv", control)
    summary = control_correlation(control)
    write_json(REVIEW / "CONTROL_PREDICTION_SUMMARY.json", summary)
    policy_rows = [row for row in rows if row["component"] == "UTILITY_PREDICTION"]
    policy = {
        domain: {
            label: float(
                np.mean(
                    [
                        row["fisher_careful_alignment"]
                        for row in policy_rows
                        if row["domain"] == domain and row["direction"] == label
                    ]
                )
            )
            for label in ("MEANINGFUL", "RANDOM_R0", "RANDOM_R1", "RANDOM_R2", "RANDOM_R3")
        }
        for domain in ("CRUXEval", "CHARCOUNT")
    }
    write_json(REVIEW / "POLICY_ALIGNMENT_SUMMARY.json", policy)
    manifest = read_json(REVIEW / "RAW_GEOMETRY_MANIFEST.json")
    freeze = {
        "transition": "GEOMETRY_COLLECTION -> GEOMETRY_FREEZE",
        "historical_semantic_outcomes_read": False,
        "raw_manifest_sha256": sha256(REVIEW / "RAW_GEOMETRY_MANIFEST.json"),
        "raw_shard_count": len(manifest["entries"]),
        "geometry_only_table_sha256": sha256(REVIEW / "GEOMETRY_ONLY_TABLE.csv"),
        "control_summary_sha256": sha256(REVIEW / "CONTROL_PREDICTION_SUMMARY.json"),
        "geometry_frozen": True,
    }
    write_json(REVIEW / "GEOMETRY_FREEZE.json", freeze)
    print(json.dumps({"geometry_freeze": True, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
