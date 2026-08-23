#!/usr/bin/env python3
"""Prospectively staged analysis for Gate 13.1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_gate13_cross_model_ministral3 as parent_analysis  # noqa: E402

from epistemic_geometry.experiments import gate13, gate13_1  # noqa: E402

REVIEW = ROOT / "review/gate13_1_all_layer_causal_atlas"
PARENT = ROOT / "review/gate13_cross_model_ministral3"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Gate 13.1 CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def rows_for(stage: str) -> list[dict[str, Any]]:
    return parent_analysis.journal_rows(REVIEW, stage, gate13_1.MODEL)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    a, b = _ranks(np.asarray(left, dtype=np.float64)), _ranks(
        np.asarray(right, dtype=np.float64)
    )
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _q_and_summary(
    rows: list[dict[str, Any]], condition: str
) -> tuple[float, dict[str, float]]:
    return parent_analysis.matched_q(rows, "BASELINE", condition)


def analyze_sweep() -> dict[str, Any]:
    rows = rows_for("ALL_LAYER_SWEEP")
    schedule = read_json(REVIEW / "ALL_LAYER_SWEEP_SCHEDULE.json")
    parent_analysis.assert_complete(rows, schedule, gate13_1.MODEL)
    _baseline_q, baseline = _q_and_summary(rows, "BASELINE")
    metrics = {}
    table = []
    for layer in range(34):
        condition = f"MEANINGFUL_L{layer}_D50"
        q, summary = _q_and_summary(rows, condition)
        metrics[layer] = {**summary, "Q": q, "baseline_accuracy": baseline["accuracy"]}
        table.append({"layer": layer, **metrics[layer]})
    candidates, eligibility = gate13_1.select_sweep_candidates(metrics)
    source = read_json(REVIEW / "SOURCE_DIRECTION_MANIFEST.json")["layers"]
    effects = [float(row["source_effect"]) for row in source]
    aurocs = [float(row["source_auroc"]) for row in source]
    q_values = [float(metrics[layer]["Q"]) for layer in range(34)]
    diagnostic = {
        "spearman_source_effect_vs_causal_Q": spearman(effects, q_values),
        "spearman_source_auroc_vs_causal_Q": spearman(aurocs, q_values),
        "development_only": True,
        "used_for_selection": False,
        "rank_table": [
            {
                "layer": layer,
                "source_effect": effects[layer],
                "source_auroc": aurocs[layer],
                "causal_Q": q_values[layer],
            }
            for layer in range(34)
        ],
    }
    write_json(REVIEW / "READOUT_CONTROL_DIAGNOSTIC.json", diagnostic)
    write_csv(REVIEW / "ALL_LAYER_SWEEP_RESULTS.csv", table)
    classification = (
        "GATE13_1_ALL_LAYER_SWEEP_PASS"
        if len(candidates) >= 2
        else "GATE13_1_ALL_LAYER_SWEEP_NO_CANDIDATE"
    )
    result = {
        "classification": classification,
        "n_items": 12,
        "layers_tested": 34,
        "metrics": {str(key): value for key, value in metrics.items()},
        "eligibility": {str(key): value for key, value in eligibility.items()},
        "quartile_candidates": candidates,
        "candidate_generation_only": True,
        "accuracy_used_for_ranking": False,
        "readout_control_diagnostic": diagnostic,
    }
    write_json(REVIEW / "ALL_LAYER_SWEEP_REPORT.json", result)
    (REVIEW / "ALL_LAYER_SWEEP_REPORT.md").write_text(
        "# Gate 13.1 all-layer D50 causal sweep\n\n"
        f"Classification: `{classification}`. Quartile candidates: `{candidates}`. "
        "This 12-item matched sweep is candidate generation only.\n",
        encoding="utf-8",
    )
    if len(candidates) < 2:
        return result

    archive = np.load(PARENT / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    differences = archive["construction_careful"].astype(np.float64) - archive[
        "construction_direct"
    ].astype(np.float64)
    direction_dir = REVIEW / "STAGE_B_DIRECTIONS"
    direction_dir.mkdir(exist_ok=True)
    conditions = []
    by_layer = {int(row["layer"]): row for row in source}
    for layer in candidates:
        meaningful = np.load(
            ROOT / by_layer[layer]["vector_path"], allow_pickle=False
        ).astype(np.float64)
        vectors = {
            "MEANINGFUL": meaningful,
            **gate13_1.stage_b_nulls(meaningful, differences[:, layer, :], layer),
        }
        for name, vector in vectors.items():
            path = direction_dir / f"L{layer}_{name}.npy"
            np.save(path, vector.astype(np.float64), allow_pickle=False)
            for dose, fraction in gate13_1.DOSE_FRACTIONS.items():
                condition = f"{name}_L{layer}_{dose}"
                conditions.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "dose": dose,
                        "alpha": float(by_layer[layer]["D100"]) * fraction,
                        "vector_path": str(path.relative_to(REVIEW)),
                        "vector_hash": gate13.vector_sha256(vector),
                    }
                )
    write_json(
        REVIEW / "STAGE_B_DIRECTION_MANIFEST.json",
        {
            "candidate_layers": candidates,
            "conditions": conditions,
            "nulls_frozen_before_stage_b": True,
        },
    )
    items = read_json(REVIEW / "LAYER_DOSE_ITEMS.json")["items"]
    write_json(
        REVIEW / "LAYER_DOSE_SCHEDULE.json",
        gate13_1.build_layer_dose_schedule(
            [str(row["item_id"]) for row in items], candidates
        ),
    )
    write_json(
        REVIEW / "SWEEP_CANDIDATE_LOCK.json",
        {
            "status": "FROZEN_PRE_STAGE_B",
            "candidate_layers": candidates,
            "selection_rule": "maximum eligible Q per historical depth quartile",
            "accuracy_used_for_ranking": False,
            "stage_b_direction_manifest": "STAGE_B_DIRECTION_MANIFEST.json",
            "stage_b_schedule": "LAYER_DOSE_SCHEDULE.json",
        },
    )
    return result


def analyze_layer_dose() -> dict[str, Any]:
    rows = rows_for("LAYER_DOSE_QUALIFICATION")
    schedule = read_json(REVIEW / "LAYER_DOSE_SCHEDULE.json")
    parent_analysis.assert_complete(rows, schedule, gate13_1.MODEL)
    _baseline_q, baseline = _q_and_summary(rows, "BASELINE")
    candidates = read_json(REVIEW / "SWEEP_CANDIDATE_LOCK.json")["candidate_layers"]
    source = {
        int(row["layer"]): row
        for row in read_json(REVIEW / "SOURCE_DIRECTION_MANIFEST.json")["layers"]
    }
    metrics: dict[tuple[int, str], dict[str, float]] = {}
    rows_out = []
    for layer in candidates:
        for dose in gate13_1.DOSE_FRACTIONS:
            meaningful_q, meaningful = _q_and_summary(
                rows, f"MEANINGFUL_L{layer}_{dose}"
            )
            null_q = [
                _q_and_summary(rows, f"{kind}_L{layer}_{dose}")[0]
                for kind in ("ISOTROPIC_NULL", "SHUFFLED_NULL")
            ]
            values = {
                **meaningful,
                "baseline_accuracy": baseline["accuracy"],
                "Q": meaningful_q,
                "isotropic_Q": null_q[0],
                "shuffled_Q": null_q[1],
                "null_mean_Q": float(np.mean(null_q)),
                "null_max_Q": float(np.max(null_q)),
                "Q_minus_null_mean": meaningful_q - float(np.mean(null_q)),
                "Q_minus_null_max": meaningful_q - float(np.max(null_q)),
            }
            metrics[(int(layer), dose)] = values
            rows_out.append({"layer": layer, "dose": dose, **values})
    selected, proof = gate13_1.select_layer_dose(
        metrics,
        {layer: float(source[layer]["source_effect"]) for layer in candidates},
    )
    write_csv(REVIEW / "LAYER_DOSE_RESULTS.csv", rows_out)
    classification = (
        "GATE13_1_LAYER_DOSE_PASS"
        if selected is not None
        else "GATE13_1_NO_SAFE_SPECIFIC_LAYER_DOSE"
    )
    result = {
        "classification": classification,
        "n_items": 28,
        "candidate_layers": candidates,
        "metrics": {f"L{layer}_{dose}": value for (layer, dose), value in metrics.items()},
        "eligibility_proof": proof,
        "selected": None if selected is None else {"layer": selected[0], "dose": selected[1]},
    }
    write_json(REVIEW / "LAYER_DOSE_REPORT.json", result)
    (REVIEW / "LAYER_DOSE_REPORT.md").write_text(
        "# Gate 13.1 joint layer-dose qualification\n\n"
        f"Classification: `{classification}`. Selected: `{selected}`.\n",
        encoding="utf-8",
    )
    if selected is None:
        return result

    layer, dose = selected
    alpha = float(source[layer]["D100"]) * gate13_1.DOSE_FRACTIONS[dose]
    meaningful_path = ROOT / source[layer]["vector_path"]
    meaningful = np.load(meaningful_path, allow_pickle=False).astype(np.float64)
    lock = {
        "status": "FROZEN_PRE_FINAL_EVALUATION",
        "selected_layer": layer,
        "selected_dose": dose,
        "selected_alpha": alpha,
        "meaningful_vector_path": source[layer]["vector_path"],
        "meaningful_vector_hash": source[layer]["canonical_vector_hash"],
        "stage_b_metrics": metrics[selected],
        "selection_proof": proof,
        "accuracy_used_for_ranking": False,
    }
    write_json(REVIEW / "SELECTED_LAYER_DOSE_LOCK.json", lock)
    (REVIEW / "SELECTED_LAYER_DOSE_LOCK.md").write_text(
        "# Gate 13.1 selected layer-dose lock\n\n"
        f"Layer `{layer}`, dose `{dose}`, alpha `{alpha}` are frozen before final outcomes.\n",
        encoding="utf-8",
    )
    archive = np.load(PARENT / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    differences = archive["construction_careful"].astype(np.float64) - archive[
        "construction_direct"
    ].astype(np.float64)
    bank, metadata = gate13_1.final_null_bank(meaningful, differences[:, layer, :], layer)
    final_dir = REVIEW / "FINAL_RANDOM_DIRECTIONS"
    final_dir.mkdir(exist_ok=True)
    for name, vector in bank.items():
        path = final_dir / f"{name}.npy"
        np.save(path, vector.astype(np.float64), allow_pickle=False)
        metadata["records"][name]["vector_path"] = str(path.relative_to(REVIEW))
        metadata["records"][name]["file_sha256"] = gate13.file_sha256(path)
    write_json(REVIEW / "FINAL_RANDOM_BANK.json", metadata)
    items = read_json(REVIEW / "FINAL_EVALUATION_ITEMS.json")["items"]
    write_json(
        REVIEW / "FINAL_EVALUATION_SCHEDULE.json",
        gate13_1.build_final_schedule([str(row["item_id"]) for row in items]),
    )
    write_json(
        REVIEW / "SELECTED_MODEL_LOCK.json",
        {"model": gate13_1.MODEL, "revision": gate13_1.REVISION},
    )
    return result


def analyze_final() -> dict[str, Any]:
    parent_analysis.gate13.BOOTSTRAP_SEED = gate13_1.BOOTSTRAP_SEED
    parent_analysis.gate13.BOOTSTRAP_RESAMPLES = gate13_1.BOOTSTRAP_RESAMPLES
    result = parent_analysis.analyze_final(REVIEW)
    mapping = {
        "GATE13_STRONG_CROSS_MODEL_PROTOCOL_REPLICATION": (
            "GATE13_1_STRONG_CROSS_MODEL_REPLICATION"
        ),
        "GATE13_MINIMUM_CROSS_MODEL_PROTOCOL_REPLICATION": (
            "GATE13_1_MINIMUM_CROSS_MODEL_REPLICATION"
        ),
        "GATE13_CROSS_MODEL_CONTROL_WITHOUT_USEFUL_COMPLEMENTARITY": (
            "GATE13_1_CAUSAL_CONTROL_WITHOUT_USEFUL_COMPLEMENTARITY"
        ),
        "GATE13_CROSS_MODEL_NO_REPLICATION": "GATE13_1_FINAL_NO_REPLICATION",
        "GATE13_CROSS_MODEL_DESTRUCTIVE": "GATE13_1_FINAL_DESTRUCTIVE",
    }
    result["classification"] = mapping[result["classification"]]
    write_json(REVIEW / "ESTIMANDS.json", result)
    (REVIEW / "REPORT.md").write_text(
        "# Gate 13.1 — all-layer causal atlas and joint layer-dose qualification\n\n"
        f"Primary classification: `{result['classification']}`. Historical Gate 13 remains "
        "`GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`. The 57 untouched IDs, Q2, Q3, and holdout "
        "remain untouched.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sweep", "layer-dose", "final"), required=True)
    args = parser.parse_args()
    result = {
        "sweep": analyze_sweep,
        "layer-dose": analyze_layer_dose,
        "final": analyze_final,
    }[args.stage]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
