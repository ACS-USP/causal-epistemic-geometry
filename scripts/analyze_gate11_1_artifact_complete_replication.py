#!/usr/bin/env python3
"""Primary Gate 11.1 recomputation from persisted raw primitive shards only."""

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

from epistemic_geometry.experiments import gate11, gate11_1  # noqa: E402

REVIEW = ROOT / "review/gate11_1_artifact_complete_replication"
HISTORICAL = ROOT / "review/gate11_domain_conditioned_control"
VECTOR = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_contrast(
    left: np.ndarray, right: np.ndarray, seed: int, resamples: int = 5_000
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = (
            left[rng.integers(0, len(left), len(left))].mean()
            - right[rng.integers(0, len(right), len(right))].mean()
        )
    return {
        "point": float(left.mean() - right.mean()),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
    }


def load_shards() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    manifest = read_json(REVIEW / "RAW_SHARD_MANIFEST.json")
    if manifest["status"] != "COMPLETE" or len(manifest["entries"]) != 48:
        raise RuntimeError("Gate 11.1 raw shard manifest is incomplete")
    rows: list[dict[str, Any]] = []
    shards: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"raw shard hash mismatch: {entry['path']}")
        archive = np.load(path, allow_pickle=False)
        required = {
            "baseline_logits",
            "condition_logits",
            "hidden_differences",
            "prompt_token_ids",
            "continuation_token_ids",
            "checkpoint_token_indices",
            "target_next_token_ids",
            "condition_names",
            "propagation_layers",
        }
        if set(archive.files) < required:
            raise RuntimeError(f"raw shard missing arrays: {entry['path']}")
        arrays = {name: archive[name] for name in archive.files}
        if (
            arrays["baseline_logits"].dtype != np.float32
            or arrays["condition_logits"].dtype != np.float32
            or arrays["hidden_differences"].dtype != np.float32
        ):
            raise RuntimeError(f"raw shard dtype mismatch: {entry['path']}")
        if tuple(arrays["condition_names"].tolist()) != gate11_1.CONDITIONS:
            raise RuntimeError(f"condition order mismatch: {entry['path']}")
        if tuple(arrays["propagation_layers"].tolist()) != gate11_1.PROPAGATION_LAYERS:
            raise RuntimeError(f"layer order mismatch: {entry['path']}")
        shards[(str(entry["domain"]), str(entry["item_id"]))] = {**entry, "arrays": arrays}
        labels = entry["snapshot_labels"]
        for condition_index, condition in enumerate(gate11_1.CONDITIONS):
            metrics = gate11_1.logit_metrics_from_arrays(
                arrays["baseline_logits"],
                arrays["condition_logits"][condition_index],
                arrays["target_next_token_ids"],
            )
            hidden = gate11_1.hidden_metrics_from_differences(
                arrays["hidden_differences"][condition_index]
            )
            textual = arrays["condition_logits"][1] - arrays["baseline_logits"]
            displacement = arrays["condition_logits"][condition_index] - arrays["baseline_logits"]
            alignment = np.sum(displacement * textual, axis=-1) / np.maximum(
                np.linalg.norm(displacement, axis=-1) * np.linalg.norm(textual, axis=-1),
                np.finfo(np.float64).tiny,
            )
            for snapshot_index, label in enumerate(labels):
                row = {
                    "domain": entry["domain"],
                    "item_id": entry["item_id"],
                    "condition": condition,
                    "checkpoint": label,
                    "checkpoint_token_index": int(
                        arrays["checkpoint_token_indices"][snapshot_index]
                    ),
                    "next_token_kl": float(metrics["next_token_kl"][snapshot_index]),
                    "symmetric_js": float(metrics["symmetric_js"][snapshot_index]),
                    "logit_l2": float(metrics["logit_l2"][snapshot_index]),
                    "logit_cosine": float(metrics["logit_cosine"][snapshot_index]),
                    "top1_flip": int(metrics["top1_flip"][snapshot_index]),
                    "target_logprob_shift": float(metrics["target_logprob_shift"][snapshot_index]),
                    "careful_logit_alignment": float(alignment[snapshot_index])
                    if condition == gate11.TF_MEANINGFUL
                    else None,
                }
                row.update({key: float(value[snapshot_index]) for key, value in hidden.items()})
                rows.append(row)
    if len(shards) != 48 or len(rows) == 0:
        raise RuntimeError("unexpected raw shard count")
    return rows, shards


def item_aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["item_id"], row["condition"])].append(row)
    metrics = (
        "next_token_kl",
        "symmetric_js",
        "logit_l2",
        "logit_cosine",
        "top1_flip",
        "target_logprob_shift",
        "careful_logit_alignment",
        "A27",
        "A28",
        "A30",
        "A32",
        "A35",
    )
    return {
        key: {
            metric: float(np.mean([row[metric] for row in values if row[metric] is not None]))
            for metric in metrics
            if any(row[metric] is not None for row in values)
        }
        for key, values in grouped.items()
    }


def propagation_outputs(
    item_rows: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    domain_values: dict[str, dict[str, np.ndarray]] = {}
    for domain in ("CRUXEval", "CHARCOUNT"):
        items = sorted(
            {item_id for current_domain, item_id, _ in item_rows if current_domain == domain}
        )
        domain_values[domain] = {}
        for metric in ("next_token_kl", "A35", "top1_flip"):
            meaningful = np.asarray(
                [item_rows[(domain, item, gate11.TF_MEANINGFUL)][metric] for item in items]
            )
            random_matrix = np.asarray(
                [
                    [
                        item_rows[(domain, item, condition)][metric]
                        for condition in gate11.TF_RANDOMS
                    ]
                    for item in items
                ]
            )
            summaries.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "meaningful": float(meaningful.mean()),
                    "random_mean": float(random_matrix.mean()),
                    "random_max": float(random_matrix.mean(axis=0).max()),
                    "meaningful_minus_random_mean": float(meaningful.mean() - random_matrix.mean()),
                    "meaningful_minus_random_max": float(
                        meaningful.mean() - random_matrix.mean(axis=0).max()
                    ),
                }
            )
            domain_values[domain][metric] = meaningful - random_matrix.mean(axis=1)
        crux_alignment = np.asarray(
            [
                item_rows[(domain, item, gate11.TF_MEANINGFUL)]["careful_logit_alignment"]
                for item in items
            ]
        )
        alignment.append(
            {"domain": domain, "mean_careful_logit_alignment": float(crux_alignment.mean())}
        )
        domain_values[domain]["alignment"] = crux_alignment
    contrasts = {
        metric: bootstrap_contrast(
            domain_values["CRUXEval"][metric], domain_values["CHARCOUNT"][metric], 20260825 + index
        )
        for index, metric in enumerate(("next_token_kl", "A35", "top1_flip", "alignment"))
    }
    crux_kl = next(
        row for row in summaries if row["domain"] == "CRUXEval" and row["metric"] == "next_token_kl"
    )
    control_gain = bool(
        crux_kl["meaningful_minus_random_mean"] > 0
        and sum(contrasts[name]["q025"] > 0 for name in ("next_token_kl", "A35", "top1_flip")) >= 2
    )
    realization = bool(
        alignment[0]["mean_careful_logit_alignment"] > 0 and contrasts["alignment"]["q025"] > 0
    )
    return (
        summaries,
        alignment,
        {
            "contrasts": contrasts,
            "control_gain_shift_supported": control_gain,
            "policy_realization_shift_supported": realization,
        },
    )


def source_axis_recomputation() -> dict[str, Any]:
    journal = read_jsonl(HISTORICAL / "source_activation_journal.jsonl")
    frozen = np.load(VECTOR, allow_pickle=False).astype(np.float64).reshape(-1)
    result: dict[str, Any] = {}
    for domain in ("CRUXEval", "CHARCOUNT"):
        rows = [
            row
            for row in journal
            if row["domain"] == domain
            and row["variant"] in ("P1_SOURCE_CAREFUL", "P2_SOURCE_DIRECT")
        ]
        careful = np.stack(
            [
                np.load(ROOT / row["activation_path"], allow_pickle=False)["L27"]
                for row in rows
                if row["variant"] == "P1_SOURCE_CAREFUL"
            ]
        ).astype(np.float64)
        direct = np.stack(
            [
                np.load(ROOT / row["activation_path"], allow_pickle=False)["L27"]
                for row in rows
                if row["variant"] == "P2_SOURCE_DIRECT"
            ]
        ).astype(np.float64)
        gaps = (careful - direct) @ frozen
        direction = (careful - direct).mean(axis=0)
        direction /= np.linalg.norm(direction)
        boot = bootstrap_contrast(
            gaps, np.zeros_like(gaps), 20260825 + (0 if domain == "CRUXEval" else 1)
        )
        result[domain] = {
            "mean_frozen_gap": float(gaps.mean()),
            "positive_gap_fraction": float(np.mean(gaps > 0)),
            "cosine_domain_direction_to_frozen": float(
                np.dot(direction, frozen) / (np.linalg.norm(direction) * np.linalg.norm(frozen))
            ),
            "bootstrap_lower": boot["q025"],
            "item_count": len(gaps),
        }
    char = result["CHARCOUNT"]
    result["source_transfer_supported"] = bool(
        char["mean_frozen_gap"] > 0
        and char["bootstrap_lower"] > 0
        and char["positive_gap_fraction"] >= 0.75
        and char["cosine_domain_direction_to_frozen"] >= 0.20
    )
    return result


def main() -> int:
    rows, shards = load_shards()
    item_rows = item_aggregate(rows)
    summaries, alignment, propagation = propagation_outputs(item_rows)
    source = source_axis_recomputation()
    historical_utility = read_json(HISTORICAL / "POLICY_UTILITY_REANALYSIS.json")
    utility_shift = bool(
        historical_utility["domain_contrasts"]["meaningful_accuracy"]["q025"] > 0
        and historical_utility["domain_contrasts"]["textual_accuracy"]["q025"] > 0
    )
    component_flags = {
        "source_transfer": source["source_transfer_supported"],
        "control_gain_shift": propagation["control_gain_shift_supported"],
        "policy_realization_shift": propagation["policy_realization_shift_supported"],
        "policy_utility_shift": utility_shift,
    }
    classification = gate11.classify_components(**component_flags)
    csv_write(REVIEW / "RECOMPUTED_CONTROL_GAIN.csv", summaries)
    csv_write(REVIEW / "RECOMPUTED_CAREFUL_ALIGNMENT.csv", alignment)
    write_json(
        REVIEW / "RECOMPUTED_COMPONENT_DIAGNOSTICS.json",
        {
            **component_flags,
            "classification": classification,
            "source": source,
            "propagation": propagation,
        },
    )
    write_json(
        REVIEW / "PRIMARY_RECOMPUTATION.json",
        {
            "classification": classification,
            "raw_shards": len(shards),
            "raw_rows": len(rows),
            "item_rows": len(item_rows),
            "source_axis": source,
            "control_gain": {"summary": summaries, "contrasts": propagation["contrasts"]},
            "policy_realization": {
                "summary": alignment,
                "contrast": propagation["contrasts"]["alignment"],
            },
            "policy_utility": {
                "historical_artifact_sha256": hashlib.sha256(
                    (HISTORICAL / "POLICY_UTILITY_REANALYSIS.json").read_bytes()
                ).hexdigest(),
                "supported": utility_shift,
            },
            "historical_gate11_classification": read_json(
                HISTORICAL / "COMPONENT_DIAGNOSTICS.json"
            )["classification"],
            "historical_primary_synthesis": read_json(HISTORICAL / "REPORT.md")
            .split("PRIMARY SYNTHESIS", 1)[1]
            .splitlines()[2]
            .strip()
            if "PRIMARY SYNTHESIS" in (HISTORICAL / "REPORT.md").read_text(encoding="utf-8")
            else None,
            "historical_result_changed": False,
            "primitive_metrics_from_persisted_arrays": True,
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "raw_shards": len(shards),
                "raw_rows": len(rows),
                "max_summary_rows": len(summaries),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
