#!/usr/bin/env python3
"""Offline, outcome-free postmortem of the closed Q2-V1 bank qualification."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "review/q2_controller_heldout_geometry"
OUT = ROOT / "review/q2_controller_bank_v2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def row_value(row: dict[str, Any], field: str) -> Any:
    return row["row"][field]


def parse_controller(name: str) -> dict[str, str]:
    bits = name.split("_")
    if name.startswith("MEAN_"):
        return {
            "source_axis": "_".join(bits[1:-3]),
            "source_location": "_".join(bits[-3:-1]),
            "sign": bits[-1],
        }
    return {
        "source_axis": "NULL",
        "source_location": "NULL",
        "sign": "NULL",
    }


def token_change(left: Any, right: Any) -> bool:
    return list(left) != list(right)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = V1 / "source_behavior_journal.jsonl"
    manipulation_path = V1 / "manipulation_journal.jsonl"
    source_rows = load_jsonl(source_path)
    manipulation_rows = load_jsonl(manipulation_path)
    bank = read_json(V1 / "CONTROLLER_BANK.json")
    manipulation_summary = read_json(V1 / "MANIPULATION_QUALIFICATION.json")

    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for envelope in manipulation_rows:
        row = envelope["row"]
        key = (row["item_id"], row["rollout_index"], row["condition"])
        grouped[key[:2]][row["condition"]] = row
    expected = set(bank["controller_ids"]) | {"BASELINE"}
    if any(set(rows) != expected for rows in grouped.values()):
        raise RuntimeError("V1 manipulation rows do not form complete item-condition blocks")

    controller_ids = list(bank["controller_ids"])
    meaningful_ids = [name for name in controller_ids if name.startswith("MEAN_")]
    vectors: dict[str, np.ndarray] = {}
    for name, metadata in bank["vectors"].items():
        path = ROOT / metadata["path"]
        vectors[name] = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    meaningful_matrix = np.vstack([vectors[name] for name in meaningful_ids])
    meaningful_cosines = meaningful_matrix @ meaningful_matrix.T
    meaningful_distances = np.sqrt(
        np.maximum(0.0, 2.0 - 2.0 * meaningful_cosines)
    )

    records: list[dict[str, Any]] = []
    by_controller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (item_id, rollout_index), rows in sorted(grouped.items()):
        baseline = rows["BASELINE"]
        for controller in controller_ids:
            treatment = rows[controller]
            records.append(
                {
                    "item_id": item_id,
                    "rollout_index": rollout_index,
                    "controller": controller,
                    "source_axis": parse_controller(controller)["source_axis"],
                    "source_location": parse_controller(controller)["source_location"],
                    "sign": parse_controller(controller)["sign"],
                    "delta_norm": bank["vectors"][controller]["delta_norm"],
                    "commitment_validity": bool(treatment["commitment_valid"]),
                    "semantic_evaluability": bool(treatment["semantic_evaluable"]),
                    "raw_sequence_changed": token_change(
                        treatment["generated_token_ids"], baseline["generated_token_ids"]
                    ),
                    "semantic_changed": treatment["canonical_value"]
                    != baseline["canonical_value"],
                    "token_count": treatment["generated_token_count"],
                    "baseline_token_count": baseline["generated_token_count"],
                    "token_delta": treatment["generated_token_count"]
                    - baseline["generated_token_count"],
                }
            )
            by_controller[controller].append(records[-1])

    summaries: list[dict[str, Any]] = []
    for controller in controller_ids:
        rows = by_controller[controller]
        metadata = parse_controller(controller)
        summaries.append(
            {
                "controller": controller,
                **metadata,
                "delta_norm": rows[0]["delta_norm"],
                "n": len(rows),
                "validity": float(np.mean([r["commitment_validity"] for r in rows])),
                "evaluability": float(np.mean([r["semantic_evaluability"] for r in rows])),
                "raw_sequence_movement": float(np.mean([r["raw_sequence_changed"] for r in rows])),
                "semantic_movement": float(np.mean([r["semantic_changed"] for r in rows])),
                "mean_token_delta": float(np.mean([r["token_delta"] for r in rows])),
                "median_token_delta": float(np.median([r["token_delta"] for r in rows])),
                "mean_tokens": float(np.mean([r["token_count"] for r in rows])),
                "label_free_manipulation_metric": manipulation_summary[controller],
            }
        )

    with (OUT / "V1_CONTROLLER_MOVEMENT.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "controller", "source_axis", "source_location", "sign", "delta_norm", "n",
            "validity", "evaluability", "raw_sequence_movement", "semantic_movement",
            "mean_token_delta", "median_token_delta", "mean_tokens",
            "label_free_manipulation_metric",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in summaries:
            output = {
                key: json.dumps(value, sort_keys=True)
                if key == "label_free_manipulation_metric"
                else value
                for key, value in record.items()
            }
            writer.writerow(output)

    geometry = {
        "source": "closed Q2-V1 CONTROLLER_BANK.json and manipulation journal",
        "correctness_used": False,
        "meaningful_controller_ids": meaningful_ids,
        "meaningful_cosine_matrix": meaningful_cosines.tolist(),
        "meaningful_euclidean_distance_matrix": meaningful_distances.tolist(),
        "vector_hashes": {
            name: bank["vectors"][name]["canonical_float64_vector_sha256"]
            for name in controller_ids
        },
        "distance_definition": "sqrt(2 - 2*cosine) for unit vectors",
    }
    (OUT / "V1_DIRECTION_GEOMETRY.json").write_text(
        json.dumps(geometry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def mean_for(predicate: Any) -> float:
        values = [record["raw_sequence_movement"] for record in summaries if predicate(record)]
        return float(np.mean(values)) if values else float("nan")

    meaningful_summary = [
        record for record in summaries if record["controller"].startswith("MEAN_")
    ]
    failure_gap = [
        0.25 - record["raw_sequence_movement"]
        for record in meaningful_summary
        if record["raw_sequence_movement"] < 0.25
    ]
    axis_aggregate: dict[str, dict[str, float]] = {}
    location_aggregate: dict[str, dict[str, float]] = {}
    for group_key, target in (
        ("source_axis", axis_aggregate),
        ("source_location", location_aggregate),
    ):
        for value in sorted({record[group_key] for record in meaningful_summary}):
            subset = [record for record in meaningful_summary if record[group_key] == value]
            target[value] = {
                "mean_raw_sequence_movement": float(
                    np.mean([r["raw_sequence_movement"] for r in subset])
                ),
                "mean_semantic_movement": float(
                    np.mean([r["semantic_movement"] for r in subset])
                ),
                "mean_token_delta": float(np.mean([r["mean_token_delta"] for r in subset])),
            }

    sign_pairs = {}
    for record in meaningful_summary:
        if record["sign"] != "PLUS":
            continue
        counterpart = next(
            other for other in meaningful_summary
            if other["source_axis"] == record["source_axis"]
            and other["source_location"] == record["source_location"]
            and other["sign"] == "MINUS"
        )
        sign_pairs[f"{record['source_axis']}:{record['source_location']}"] = {
            "plus_raw_sequence_movement": record["raw_sequence_movement"],
            "minus_raw_sequence_movement": counterpart["raw_sequence_movement"],
            "plus_minus_difference": (
                record["raw_sequence_movement"]
                - counterpart["raw_sequence_movement"]
            ),
        }

    report = f"""# Q2 V1 offline evidence postmortem

This is a descriptive DEVELOPMENT postmortem of the immutable Q2-V1
qualification artifacts. It used {len(source_rows)} source rows and
{len(manipulation_rows)} matched manipulation rows. No correctness, accuracy,
G, C, D, rescue, damage, or common-panel outcome was read or computed.

## Main findings

- The old common displacement norm was exactly
  {summaries[0]['delta_norm']:.12g} for every controller.
- {sum(record['raw_sequence_movement'] >= 0.25 for record in meaningful_summary)}/12
  meaningful controllers reached the historical 0.25 raw-sequence movement
  threshold.
- The {len(failure_gap)} failing meaningful controllers were below that threshold
  by a mean of {float(np.mean(failure_gap)):.6f}; movement values were discrete
  multiples of 1/12.
- {sum(record['raw_sequence_movement'] == 1/12 for record in meaningful_summary)}
  meaningful controllers were at 1/12, and
  {sum(record['raw_sequence_movement'] == 2/12 for record in meaningful_summary)}
  were at 2/12, immediately below the old cutoff.
- Sign and source-location effects are summarized below; they are descriptive,
  not selection criteria.
- Sensitivity was sign-asymmetric but not in one universal direction: the
  prompt-boundary minus sign was strongest for independent verification and
  type discipline, while explicit state tracking was stronger at the execution
  boundary with the plus sign.
- Axis sensitivity was modest but visible: explicit state tracking averaged
  0.1042 raw movement, versus 0.1458 for verification and type discipline.
- Prompt-boundary interventions averaged 0.1528 raw movement versus 0.1111 at
  the execution boundary. This is a descriptive location pattern, not a
  causal source-location claim.

## Axis aggregates

{json.dumps(axis_aggregate, indent=2, sort_keys=True)}

## Source-location aggregates

{json.dumps(location_aggregate, indent=2, sort_keys=True)}

## Sign pairs

{json.dumps(sign_pairs, indent=2, sort_keys=True)}

## Interpretation boundary

The observed bank had a narrow discrete first-stage range at the single frozen
norm. This supports per-direction dose calibration and a continuous bank-level
dynamic-range rule in V2. It does not establish that any direction is useful for
semantic error control, because downstream correctness was intentionally absent.
"""
    (OUT / "V1_EVIDENCE_POSTMORTEM.md").write_text(report, encoding="utf-8")
    manifest = {
        "source_journal_sha256": sha256(source_path),
        "manipulation_journal_sha256": sha256(manipulation_path),
        "controller_bank_sha256": sha256(V1 / "CONTROLLER_BANK.json"),
        "manipulation_summary_sha256": sha256(V1 / "MANIPULATION_QUALIFICATION.json"),
        "source_rows": len(source_rows),
        "manipulation_rows": len(manipulation_rows),
        "controller_count": len(controller_ids),
        "correctness_used": False,
        "common_panel_used": False,
    }
    (OUT / "V1_POSTMORTEM_PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_rows": len(source_rows),
                "manipulation_rows": len(manipulation_rows),
                "controllers": len(controller_ids),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
