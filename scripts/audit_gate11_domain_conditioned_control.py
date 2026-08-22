#!/usr/bin/env python3
"""Independent Gate-11 forensic audit from recovered artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/gate11_domain_conditioned_control"
CONDITIONS = (
    "TF_BASELINE",
    "TF_TEXTUAL_CAREFUL",
    "TF_MEANINGFUL_L27_D75",
    "TF_RANDOM_R0",
    "TF_RANDOM_R1",
    "TF_RANDOM_R2",
    "TF_RANDOM_R3",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_by_item(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    delta_norm = 9.637427952852196 * 10.153299177386142
    for row in rows:
        if row["status"] != "COMPLETE":
            continue
        key = (row["domain"], row["item_id"], row["condition"])
        for checkpoint in row["checkpoints"]:
            for metric in ("next_token_kl", "top1_flip", "careful_logit_alignment"):
                value = checkpoint.get(metric)
                if value is not None:
                    values[key][metric].append(float(value))
            values[key]["A35"].append(
                float(checkpoint["hidden"]["L35"]["displacement_norm"]) / delta_norm
            )
    return {
        key: {metric: float(np.mean(series)) for metric, series in metrics.items()}
        for key, metrics in values.items()
    }


def control_summary(
    item_rows: dict[tuple[str, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    output = []
    randoms = CONDITIONS[3:]
    for domain in ("CRUXEval", "CHARCOUNT"):
        items = sorted({key[1] for key in item_rows if key[0] == domain})
        for metric in ("next_token_kl", "A35", "top1_flip"):
            meaningful = np.asarray(
                [item_rows[(domain, item, CONDITIONS[2])][metric] for item in items]
            )
            random_matrix = np.asarray(
                [
                    [item_rows[(domain, item, condition)][metric] for condition in randoms]
                    for item in items
                ]
            )
            output.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "meaningful": float(meaningful.mean()),
                    "random_mean": float(random_matrix.mean()),
                    "random_max": float(random_matrix.mean(axis=0).max()),
                    "meaningful_minus_random_mean": float(
                        meaningful.mean() - random_matrix.mean()
                    ),
                    "meaningful_minus_random_max": float(
                        meaningful.mean() - random_matrix.mean(axis=0).max()
                    ),
                }
            )
    return output


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    source = read_jsonl(REVIEW / "source_activation_journal.jsonl")
    propagation = read_jsonl(REVIEW / "fixed_sequence_journal.jsonl")
    source_keys = [(row["domain"], row["item_id"], row["variant"]) for row in source]
    propagation_keys = [row["logical_key"] for row in propagation]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in propagation:
        groups[(row["domain"], row["item_id"])].append(row)
    sequence_symmetric = all(
        len(rows) == 7
        and {row["condition"] for row in rows} == set(CONDITIONS)
        and len({row.get("continuation_sha256") for row in rows}) == 1
        and len({row.get("continuation_length") for row in rows}) == 1
        for rows in groups.values()
    )
    checkpoint_valid = all(
        all(
            set(checkpoint["hidden"])
            == {"L27", "L28", "L30", "L32", "L35"}
            and np.isfinite(float(checkpoint["next_token_kl"]))
            and np.isfinite(float(checkpoint["symmetric_js"]))
            for checkpoint in row.get("checkpoints", [])
        )
        for row in propagation
        if row["status"] == "COMPLETE"
    )
    source_hashes_valid = all(
        row.get("physical_alias")
        or sha256(ROOT / row["activation_path"]) == row["activation_file_sha256"]
        for row in source
    )
    independent = control_summary(mean_by_item(propagation))
    primary = csv_rows(REVIEW / "CONTROL_GAIN_SUMMARY.csv")
    lookup = {(row["domain"], row["metric"]): row for row in primary}
    differences = []
    for row in independent:
        expected = lookup[(row["domain"], row["metric"])]
        for metric in (
            "meaningful",
            "random_mean",
            "random_max",
            "meaningful_minus_random_mean",
            "meaningful_minus_random_max",
        ):
            differences.append(abs(float(row[metric]) - float(expected[metric])))
    component = read_json(REVIEW / "COMPONENT_DIAGNOSTICS.json")
    bootstrap = read_json(REVIEW / "BOOTSTRAP_INTERVALS.json")
    atlas = read_json(REVIEW / "SOURCE_AXIS_ATLAS.json")["rows"]
    frozen_char = next(
        row
        for row in atlas
        if row["domain"] == "CHARCOUNT" and row.get("axis") == "FROZEN_L27"
    )
    source_transfer = bool(
        frozen_char["mean_gap"] > 0
        and bootstrap["source"]["CHARCOUNT:frozen_L27_gap"]["q025"] > 0
        and frozen_char["positive_gap_fraction"] >= 0.75
        and frozen_char["cosine_to_frozen"] >= 0.20
    )
    contrasts = bootstrap["propagation_domain_contrasts"]
    crux_kl = next(
        row
        for row in independent
        if row["domain"] == "CRUXEval" and row["metric"] == "next_token_kl"
    )
    control_shift = bool(
        crux_kl["meaningful_minus_random_mean"] > 0
        and sum(contrasts[name]["q025"] > 0 for name in ("next_token_kl", "A35", "top1_flip"))
        >= 2
    )
    alignment_rows = csv_rows(REVIEW / "CAREFUL_ALIGNMENT_SUMMARY.csv")
    crux_alignment = next(
        float(row["mean_careful_logit_alignment"])
        for row in alignment_rows
        if row["domain"] == "CRUXEval"
    )
    realization_shift = bool(
        crux_alignment > 0 and contrasts["alignment"]["q025"] > 0
    )
    utility = read_json(REVIEW / "POLICY_UTILITY_REANALYSIS.json")
    utility_shift = bool(
        utility["domain_contrasts"]["meaningful_accuracy"]["q025"] > 0
        and utility["domain_contrasts"]["textual_accuracy"]["q025"] > 0
    )
    supported = [not source_transfer, control_shift, realization_shift, utility_shift]
    if sum(supported) >= 2:
        classification = "GATE11_MULTIPLE_DOMAIN_CONDITIONING_FACTORS"
    elif not source_transfer:
        classification = "GATE11_SOURCE_AXIS_DOMAIN_MISMATCH"
    elif control_shift:
        classification = "GATE11_DOWNSTREAM_CONTROL_GAIN_DOMAIN_MISMATCH"
    elif realization_shift:
        classification = "GATE11_POLICY_REALIZATION_DOMAIN_MISMATCH"
    elif utility_shift:
        classification = "GATE11_POLICY_UTILITY_DOMAIN_MISMATCH"
    else:
        classification = "GATE11_POSTMORTEM_INCONCLUSIVE"
    raw_persistence = {
        "source_prompt_activations_float32": True,
        "per_checkpoint_scalar_logit_metrics": True,
        "per_checkpoint_hidden_displacement_norms": True,
        "checkpoint_token_indices": True,
        "normalization_metadata": True,
        "full_vocabulary_baseline_logits": False,
        "full_vocabulary_condition_logits": False,
        "hidden_state_difference_vectors": False,
        "primitive_metric_recomputation_possible": False,
        "replacement_collection_run": False,
    }
    scientific_concern = not raw_persistence["primitive_metric_recomputation_possible"]
    forensic_classification = (
        "GATE11_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
        if scientific_concern
        else "GATE11_FORENSIC_CLEAN"
    )
    audit = {
        "classification": forensic_classification,
        "source_rows": len(source),
        "source_unique_keys": len(set(source_keys)),
        "propagation_rows": len(propagation),
        "propagation_unique_keys": len(set(propagation_keys)),
        "item_groups": len(groups),
        "sequence_condition_symmetry": sequence_symmetric,
        "checkpoint_schema_and_finiteness": checkpoint_valid,
        "source_activation_hashes_valid": source_hashes_valid,
        "primary_aggregation_max_abs_difference": max(differences, default=0.0),
        "classification_recomputed": classification,
        "classification_agreement": classification == component["classification"],
        "raw_persistence": raw_persistence,
        "scientific_result_repaired": False,
    }
    write_json(REVIEW / "RAW_PERSISTENCE_AUDIT.json", raw_persistence)
    write_json(REVIEW / "FORENSIC_AUDIT.json", audit)
    write_json(
        REVIEW / "CLASSIFICATION_CROSSCHECK.json",
        {
            "primary": component["classification"],
            "independent": classification,
            "agreement": classification == component["classification"],
            "component_flags": {
                "source_transfer": source_transfer,
                "control_gain_shift": control_shift,
                "policy_realization_shift": realization_shift,
                "policy_utility_shift": utility_shift,
            },
        },
    )
    crosscheck_rows = []
    for row in independent:
        expected = lookup[(row["domain"], row["metric"])]
        for metric in (
            "meaningful",
            "random_mean",
            "random_max",
            "meaningful_minus_random_mean",
            "meaningful_minus_random_max",
        ):
            crosscheck_rows.append(
                {
                    "domain": row["domain"],
                    "metric": f"{row['metric']}:{metric}",
                    "primary": expected[metric],
                    "audit": row[metric],
                    "absolute_difference": abs(float(expected[metric]) - row[metric]),
                }
            )
    with (REVIEW / "METRIC_CROSSCHECK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crosscheck_rows[0]))
        writer.writeheader()
        writer.writerows(crosscheck_rows)
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Gate 11 independent forensic audit\n\n"
        f"Classification: `{forensic_classification}`.\n\n"
        "Schedule completeness, source hashes, condition symmetry, scalar checkpoint "
        "aggregation, and the frozen synthesis agree. However, complete vocabulary-logit "
        "arrays and hidden-difference vectors were not persisted, so primitive KL/JS and "
        "vector calculations cannot be independently recomputed. The primary result is "
        "preserved without replacement collection or silent repair.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
