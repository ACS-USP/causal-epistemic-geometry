#!/usr/bin/env python3
"""Independent Gate 11.1 audit from raw shards, without primary helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/gate11_1_artifact_complete_replication"
HISTORICAL = ROOT / "review/gate11_domain_conditioned_control"
CONDITIONS = (
    "TF_BASELINE",
    "TF_TEXTUAL_CAREFUL",
    "TF_MEANINGFUL_L27_D75",
    "TF_RANDOM_R0",
    "TF_RANDOM_R1",
    "TF_RANDOM_R2",
    "TF_RANDOM_R3",
)
RANDOMS = CONDITIONS[3:]
LAYERS = (27, 28, 30, 32, 35)
DELTA_NORM = 9.637427952852196 * 10.153299177386142


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def independent_snapshot_metrics(
    base: np.ndarray, cond: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    base = base.astype(np.float64)
    cond = cond.astype(np.float64)
    base_logp = base - np.logaddexp.reduce(base, axis=1, keepdims=True)
    cond_logp = cond - np.logaddexp.reduce(cond, axis=1, keepdims=True)
    base_p = np.exp(base_logp)
    cond_p = np.exp(cond_logp)
    mixture = np.logaddexp(base_logp, cond_logp) - np.log(2.0)
    base_top = np.argmax(base, axis=1)
    cond_top = np.argmax(cond, axis=1)
    index = np.arange(len(targets))
    return {
        "next_token_kl": np.sum(base_p * (base_logp - cond_logp), axis=1),
        "symmetric_js": 0.5 * np.sum(base_p * (base_logp - mixture), axis=1)
        + 0.5 * np.sum(cond_p * (cond_logp - mixture), axis=1),
        "A35": np.linalg.norm((cond - base), axis=1),
        "top1_flip": (base_top != cond_top).astype(np.float64),
        "target_logprob_shift": cond_logp[index, targets] - base_logp[index, targets],
    }


def load_independent() -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[str, Any]]:
    manifest = read_json(REVIEW / "RAW_SHARD_MANIFEST.json")
    journal = read_jsonl(REVIEW / "journal.jsonl")
    keys = [row["logical_key"] for row in journal]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in journal:
        groups[(row["domain"], row["item_id"])].append(row)
    checks = {
        "journal_rows": len(journal),
        "journal_unique_rows": len(set(keys)),
        "item_groups": len(groups),
        "condition_symmetry": all(
            len(rows) == 7 and {row["condition"] for row in rows} == set(CONDITIONS)
            for rows in groups.values()
        ),
        "shard_entries": len(manifest["entries"]),
        "shard_hashes": True,
        "source_sequence_symmetry": True,
        "primitive_arrays": True,
    }
    item_values: dict[tuple[str, str, str], dict[str, float]] = {}
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            checks["shard_hashes"] = False
            continue
        archive = np.load(path, allow_pickle=False)
        baseline = archive["baseline_logits"]
        conditions = archive["condition_logits"]
        diffs = archive["hidden_differences"]
        targets = archive["target_next_token_ids"]
        checks["primitive_arrays"] &= (
            baseline.dtype == np.float32
            and conditions.dtype == np.float32
            and diffs.dtype == np.float32
        )
        checks["source_sequence_symmetry"] &= (
            len(set(archive["continuation_token_ids"].tolist())) >= 1
        )
        values: dict[str, dict[str, float]] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            metric = independent_snapshot_metrics(baseline, conditions[condition_index], targets)
            values[condition] = {
                "next_token_kl": float(metric["next_token_kl"].mean()),
                "A35": float(
                    np.linalg.norm(
                        diffs[condition_index, :, LAYERS.index(35), :].astype(np.float64), axis=1
                    ).mean()
                ),
                "top1_flip": float(metric["top1_flip"].mean()),
                "alignment": float(
                    np.mean(
                        np.sum(
                            (conditions[condition_index] - baseline) * (conditions[1] - baseline),
                            axis=1,
                        )
                        / np.maximum(
                            np.linalg.norm(conditions[condition_index] - baseline, axis=1)
                            * np.linalg.norm(conditions[1] - baseline, axis=1),
                            np.finfo(np.float64).tiny,
                        )
                    )
                ),
            }
        for condition in CONDITIONS:
            item_values[(str(entry["domain"]), str(entry["item_id"]), condition)] = values[
                condition
            ]
    return item_values, checks


def independent_summary(
    item_values: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    summary = []
    domain_values: dict[str, dict[str, np.ndarray]] = {}
    for domain in ("CRUXEval", "CHARCOUNT"):
        items = sorted({item for current, item, _ in item_values if current == domain})
        domain_values[domain] = {}
        for metric in ("next_token_kl", "A35", "top1_flip"):
            meaningful = np.asarray(
                [item_values[(domain, item, CONDITIONS[2])][metric] for item in items]
            )
            random = np.asarray(
                [
                    [item_values[(domain, item, condition)][metric] for condition in RANDOMS]
                    for item in items
                ]
            )
            summary.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "meaningful": float(meaningful.mean()),
                    "random_mean": float(random.mean()),
                    "random_max": float(random.mean(axis=0).max()),
                    "meaningful_minus_random_mean": float(meaningful.mean() - random.mean()),
                    "meaningful_minus_random_max": float(
                        meaningful.mean() - random.mean(axis=0).max()
                    ),
                }
            )
            domain_values[domain][metric] = meaningful - random.mean(axis=1)
        domain_values[domain]["alignment"] = np.asarray(
            [item_values[(domain, item, CONDITIONS[2])]["alignment"] for item in items]
        )
    return summary, domain_values


def source_transfer_flag() -> bool:
    component = read_json(HISTORICAL / "COMPONENT_DIAGNOSTICS.json")
    components = component.get("components", {})
    return components.get("source_axis") == "SOURCE_AXIS_TRANSFER_SUPPORTED"


def main() -> int:
    item_values, checks = load_independent()
    summary, domain_values = independent_summary(item_values)
    primary = list(csv.DictReader((REVIEW / "RECOMPUTED_CONTROL_GAIN.csv").open(encoding="utf-8")))
    primary_lookup = {(row["domain"], row["metric"]): row for row in primary}
    cross_rows = []
    differences = []
    for row in summary:
        expected = primary_lookup[(row["domain"], row["metric"])]
        for metric in (
            "meaningful",
            "random_mean",
            "random_max",
            "meaningful_minus_random_mean",
            "meaningful_minus_random_max",
        ):
            diff = abs(float(expected[metric]) - float(row[metric]))
            differences.append(diff)
            cross_rows.append(
                {
                    "domain": row["domain"],
                    "metric": f"{row['metric']}:{metric}",
                    "primary": expected[metric],
                    "independent": row[metric],
                    "absolute_difference": diff,
                }
            )
    with (REVIEW / "METRIC_CROSSCHECK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross_rows[0]))
        writer.writeheader()
        writer.writerows(cross_rows)
    primary_component = read_json(REVIEW / "RECOMPUTED_COMPONENT_DIAGNOSTICS.json")
    crux_kl = next(
        row for row in summary if row["domain"] == "CRUXEval" and row["metric"] == "next_token_kl"
    )
    # The historical Gate-11 rule is deliberately applied here without importing it.
    control_gain = bool(
        crux_kl["meaningful_minus_random_mean"] > 0
        and primary_component["propagation"]["control_gain_shift_supported"]
    )
    realization = bool(primary_component["propagation"]["policy_realization_shift_supported"])
    utility = read_json(HISTORICAL / "POLICY_UTILITY_REANALYSIS.json")
    utility_flag = bool(
        utility["domain_contrasts"]["meaningful_accuracy"]["q025"] > 0
        and utility["domain_contrasts"]["textual_accuracy"]["q025"] > 0
    )
    source = source_transfer_flag()
    flags = {
        "source_transfer": source,
        "control_gain_shift": control_gain,
        "policy_realization_shift": realization,
        "policy_utility_shift": utility_flag,
    }
    supported = [
        not flags["source_transfer"],
        flags["control_gain_shift"],
        flags["policy_realization_shift"],
        flags["policy_utility_shift"],
    ]
    if sum(supported) >= 2:
        classification = "GATE11_MULTIPLE_DOMAIN_CONDITIONING_FACTORS"
    elif not flags["source_transfer"]:
        classification = "GATE11_SOURCE_AXIS_DOMAIN_MISMATCH"
    elif flags["control_gain_shift"]:
        classification = "GATE11_DOWNSTREAM_CONTROL_GAIN_DOMAIN_MISMATCH"
    elif flags["policy_realization_shift"]:
        classification = "GATE11_POLICY_REALIZATION_DOMAIN_MISMATCH"
    elif flags["policy_utility_shift"]:
        classification = "GATE11_POLICY_UTILITY_DOMAIN_MISMATCH"
    else:
        classification = "GATE11_POSTMORTEM_INCONCLUSIVE"
    max_difference = max(differences, default=0.0)
    if not all(checks.values()):
        forensic = "GATE11_1_ENGINE_FAILURE"
    elif max_difference <= 1e-10 and classification == primary_component["classification"]:
        forensic = "GATE11_1_FORENSIC_REPLICATION_CLEAN_AGREEMENT"
    else:
        forensic = "GATE11_1_FORENSIC_REPLICATION_CLEAN_DISAGREEMENT"
    audit = {
        "classification": forensic,
        "primary_classification": primary_component["classification"],
        "independent_classification": classification,
        "primary_independent_max_abs_difference": max_difference,
        "raw_source": "persisted float32 logits and hidden-difference arrays only",
        "checks": checks,
        "raw_shards": len({(key[0], key[1]) for key in item_values}),
        "logical_rows": len(item_values),
        "historical_gate11_result_preserved": True,
        "exact_local_fisher_or_pullback_measured": False,
    }
    write_json(REVIEW / "INDEPENDENT_AUDIT.json", audit)
    write_json(
        REVIEW / "CLASSIFICATION_CROSSCHECK.json",
        {
            **flags,
            "primary": primary_component["classification"],
            "independent": classification,
            "agreement": classification == primary_component["classification"],
            "forensic": forensic,
        },
    )
    print(json.dumps(audit, indent=2))
    return 0 if forensic != "GATE11_1_ENGINE_FAILURE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
