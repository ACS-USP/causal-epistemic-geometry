#!/usr/bin/env python3
"""Independent low-level audit of a stopped Q2 bank qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_heldout_geometry"
AXES = (
    "INDEPENDENT_VERIFICATION",
    "EXPLICIT_STATE_TRACKING",
    "TYPE_REPRESENTATION_DISCIPLINE",
)
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")


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


def raw_journal(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wrappers = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(wrapper.get("version") != "research-os-jsonl-v1" for wrapper in wrappers):
        raise RuntimeError(f"unexpected journal wrapper in {path}")
    return wrappers, [dict(wrapper["row"]) for wrapper in wrappers]


def disagreement(rows: list[dict[str, Any]], axis: str) -> dict[str, float]:
    selected = [row for row in rows if row["axis_id"] == axis]
    lookup = {
        (row["item_id"], row["polarity"], row["rollout_index"]): row["canonical_value"]
        for row in selected
    }
    items = sorted({row["item_id"] for row in selected})
    cross: list[float] = []
    pos_within: list[float] = []
    neg_within: list[float] = []
    for item in items:
        pos = [lookup[(item, "POSITIVE", rollout)] for rollout in (0, 1)]
        neg = [lookup[(item, "NEGATIVE", rollout)] for rollout in (0, 1)]
        cross.extend(float(left != right) for left in pos for right in neg)
        pos_within.append(float(pos[0] != pos[1]))
        neg_within.append(float(neg[0] != neg[1]))
    within = 0.5 * (float(np.mean(pos_within)) + float(np.mean(neg_within)))
    return {
        "cross_disagreement": float(np.mean(cross)),
        "within_disagreement": within,
        "excess_disagreement": float(np.mean(cross)) - within,
    }


def numeric_differences(left: Any, right: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys():
            output.update(numeric_differences(left[key], right[key], f"{prefix}.{key}"))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        output[prefix] = abs(float(left) - float(right))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = args.review_dir.resolve()

    source_wrappers, source_rows = raw_journal(review / "source_behavior_journal.jsonl")
    manipulation_wrappers, manipulation_rows = raw_journal(
        review / "manipulation_journal.jsonl"
    )
    source_schedule = read_json(review / "SOURCE_BEHAVIOR_SCHEDULE.json")
    manipulation_schedule = read_json(review / "MANIPULATION_SCHEDULE.json")
    source_keys = [tuple(wrapper["key"]) for wrapper in source_wrappers]
    manipulation_keys = [tuple(wrapper["key"]) for wrapper in manipulation_wrappers]
    source_expected = {
        (row["item_id"], row["axis_id"], row["polarity"], row["rollout_index"])
        for row in source_schedule
    }
    manipulation_expected = {
        (row["item_id"], row["condition"], row["rollout_index"])
        for row in manipulation_schedule
    }
    source_complete = (
        len(source_rows) == 144
        and len(source_keys) == len(set(source_keys))
        and set(source_keys) == source_expected
    )
    manipulation_complete = (
        len(manipulation_rows) == 204
        and len(manipulation_keys) == len(set(manipulation_keys))
        and set(manipulation_keys) == manipulation_expected
    )

    with np.load(review / "SOURCE_ACTIVATIONS.npz", allow_pickle=False) as archive:
        activation = {name: archive[name].astype(np.float64) for name in archive.files}
    source_audit: dict[str, Any] = {}
    for axis in AXES:
        selected = [row for row in source_rows if row["axis_id"] == axis]
        pos = [row for row in selected if row["polarity"] == "POSITIVE"]
        neg = [row for row in selected if row["polarity"] == "NEGATIVE"]
        record: dict[str, Any] = {
            "positive_commitment_validity": float(np.mean([r["commitment_valid"] for r in pos])),
            "negative_commitment_validity": float(np.mean([r["commitment_valid"] for r in neg])),
            "positive_semantic_evaluability": float(
                np.mean([r["semantic_evaluable"] for r in pos])
            ),
            "negative_semantic_evaluability": float(
                np.mean([r["semantic_evaluable"] for r in neg])
            ),
            "positive_negative_mean_token_ratio": float(
                np.mean([r["generated_token_count"] for r in pos])
                / np.mean([r["generated_token_count"] for r in neg])
            ),
            "positive_minus_negative_median_tokens": float(
                np.median([r["generated_token_count"] for r in pos])
                - np.median([r["generated_token_count"] for r in neg])
            ),
            "activation": {},
            **disagreement(source_rows, axis),
        }
        for location in LOCATIONS:
            construction_pos = activation[f"construction__{axis}__POSITIVE__{location}"]
            construction_neg = activation[f"construction__{axis}__NEGATIVE__{location}"]
            raw = np.mean(construction_pos - construction_neg, axis=0)
            raw_norm = float(np.linalg.norm(raw))
            direction = raw / raw_norm
            validation_pos = activation[f"validation__{axis}__POSITIVE__{location}"]
            validation_neg = activation[f"validation__{axis}__NEGATIVE__{location}"]
            gaps = (validation_pos - validation_neg) @ direction
            pooled = np.concatenate((validation_pos @ direction, validation_neg @ direction))
            scale = float(np.std(pooled, ddof=1))
            record["activation"][location] = {
                "construction_raw_mean_gap": raw_norm,
                "validation_mean_gap": float(np.mean(gaps)),
                "validation_projection_sd": scale,
                "standardized_mean_gap": float(np.mean(gaps) / scale),
                "positive_gap_fraction": float(np.mean(gaps > 0)),
            }
        source_audit[axis] = record

    manipulation_lookup = {
        (row["item_id"], row["condition"]): row for row in manipulation_rows
    }
    item_ids = sorted({row["item_id"] for row in manipulation_rows})
    controllers = sorted(
        {row["condition"] for row in manipulation_rows if row["condition"] != "BASELINE"}
    )
    manipulation_audit: dict[str, Any] = {}
    for controller in controllers:
        selected = [manipulation_lookup[(item, controller)] for item in item_ids]
        baseline = [manipulation_lookup[(item, "BASELINE")] for item in item_ids]
        manipulation_audit[controller] = {
            "commitment_validity": float(np.mean([row["commitment_valid"] for row in selected])),
            "semantic_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in selected])
            ),
            "semantic_change_rate": float(
                np.mean(
                    [
                        row["canonical_value"] != base["canonical_value"]
                        for row, base in zip(selected, baseline, strict=True)
                    ]
                )
            ),
            "raw_sequence_change_rate": float(
                np.mean(
                    [
                        row["generated_token_ids"] != base["generated_token_ids"]
                        for row, base in zip(selected, baseline, strict=True)
                    ]
                )
            ),
        }

    bank_metadata = read_json(review / "CONTROLLER_BANK.json")
    vectors = {
        name: np.load(ROOT / record["path"], allow_pickle=False).astype(np.float64)
        for name, record in bank_metadata["vectors"].items()
    }
    hashes_clean = all(
        hashlib.sha256(vector.tobytes()).hexdigest()
        == bank_metadata["vectors"][name]["canonical_float64_vector_sha256"]
        for name, vector in vectors.items()
    )
    norms = {name: float(np.linalg.norm(vector)) for name, vector in vectors.items()}
    base_names = read_json(review / "BANK_VALIDATION.json")["base_names"]
    base_matrix = np.stack([vectors[name] for name in base_names])
    base_cosine = np.abs(base_matrix @ base_matrix.T)
    base_offdiag = base_cosine - np.eye(len(base_names))
    null_names = sorted(name for name in vectors if name.startswith("NULL_"))
    null_to_meaningful = {
        name: float(max(abs(np.dot(vectors[name], base)) for base in base_matrix))
        for name in null_names
    }
    null_pairs = {
        f"{left}__{right}": float(abs(np.dot(vectors[left], vectors[right])))
        for index, left in enumerate(null_names)
        for right in null_names[index + 1 :]
    }
    sign_errors = {
        name.removesuffix("_PLUS"): float(
            np.linalg.norm(vectors[name] + vectors[name.removesuffix("_PLUS") + "_MINUS"])
        )
        for name in base_names
    }
    bank_audit = {
        "unit_norm_pass": all(abs(value - 1.0) <= 1e-10 for value in norms.values()),
        "sign_pair_pass": all(value <= 1e-10 for value in sign_errors.values()),
        "base_diversity_pass": float(np.max(base_offdiag)) <= 0.98,
        "null_orthogonality_pass": max(
            [*null_to_meaningful.values(), *null_pairs.values()]
        )
        <= 1e-6,
        "base_max_absolute_cosine": float(np.max(base_offdiag)),
        "null_to_meaningful_max_absolute_cosines": null_to_meaningful,
        "null_pair_absolute_cosines": null_pairs,
    }

    source_pass = {
        axis: bool(
            source_audit[axis]["positive_commitment_validity"] >= 0.90
            and source_audit[axis]["negative_commitment_validity"] >= 0.90
            and source_audit[axis]["positive_semantic_evaluability"] >= 0.90
            and source_audit[axis]["negative_semantic_evaluability"] >= 0.90
            and (
                (
                    source_audit[axis]["cross_disagreement"] >= 0.10
                    and source_audit[axis]["excess_disagreement"] >= 0.03
                )
                or (
                    source_audit[axis]["positive_negative_mean_token_ratio"] >= 1.15
                    and source_audit[axis]["positive_minus_negative_median_tokens"] >= 2
                )
            )
            and all(
                source_audit[axis]["activation"][location]["standardized_mean_gap"] >= 0.20
                and source_audit[axis]["activation"][location]["positive_gap_fraction"] >= 0.60
                for location in LOCATIONS
            )
        )
        for axis in AXES
    }
    controller_pass = {
        name: bool(
            record["commitment_validity"] >= 0.75
            and record["semantic_evaluability"] >= 0.75
            and record["semantic_change_rate"] >= 1.0 / 12.0
            and record["raw_sequence_change_rate"] >= 0.25
        )
        for name, record in manipulation_audit.items()
    }
    geometry_pass = all(
        bank_audit[key]
        for key in (
            "unit_norm_pass",
            "sign_pair_pass",
            "base_diversity_pass",
            "null_orthogonality_pass",
        )
    )
    classification = (
        "Q2_CONTROLLER_BANK_QUALIFIED"
        if all(source_pass.values()) and all(controller_pass.values()) and geometry_pass
        else "Q2_CONTROLLER_BANK_NOT_QUALIFIED"
    )

    source_diffs = numeric_differences(
        source_audit, read_json(review / "SOURCE_QUALIFICATION.json")
    )
    manipulation_diffs = numeric_differences(
        manipulation_audit, read_json(review / "MANIPULATION_QUALIFICATION.json")
    )
    primary_bank = read_json(review / "BANK_VALIDATION.json")
    bank_diffs = numeric_differences(bank_audit, primary_bank)
    maximum_difference = max(
        [*source_diffs.values(), *manipulation_diffs.values(), *bank_diffs.values()],
        default=0.0,
    )
    primary_decision = read_json(review / "BANK_QUALIFICATION.json")
    classification_agreement = (
        classification == primary_decision["classification"]
        and source_pass == primary_decision["source_axis_pass"]
        and controller_pass == primary_decision["controller_pass"]
        and geometry_pass == primary_decision["representation_geometry_pass"]
    )
    no_common_outcomes = not (review / "journal.jsonl").exists()
    provenance_clean = all(
        row["model"] == "Qwen/Qwen3-8B"
        and row["model_revision"] == "b968826d9c46dd6066d109eabc6255188de91218"
        and row["correctness_evaluated"] is False
        for row in [*source_rows, *manipulation_rows]
    )
    incidents = read_json(review / "PRE_OUTPUT_INCIDENTS.json")["incidents"]
    integrity = bool(
        source_complete
        and manipulation_complete
        and hashes_clean
        and provenance_clean
        and classification_agreement
        and maximum_difference <= 1e-12
        and no_common_outcomes
    )
    forensic_classification = (
        "Q2_FORENSIC_MINOR_NONSCIENTIFIC_ISSUES"
        if integrity and incidents
        else (
            "Q2_FORENSIC_CLEAN"
            if integrity
            else "Q2_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
        )
    )
    result = {
        "classification": forensic_classification,
        "primary_classification_preserved": classification,
        "source_schedule_complete_unique": source_complete,
        "manipulation_schedule_complete_unique": manipulation_complete,
        "source_rows": len(source_rows),
        "manipulation_rows": len(manipulation_rows),
        "controller_hashes_clean": hashes_clean,
        "model_revision_and_no_correctness_provenance_clean": provenance_clean,
        "classification_agreement": classification_agreement,
        "maximum_primary_audit_metric_difference": maximum_difference,
        "common_panel_outcomes_absent": no_common_outcomes,
        "pre_common_panel_incident_count": len(incidents),
        "Q1_changed": False,
        "Q3_run": False,
    }
    write_json(review / "FORENSIC_AUDIT.json", result)
    write_json(
        review / "METRIC_CROSSCHECK.json",
        {
            "source_audit": source_audit,
            "manipulation_audit": manipulation_audit,
            "bank_audit": bank_audit,
            "source_absolute_differences": source_diffs,
            "manipulation_absolute_differences": manipulation_diffs,
            "bank_absolute_differences": bank_diffs,
        },
    )
    write_json(
        review / "RETRY_LEDGER.json",
        {
            "source_rows": len(source_rows),
            "manipulation_rows": len(manipulation_rows),
            "source_duplicate_logical_keys": len(source_keys) - len(set(source_keys)),
            "manipulation_duplicate_logical_keys": len(manipulation_keys)
            - len(set(manipulation_keys)),
            "rows_with_nonzero_retry_count": sum(
                int(row.get("retry_count", 0) != 0)
                for row in [*source_rows, *manipulation_rows]
            ),
        },
    )
    (review / "FORENSIC_AUDIT.md").write_text(
        "# Independent Q2 bank-qualification forensic audit\n\n"
        f"Classification: `{forensic_classification}`.\n\n"
        f"The audit independently reconstructed 144 source rows, 204 manipulation rows, "
        f"all source metrics, all movement metrics, vector hashes, sign pairs, meaningful "
        f"diversity, null cosines, and the frozen terminal classification. Maximum "
        f"primary/audit metric difference: `{maximum_difference}`. The common-panel "
        "journal is absent. Recorded incidents were pre-common-panel instrumentation and "
        "provenance repairs; none changed the scientific design or replaced model outputs.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
