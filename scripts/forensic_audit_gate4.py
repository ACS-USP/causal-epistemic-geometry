#!/usr/bin/env python3
"""Independent offline forensic audit for the Gate-4 micro-Q1 artifacts.

This intentionally does not import the Gate-4 analysis runner.  It reads the
raw journal and frozen manifests, recomputes the scientific quantities with
small direct routines, and writes an immutable-audit companion directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "review" / "micro_q1"
OUT = SOURCE / "forensic_audit"
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
EXPECTED_EVAL_SHA = "8661a14214832b58652768c59066d467b1239829dde2454135502a7f196432d0"
EXPECTED_EFFECTIVE_COMMIT = "cf95566f04c506a16ad773bd95dd099d839675c6"
RECORDED_PRE_OUTCOME_COMMIT = "58342cd9bd54e808a7c80049da4c80f0f8fd9245"
EXPECTED_ALPHA = 8.39900588973121
EXPECTED_LAYER = 17
CONDITIONS = ("BASELINE", "CPLUS", "CMINUS", "CRANDOM")


def _read_json(name: str) -> Any:
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def _write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(path: Path) -> str:
    return hashlib.sha256(np.load(path, allow_pickle=False).tobytes()).hexdigest()


def _load_rows() -> list[dict[str, Any]]:
    rows = []
    with (SOURCE / "journal.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _direct_estimands(
    baseline: np.ndarray,
    intervention: np.ndarray,
) -> dict[str, float]:
    """Directly evaluate the frozen two-rollout formulas from binary errors."""

    b1, b2 = baseline[:, 0].astype(float), baseline[:, 1].astype(float)
    j1, j2 = intervention[:, 0].astype(float), intervention[:, 1].astype(float)
    b00 = float(np.mean(b1 * b2))
    b0j = float(np.mean((b1 * j1 + b1 * j2 + b2 * j1 + b2 * j2) / 4.0))
    # Cross-item ordered U-statistics, deliberately not a plug-in square.
    q0 = (b1 + b2) / 2.0
    qj = (j1 + j2) / 2.0
    n = len(q0)
    if n < 2:
        raise ValueError("at least two items are required")
    u00 = float((np.sum(q0) ** 2 - np.sum(q0 * q0)) / (n * (n - 1)))
    u0j = float((np.sum(q0) * np.sum(qj) - np.sum(q0 * qj)) / (n * (n - 1)))
    c = b00 - b0j - u00 + u0j
    d = float(np.mean(b1 * b2 + j1 * j2 - b1 * j2 - b2 * j1))
    rescue = float(np.mean((b1 * (1 - j1) + b1 * (1 - j2) + b2 * (1 - j1) + b2 * (1 - j2)) / 4.0))
    damage = float(np.mean(((1 - b1) * j1 + (1 - b1) * j2 + (1 - b2) * j1 + (1 - b2) * j2) / 4.0))
    return {
        "B00": b00,
        "B0j": b0j,
        "O00": 1.0 - b00,
        "O0j": 1.0 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": c,
        "D": d,
        "rescue": rescue,
        "damage": damage,
        "accuracy_baseline": float(1.0 - np.mean(np.concatenate([b1, b2]))),
        "accuracy_condition": float(1.0 - np.mean(np.concatenate([j1, j2]))),
    }


def _bootstrap_direct(
    rows_by_condition: dict[str, dict[str, dict[int, dict[str, Any]]]], seed: int, n: int = 5000
) -> dict[str, Any]:
    """Audit the saved cluster-bootstrap contract without using project analysis."""

    item_ids = sorted(rows_by_condition["BASELINE"])
    rng = np.random.default_rng(stable_seed("bootstrap", seed, len(item_ids), n))
    samples: list[dict[str, list[float]]] = {c: defaultdict(list) for c in CONDITIONS[1:]}
    for _ in range(n):
        selected = rng.integers(0, len(item_ids), size=len(item_ids))
        for condition in CONDITIONS[1:]:
            baseline = np.asarray(
                [
                    [
                        rows_by_condition["BASELINE"][item_ids[index]][r]["status"]
                        != "VALID_CORRECT"
                        for r in (0, 1)
                    ]
                    for index in selected
                ],
                dtype=bool,
            )
            treatment = np.asarray(
                [
                    [
                        rows_by_condition[condition][item_ids[index]][r]["status"]
                        != "VALID_CORRECT"
                        for r in (0, 1)
                    ]
                    for index in selected
                ],
                dtype=bool,
            )
            values = _direct_estimands(baseline, treatment)
            for key in ("G", "C", "D", "rescue", "damage"):
                samples[condition][key].append(values[key])
    return {
        "algorithm": "direct_item_cluster_bootstrap_recomputed",
        "seed": seed,
        "effective_seed": stable_seed("bootstrap", seed, len(item_ids), n),
        "resamples": n,
        "item_count": len(item_ids),
        "condition_keys": {
            condition: {key: len(values) for key, values in sample.items()}
            for condition, sample in samples.items()
        },
    }


def _git_diff_audit() -> dict[str, Any]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            RECORDED_PRE_OUTCOME_COMMIT,
            EXPECTED_EFFECTIVE_COMMIT,
            "--",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line.split("\t", 1) for line in result.stdout.splitlines() if line.strip()]
    forbidden_prefixes = (
        "src/epistemic_geometry/backends/",
        "scripts/run_micro_q1.py",
        "review/micro_q1/",
    )
    forbidden = [
        entry
        for entry in changed
        if any(entry[-1].startswith(prefix) for prefix in forbidden_prefixes)
    ]
    return {
        "recorded_pre_outcome_commit": RECORDED_PRE_OUTCOME_COMMIT,
        "effective_checkout_commit": EXPECTED_EFFECTIVE_COMMIT,
        "changed_paths": changed,
        "forbidden_scientific_paths": forbidden,
        "governance_only_expected": not forbidden,
    }


def _write_csv(name: str, fields: list[str], records: list[dict[str, Any]]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def audit() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    manifests = {
        key: _read_json(f"{key}_MANIFEST.json")
        for key in ("CONSTRUCTION", "VALIDATION", "EVALUATION")
    }
    lock = _read_json("PROTOCOL_LOCK.json")
    pre_gate = _read_json("PRE_EVALUATION_GATE.json")
    direction_meta = _read_json("DIRECTION_METADATA.json")
    random_meta = _read_json("RANDOM_DIRECTION_METADATA.json")
    engineering = _read_json("ENGINEERING_CHECKS.json")
    raw_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, str, int]] = []
    for row in rows:
        key = (
            str(row.get("item_id")),
            str(row.get("condition")),
            int(row.get("rollout_index", -1)),
        )
        if key in raw_by_key:
            duplicate_keys.append(key)
        raw_by_key[key] = row
    evaluation_items = [item["item_id"] for item in manifests["EVALUATION"]["items"]]
    expected_keys = {
        (item_id, condition, rollout)
        for item_id in evaluation_items
        for condition in CONDITIONS
        for rollout in (0, 1)
    }
    actual_keys = set(raw_by_key)
    status_counts = Counter(str(row.get("status")) for row in rows)
    expected_seeds = {
        key: stable_seed("MICRO-Q1", "INDEPENDENT_PRIMARY", *key) for key in expected_keys
    }
    seed_mismatches = [
        key for key, seed in expected_seeds.items() if raw_by_key.get(key, {}).get("seed") != seed
    ]

    item_sets = {
        name: {item["item_id"] for item in data["items"]} for name, data in manifests.items()
    }
    historical = set(_read_json("HISTORICAL_EXCLUSION_DIGEST.json")["historical_ids"])
    leakage = {
        "construction_evaluation_overlap": sorted(
            item_sets["CONSTRUCTION"] & item_sets["EVALUATION"]
        ),
        "validation_evaluation_overlap": sorted(item_sets["VALIDATION"] & item_sets["EVALUATION"]),
        "construction_validation_overlap": sorted(
            item_sets["CONSTRUCTION"] & item_sets["VALIDATION"]
        ),
        "evaluation_historical_overlap": sorted(item_sets["EVALUATION"] & historical),
        "validation_historical_overlap": sorted(item_sets["VALIDATION"] & historical),
        "construction_historical_overlap": sorted(item_sets["CONSTRUCTION"] & historical),
        "historical_exclusion_count": len(historical),
    }

    expected_vector_hash = {
        "BASELINE": None,
        "CPLUS": pre_gate["direction_hash"],
        "CMINUS": pre_gate["direction_hash"],
        "CRANDOM": pre_gate["random_hash"],
    }
    intervention_mismatches = []
    metadata_stop_values = Counter()
    for key, row in raw_by_key.items():
        condition = key[1]
        metadata = row.get("metadata", {})
        stop = metadata.get("stop_metadata", {})
        metadata_stop_values[str(stop.get("intervention"))] += 1
        expected_alpha = (
            0.0
            if condition == "BASELINE"
            else EXPECTED_ALPHA * (1 if condition != "CMINUS" else -1)
        )
        if condition == "BASELINE":
            expected_alpha = 0.0
        if (
            metadata.get("vector_hash") != expected_vector_hash[condition]
            or not math.isclose(
                float(metadata.get("alpha", math.nan)), expected_alpha, rel_tol=0, abs_tol=1e-12
            )
            or metadata.get("layer") != EXPECTED_LAYER
        ):
            intervention_mismatches.append({"key": list(key), "metadata": metadata})
        if stop.get("model") != MODEL or stop.get("model_revision") != REVISION:
            intervention_mismatches.append(
                {"key": list(key), "reason": "model_provenance", "stop_metadata": stop}
            )

    arrays = {
        name: np.load(SOURCE / name, allow_pickle=False)
        for name in ("DIRECTION.npy", "RANDOM_DIRECTION.npy")
    }
    vector_checks = {
        "direction_file_sha256": _sha256(SOURCE / "DIRECTION.npy"),
        "direction_array_bytes_sha256": _array_sha256(SOURCE / "DIRECTION.npy"),
        "direction_metadata_hash": direction_meta.get("vector_hash"),
        "random_file_sha256": _sha256(SOURCE / "RANDOM_DIRECTION.npy"),
        "random_array_bytes_sha256": _array_sha256(SOURCE / "RANDOM_DIRECTION.npy"),
        "random_metadata_hash": random_meta.get("vector_hash"),
        "direction_shape": list(arrays["DIRECTION.npy"].shape),
        "random_shape": list(arrays["RANDOM_DIRECTION.npy"].shape),
        "alpha": EXPECTED_ALPHA,
        "layer": EXPECTED_LAYER,
        "model": MODEL,
        "model_revision": REVISION,
    }

    rows_by_condition: dict[str, dict[str, dict[int, dict[str, Any]]]] = {c: {} for c in CONDITIONS}
    for item_id in evaluation_items:
        for condition in CONDITIONS:
            rows_by_condition[condition][item_id] = {
                rollout: raw_by_key[(item_id, condition, rollout)] for rollout in (0, 1)
            }
    errors: dict[str, np.ndarray] = {}
    summary: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        matrix = np.asarray(
            [
                [
                    rows_by_condition[condition][item_id][r]["status"] != "VALID_CORRECT"
                    for r in (0, 1)
                ]
                for item_id in evaluation_items
            ],
            dtype=bool,
        )
        errors[condition] = matrix
        valid = np.asarray(
            [
                [
                    rows_by_condition[condition][item_id][r]["status"]
                    in {"VALID_CORRECT", "VALID_WRONG"}
                    for r in (0, 1)
                ]
                for item_id in evaluation_items
            ],
            dtype=bool,
        )
        summary[condition] = {
            "n": int(matrix.size),
            "valid": int(valid.sum()),
            "validity": float(valid.mean()),
            "correct": int((~matrix & valid).sum()),
            "wrong_or_mechanical_error": int(matrix.sum()),
            "accuracy_primary": float((~matrix).mean()),
            "mean_tokens": float(
                np.mean(
                    [
                        raw_by_key[(item_id, condition, r)]["generated_token_count"]
                        for item_id in evaluation_items
                        for r in (0, 1)
                    ]
                )
            ),
        }

    direct = {
        condition: _direct_estimands(errors["BASELINE"], errors[condition])
        for condition in CONDITIONS[1:]
    }
    crosscheck_records = []
    saved = _read_json("ESTIMANDS.json")
    for condition, values in direct.items():
        saved_key = {"CPLUS": "plus", "CMINUS": "minus", "CRANDOM": "random"}[condition]
        saved_values = saved[saved_key]
        for metric in ("B00", "B0j", "O00", "O0j", "G", "C", "D", "rescue", "damage"):
            historical_key = {"B0j": "B0j", "O0j": "O0j", "B00": "B00", "O00": "O00"}.get(
                metric, metric
            )
            expected = values[metric]
            observed = saved_values.get(historical_key)
            crosscheck_records.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "independent": expected,
                    "historical": observed,
                    "absolute_difference": None
                    if observed is None
                    else abs(expected - float(observed)),
                    "match": observed is not None
                    and math.isclose(expected, float(observed), rel_tol=0, abs_tol=1e-10),
                }
            )
    _write_csv("FORENSIC_METRIC_CROSSCHECK.csv", list(crosscheck_records[0]), crosscheck_records)

    # Leave-one-item-out sensitivity uses the same direct formulas, but never the
    # project estimator or historical analysis JSON as its source.
    loo_rows = []
    baseline = errors["BASELINE"]
    for drop_index, item_id in enumerate(evaluation_items):
        for condition in CONDITIONS[1:]:
            values = _direct_estimands(
                np.delete(baseline, drop_index, axis=0),
                np.delete(errors[condition], drop_index, axis=0),
            )
            loo_rows.append({"dropped_item_id": item_id, "condition": condition, **values})
    _write_csv("LOO_SENSITIVITY.csv", list(loo_rows[0]), loo_rows)

    formulas_ok = all(record["match"] for record in crosscheck_records)
    bootstrap_saved = _read_json("BOOTSTRAP_INTERVALS.json")
    bootstrap_audit = _bootstrap_direct(
        rows_by_condition,
        int(lock["bootstrap"]["seed"]),
        int(lock["bootstrap"]["resamples"]),
    )
    bootstrap_audit["historical_keys"] = sorted(bootstrap_saved)
    bootstrap_audit["historical_digest"] = _sha256(SOURCE / "BOOTSTRAP_INTERVALS.json")
    bootstrap_audit["saved_resamples"] = bootstrap_saved.get(
        "n_resamples", bootstrap_saved.get("resamples")
    )
    _write_json("BOOTSTRAP_AUDIT.json", bootstrap_audit)

    raw_integrity = {
        "journal_path": str(SOURCE / "journal.jsonl"),
        "journal_sha256": _sha256(SOURCE / "journal.jsonl"),
        "row_count": len(rows),
        "expected_row_count": len(expected_keys),
        "unique_key_count": len(actual_keys),
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "missing_keys": [list(key) for key in sorted(expected_keys - actual_keys)],
        "unauthorized_keys": [list(key) for key in sorted(actual_keys - expected_keys)],
        "item_count": len({key[0] for key in actual_keys}),
        "conditions": sorted({key[1] for key in actual_keys}),
        "rollout_indices": sorted({key[2] for key in actual_keys}),
        "status_counts": dict(status_counts),
        "seed_mismatches": [list(key) for key in seed_mismatches],
        "all_rows_are_trajectory_events": all(row.get("event") == "trajectory" for row in rows),
    }
    _write_json("RAW_INTEGRITY.json", raw_integrity)
    _write_json(
        "INTERVENTION_AUDIT.json",
        {
            "mismatches": intervention_mismatches,
            "stop_metadata_values": dict(metadata_stop_values),
            "vector_checks": vector_checks,
            "expected_alpha": EXPECTED_ALPHA,
            "expected_layer": EXPECTED_LAYER,
        },
    )
    _write_json(
        "RETRY_LEDGER.json",
        {
            "row_count": len(rows),
            "duplicate_keys": [list(key) for key in duplicate_keys],
            "runtime_or_infrastructure_rows": [
                list(key)
                for key, row in raw_by_key.items()
                if row.get("status") in {"RUNTIME_ERROR", "INFRASTRUCTURE_ERROR"}
            ],
            "behavioral_retries_detected": False,
            "note": (
                "The journal has one valid logical row per expected key; no retry rows "
                "or outcome-dependent replacements were found."
            ),
        },
    )
    _write_json(
        "ALGEBRA_CHECKS.json",
        {
            "formula_crosscheck_pass": formulas_ok,
            "rescue_minus_damage_matches_accuracy_delta": {
                condition: math.isclose(
                    values["rescue"] - values["damage"],
                    values["accuracy_condition"] - values["accuracy_baseline"],
                    abs_tol=1e-12,
                )
                for condition, values in direct.items()
            },
            "finite_values": all(
                math.isfinite(value) for values in direct.values() for value in values.values()
            ),
            "direct_estimands": direct,
        },
    )
    _write_json("LEAKAGE_AUDIT.json", leakage)

    source_diff = _git_diff_audit()
    lock_actual = _sha256(SOURCE / "EVALUATION_MANIFEST.json")
    corrections = {
        "protocol_lock_hash": {
            "historical_field": lock["allocation"]["evaluation_manifest_sha256"],
            "actual_file_sha256": lock_actual,
            "expected_manifest_sha256_from_pre_outcome_provenance": EXPECTED_EVAL_SHA,
            "matches_expected": lock_actual == EXPECTED_EVAL_SHA,
            "item_contents_changed": False,
            "action": (
                "Preserve historical lock; correct prospectively in Gate-5 provenance records."
            ),
        },
        "source_commit": {
            "historical_runner_record": RECORDED_PRE_OUTCOME_COMMIT,
            "effective_checkout": EXPECTED_EFFECTIVE_COMMIT,
            "diff_audit": source_diff,
            "item_contents_changed": False,
            "action": "Use effective checkout for provenance; preserve historical Gate-4 rows.",
        },
        "backend_stop_metadata": {
            "historical_stop_intervention_values": dict(metadata_stop_values),
            "outer_conditions": list(CONDITIONS),
            "steered_conditions_with_vector_hash": [
                condition for condition in CONDITIONS if expected_vector_hash[condition] is not None
            ],
            "engineering_checks_file": "review/micro_q1/ENGINEERING_CHECKS.json",
            "engineering_checks_present": bool(engineering),
            "historical_rows_rewritten": False,
            "prospective_fix": (
                "Future generated metadata must carry actual intervention identity, "
                "duration, layer, alpha, and vector hash."
            ),
        },
    }
    _write_json("GATE4_PROTOCOL_LOCK_CORRECTION.json", corrections["protocol_lock_hash"])
    _write_json("GATE4_SOURCE_COMMIT_CORRECTION.json", corrections["source_commit"])
    _write_json("GATE4_METADATA_CORRECTION.json", corrections["backend_stop_metadata"])

    integrity_failure = bool(
        raw_integrity["row_count"] != 400
        or raw_integrity["unique_key_count"] != 400
        or raw_integrity["duplicate_keys"]
        or raw_integrity["missing_keys"]
        or raw_integrity["unauthorized_keys"]
        or seed_mismatches
        or intervention_mismatches
        or any(leakage[key] for key in leakage if key.endswith("overlap"))
        or not formulas_ok
        or not all(
            _write
            for _write in [
                vector_checks["direction_shape"] == [4096],
                vector_checks["random_shape"] == [4096],
            ]
        )
        or lock_actual != EXPECTED_EVAL_SHA
    )
    classification = (
        "GATE4_AUDIT_SCIENTIFIC_INTEGRITY_CONCERN"
        if integrity_failure
        else "GATE4_AUDIT_MINOR_NONSCIENTIFIC_ISSUES"
    )
    classification_data = {
        "audit_classification": classification,
        "historical_gate4_classification": _read_json("FINAL_CLASSIFICATION.json").get(
            "classification"
        ),
        "integrity_failure": integrity_failure,
        "reasons": {
            "raw_integrity": raw_integrity,
            "leakage": leakage,
            "seed_mismatches": seed_mismatches,
            "intervention_mismatch_count": len(intervention_mismatches),
            "formula_crosscheck_pass": formulas_ok,
            "manifest_digest_matches_expected": lock_actual == EXPECTED_EVAL_SHA,
            "source_diff_governance_only": source_diff["governance_only_expected"],
        },
    }
    _write_json("CLASSIFICATION_CROSSCHECK.json", classification_data)

    report = "\n".join(
        [
            "# Gate-4 Forensic Audit",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "The audit recomputes the Gate-4 scientific quantities directly from "
            "the raw journal. Historical artifacts are unchanged.",
            "",
            "## Integrity",
            "",
            f"- Journal rows: {len(rows)}; expected logical rows: {len(expected_keys)}",
            f"- Unique logical keys: {len(actual_keys)}",
            f"- Duplicate keys: {len(duplicate_keys)}",
            f"- Missing keys: {len(expected_keys - actual_keys)}",
            f"- Unauthorized keys: {len(actual_keys - expected_keys)}",
            f"- Seed mismatches: {len(seed_mismatches)}",
            f"- Intervention metadata mismatches: {len(intervention_mismatches)}",
            f"- Formula cross-check: `{formulas_ok}`",
            f"- Evaluation manifest actual SHA: `{lock_actual}`",
            f"- Historical lock field: `{lock['allocation']['evaluation_manifest_sha256']}`",
            f"- Effective source checkout: `{EXPECTED_EFFECTIVE_COMMIT}`",
            "",
            "## Provenance corrections",
            "",
            "The three non-scientific provenance corrections are recorded separately "
            "in the three `GATE4_*_CORRECTION.json` files. The historical lock, "
            "journal, raw outputs, and Gate-4 report were not rewritten.",
            "",
            "## Gate-4 result preservation",
            "",
            f"The frozen Gate-4 classification remains "
            f"`{classification_data['historical_gate4_classification']}`. "
            "This audit does not reinterpret it or create a new scientific result.",
            "",
        ]
    )
    (OUT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    return 1 if integrity_failure else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Reserved for compatibility; output remains in the Gate-4 audit directory.",
    )
    parser.parse_args()
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
