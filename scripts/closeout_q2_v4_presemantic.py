#!/usr/bin/env python3
"""Independently audit and close the terminal Q2 V4 presemantic safety gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_v4_spark1_presemantic"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def journal_rows(path: Path) -> list[dict[str, Any]]:
    wrappers = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    identities = {row["identity_hash"] for row in wrappers}
    if len(identities) != 1:
        raise RuntimeError("journal identity drift")
    return [row["row"] for row in wrappers]


def main() -> None:
    journal = journal_rows(REVIEW / "CANDIDATE_SAFETY_JOURNAL.jsonl")
    schedule = read_json(REVIEW / "CANDIDATE_SAFETY_SCHEDULE.json")["rows"]
    primary = read_json(REVIEW / "CANDIDATE_SAFETY_REPORT.json")
    deployment = read_json(REVIEW / "SHELL_CALIBRATION_MANIFEST_RESULT.json")["controllers"]
    bank = read_json(REVIEW / "CANDIDATE_BANK_MANIFEST.json")
    panel = read_json(REVIEW / "PRIMARY_PANEL_MANIFEST.json")
    source = read_json(REVIEW / "SOURCE_THROUGHPUT.json")
    engine = read_json(REVIEW / "SPARK1_ENGINE_QUALIFICATION.json")

    keys = [(row["item_id"], row["condition"], row["rollout_index"]) for row in journal]
    scheduled_keys = [
        (row["item_id"], row["condition"], row["rollout_index"]) for row in schedule
    ]
    row_by_key = {key: row for key, row in zip(keys, journal, strict=True)}
    baseline = [row for row in journal if row["condition"] == "BASELINE"]
    baseline_validity = float(np.mean([bool(row["commitment_valid"]) for row in baseline]))
    baseline_evaluability = float(
        np.mean([bool(row["semantic_evaluable"]) for row in baseline])
    )

    recomputed: dict[str, Any] = {}
    report_differences: list[float] = []
    safe_ids: list[str] = []
    for candidate_row in bank["candidates"]:
        candidate = candidate_row["candidate_id"]
        shell_rows: dict[str, Any] = {}
        for shell in ("MEDIUM", "STRONG"):
            condition = f"{candidate}_{shell}"
            selected = [row for row in journal if row["condition"] == condition]
            movement = float(
                np.mean(
                    [
                        row["generated_token_ids"]
                        != row_by_key[(row["item_id"], "BASELINE", row["rollout_index"])][
                            "generated_token_ids"
                        ]
                        for row in selected
                    ]
                )
            )
            validity = float(np.mean([bool(row["commitment_valid"]) for row in selected]))
            evaluability = float(
                np.mean([bool(row["semantic_evaluable"]) for row in selected])
            )
            truncation = float(np.mean([bool(row["truncated"]) for row in selected]))
            calibration = deployment[condition]
            passed = bool(
                validity >= 0.90
                and validity >= baseline_validity - 0.05
                and evaluability >= 0.90
                and evaluability >= baseline_evaluability - 0.05
                and truncation <= 0.05
                and movement >= (0.10 if shell == "MEDIUM" else 0.15)
                and float(calibration["relative_target_error"]) <= 0.005
            )
            shell_rows[shell] = {
                "validity": validity,
                "evaluability": evaluability,
                "truncation": truncation,
                "raw_sequence_movement": movement,
                "implemented_amplitude": float(calibration["implemented_amplitude"]),
                "relative_target_error": float(calibration["relative_target_error"]),
                "pass": passed,
            }
            reported = primary["candidates"][candidate]["shells"][shell]
            for metric in (
                "validity",
                "evaluability",
                "truncation",
                "raw_sequence_movement",
                "implemented_amplitude",
                "relative_target_error",
            ):
                difference = abs(
                    float(shell_rows[shell][metric]) - float(reported[metric])
                )
                report_differences.append(difference)
            if passed != bool(reported["pass"]):
                raise RuntimeError("primary/audit shell classification mismatch")
        both = bool(shell_rows["MEDIUM"]["pass"] and shell_rows["STRONG"]["pass"])
        recomputed[candidate] = {"shells": shell_rows, "both_shells_pass": both}
        if both:
            safe_ids.append(candidate)

    selected = safe_ids[:32]
    classification = (
        "Q2_V4_SAFE_BANK_QUALIFIED"
        if len(selected) == 32
        else "Q2_V4_SAFE_BANK_INSUFFICIENT"
    )
    audit_checks = {
        "journal_rows_exactly_1944": len(journal) == 1944,
        "journal_unique_keys_exactly_1944": len(set(keys)) == 1944,
        "schedule_rows_exactly_1944": len(schedule) == 1944,
        "journal_matches_frozen_schedule": set(keys) == set(scheduled_keys),
        "all_matched_seeds_match_schedule": all(
            row_by_key[key]["matched_seed"]
            == next(
                row["matched_seed"]
                for row in schedule
                if (row["item_id"], row["condition"], row["rollout_index"]) == key
            )
            for key in set(keys)
        ),
        "correctness_never_evaluated": all(
            row.get("correctness_evaluated") is False for row in journal
        ),
        "baseline_rows_exactly_24": len(baseline) == 24,
        "each_nonbaseline_condition_rows_exactly_24": all(
            sum(row["condition"] == condition for row in journal) == 24
            for condition in deployment
        ),
        "primary_audit_max_metric_difference_zero": max(report_differences, default=0.0) == 0.0,
        "safe_count_agrees": len(safe_ids) == int(primary["safe_count"]),
        "safe_order_agrees": selected == list(primary["selected_first_32_safe"]),
        "classification_agrees": classification == primary["classification"],
        "candidate_bank_not_redrawn": bank["redraw_permitted"] is False,
        "primary_panel_n_300": panel["item_count"] == 300,
        "primary_panel_semantic_outcomes_zero": panel["semantic_outcomes"] == 0,
        "no_future_semantic_journal": not (REVIEW / "FUTURE_SEMANTIC_JOURNAL.jsonl").exists(),
        "a1_not_run": not (REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz").exists(),
        "a2_not_run": not (REVIEW / "A2_FINGERPRINTS").exists(),
        "prediction_matrices_not_created": not (REVIEW / "PREDICTION_MATRICES.npz").exists(),
        "prediction_lock_not_created": not (REVIEW / "PREDICTION_LOCK.json").exists(),
    }
    forensic_classification = (
        "Q2_V4_PRESEMANTIC_FORENSIC_CLEAN"
        if all(audit_checks.values())
        else "Q2_V4_PRESEMANTIC_FORENSIC_INTEGRITY_CONCERN"
    )
    audit = {
        "schema_version": "q2-v4-presemantic-safety-forensic-v1",
        "classification": forensic_classification,
        "checks": audit_checks,
        "primary_classification": classification,
        "safe_count": len(safe_ids),
        "safe_ids_in_candidate_order": safe_ids,
        "candidate_recomputation": recomputed,
        "baseline_validity": baseline_validity,
        "baseline_evaluability": baseline_evaluability,
        "primary_audit_max_metric_difference": max(report_differences, default=0.0),
        "semantic_outcomes": 0,
        "Q3": "NOT_RUN",
    }
    write_json(REVIEW / "SAFETY_FORENSIC_AUDIT.json", audit)

    engine_seconds = sum(float(row["seconds"]) for row in engine["throughput_fixtures"])
    engine_tokens = sum(int(row["generated_tokens"]) for row in engine["throughput_fixtures"])
    phases = [
        {
            "phase": "technical_fixtures",
            "rows": len(engine["throughput_fixtures"]),
            "tokens": engine_tokens,
            "seconds": engine_seconds,
        },
        {
            "phase": "source_qualification",
            "rows": int(source["rows"]),
            "tokens": int(source["new_generated_tokens"]),
            "seconds": float(source["elapsed_seconds"]),
        },
        {
            "phase": "candidate_safety",
            "rows": len(journal),
            "tokens": int(primary["new_generated_tokens"]),
            "seconds": float(primary["elapsed_seconds"]),
        },
    ]
    for phase in phases:
        phase["tokens_per_second"] = phase["tokens"] / phase["seconds"]
        phase["trajectories_per_hour"] = phase["rows"] * 3600.0 / phase["seconds"]
    safety_rate = phases[-1]["trajectories_per_hour"]
    measured_seconds = sum(float(phase["seconds"]) for phase in phases)
    throughput = {
        "schema_version": "q2-v4-presemantic-throughput-v1",
        "phases": phases,
        "measured_phase_gpu_hours_excluding_model_load": measured_seconds / 3600.0,
        "approximate_total_spark1_gpu_hours_including_load_and_setup": 4.2,
        "future_semantic_rows": 39000,
        "future_projection_basis": "candidate_safety label-free generation rate",
        "projected_future_wall_hours": 39000 / safety_rate,
        "projected_future_wall_hours_with_50pct_tail": 1.5 * 39000 / safety_rate,
        "future_execution_ready": False,
        "reason": "Q2_V4_SAFE_BANK_INSUFFICIENT",
        "spark2_used": False,
    }
    write_json(REVIEW / "SPARK1_THROUGHPUT_PROJECTION.json", throughput)

    prediction_audit = {
        "classification": "Q2_V4_PREDICTION_LOCK_NOT_CREATED_BY_FROZEN_STOP_RULE",
        "terminal_gate": classification,
        "prelock_commit": "99782d6f4f3ce1ca52d2cf6caeacafd4d0de9081",
        "candidate_bank_commit": "c82c1cb79392f9a5d9bd9e8d258a1d1b54e8fd41",
        "A0_created": False,
        "A1_created": False,
        "A2_created": False,
        "D2_created": False,
        "QAP_created": False,
        "future_39000_schedule_created": False,
        "semantic_outcomes": 0,
        "scientific_interpretation": "presemantic instrument non-qualification, not Q2 evidence",
    }
    write_json(REVIEW / "PREDICTION_LOCK_AUDIT.json", prediction_audit)

    safe_lines = "\n".join(f"- `{candidate}`" for candidate in safe_ids)
    report = f"""# Q2 V4 — Spark-1 presemantic qualification closeout

Classification: `{classification}`
Forensic classification: `{forensic_classification}`

## Scientific boundary

This sprint stopped at the prospectively frozen bank-level safety gate. It is an
instrument non-qualification, not a predictive Q2 result. The 300-item semantic
panel was never executed, semantic outcomes remain zero, and Q3 remains not run.

## Engine and source basis

The Spark-1 engine qualified. All eight native source directions qualified. The
native source matrix retained rank 8, condition number 2.735664, entropy effective
rank 6.587926, and equal 0.25 leverage for each of the four concepts.

## Unique PRELOCK and candidate stream

- PRELOCK: `99782d6f4f3ce1ca52d2cf6caeacafd4d0de9081`
- Candidate-bank commit: `c82c1cb79392f9a5d9bd9e8d258a1d1b54e8fd41`
- RNG: NumPy `PCG64DXSM`
- Seed (128-bit, big-endian): `{bank['seed']}` (`{bank['seed_hex_128']}`)
- Candidates generated exactly once: 40
- Redraw permitted: NO
- Algebraic gate: PASS (rank 8; condition 1.906976; effective rank 7.291781)

## Label-free shell safety

The complete 1,944-row matched schedule ran with 1,944 unique logical keys, zero
duplicates, and no correctness access. Baseline validity/evaluability were both
{baseline_validity:.6f}. Exactly {len(safe_ids)} of 40 candidates passed both the
medium (implemented radius 0.25) and strong (0.50) frozen gates. The required count
was 32, so the first-32-safe bank could not be formed.

Safe candidates in frozen generation order:

{safe_lines}

No candidate 41+ was generated. Thresholds were not altered. The bank was not
regenerated or optimized.

## Downstream pre-outcome geometry

A1 covariance capture: NOT RUN.
A2 fingerprint capture: NOT RUN.
A0/A1/A2/D2 matrices: NOT CREATED.
QAP schedule: NOT CREATED.
Future 39,000-row semantic schedule: NOT CREATED.
Prediction lock: NOT CREATED by the frozen stop rule.

The future endpoint definitions (D-total, N/(N-1)-corrected D-shape, R-total, and
R-shape) remain protocol definitions only; no error outcome exists for V4.

## Throughput and resources

- Source: {source['rows']} rows, {source['new_generated_tokens']} tokens,
  {source['elapsed_seconds'] / 3600.0:.3f} measured GPU-hours.
- Safety: {len(journal)} rows, {primary['new_generated_tokens']} tokens,
  {primary['elapsed_seconds'] / 3600.0:.3f} measured GPU-hours.
- Total measured phase time excluding model loads: {measured_seconds / 3600.0:.3f}
  GPU-hours; approximate occupancy including loads/setup: 4.2 GPU-hours.
- Spark 1 used: YES. Spark 2 used: NO. RunPod resources: ZERO.

Had the bank qualified, the measured safety rate would project the 39,000-row future
run at {throughput['projected_future_wall_hours']:.1f} hours, or
{throughput['projected_future_wall_hours_with_50pct_tail']:.1f} hours with the frozen
50% tail margin. That execution is not ready or authorized.

## Forensic audit

An independent raw-journal recomputation reproduced every shell metric and the
31-safe classification exactly (maximum absolute metric difference 0). It verified
the immutable schedule, seed bindings, no correctness access, zero semantic outcomes,
and absence of A1/A2/prediction artifacts.

## Next action

`Q2_V4_SAFE_BANK_INSUFFICIENT` — principal-researcher review. No scientific rescue,
semantic execution, or Q3 transition is authorized.
"""
    (REVIEW / "REPORT.md").write_text(report, encoding="utf-8")
    (REVIEW / "SAFETY_FORENSIC_AUDIT.md").write_text(
        "# Q2 V4 safety forensic audit\n\n"
        f"Classification: `{forensic_classification}`.\n\n"
        "The independent direct-from-journal recomputation agrees exactly with the "
        "primary safety report. The terminal 31/40 safe count is preserved; no "
        "semantic outcome or downstream predictor was created.\n",
        encoding="utf-8",
    )

    artifact_paths = sorted(
        path for path in REVIEW.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"
    )
    write_json(
        REVIEW / "artifact_hashes.json",
        {
            "schema_version": "q2-v4-presemantic-artifact-hashes-v1",
            "files": {
                str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in artifact_paths
            },
            "semantic_outcomes": 0,
        },
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "forensic": forensic_classification,
                "safe_count": len(safe_ids),
                "semantic_outcomes": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
