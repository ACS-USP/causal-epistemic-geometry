#!/usr/bin/env python3
"""Run the model-free Q0 external-benchmark adapter audit.

This command is intentionally offline.  It validates checked-in normalized
fixtures and adapter/evaluator behavior; it never imports ``datasets`` and
never resolves a remote dataset.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.adapters import (  # noqa: E402
    adapter_for,
    candidate_specs,
)
from epistemic_geometry.benchmarks.external.base import (  # noqa: E402
    ExternalStatus,
    score_external_response,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    git_metadata,
    stable_digest,
)

FIXTURES = {
    "RE2-Bench": ROOT / "examples/external_benchmark_fixtures/re2bench_output.jsonl",
    "LiveCodeBench": ROOT / "examples/external_benchmark_fixtures/livecodebench_test_output.jsonl",
    "CRUXEval": ROOT / "examples/external_benchmark_fixtures/cruxeval_output.jsonl",
    "LiveBench": ROOT / "examples/external_benchmark_fixtures/livebench_objective.jsonl",
}


def main() -> int:
    output = ROOT / "review/external_benchmark_qualification"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    fixture_digests: dict[str, str] = {}
    for spec in candidate_specs():
        row: dict[str, object] = {
            "candidate": spec.name,
            "subtask": spec.subtask,
            "official_source": spec.official_source,
            "fixture_path": str(FIXTURES[spec.name].relative_to(ROOT)),
            "q0_status": "FAIL",
            "n_fixture_items": 0,
            "objective_evaluator": spec.objective_evaluation,
            "reason": "",
        }
        try:
            adapter = adapter_for(spec.name)
            items = adapter.load_items(FIXTURES[spec.name])
            validation = adapter.validate(items)
            fixture_digests[spec.name] = str(validation["item_digest"])
            good = score_external_response(
                items[0], f"working\nFINAL: {items[0].reference_answer}", rollout_seed=0
            )
            wrong_answer = (
                "999" if items[0].evaluator != "exact" else "definitely-not-the-reference"
            )
            wrong = score_external_response(
                items[0], f"FINAL: {wrong_answer}", rollout_seed=0
            )
            malformed = score_external_response(
                items[0], "I think the answer is unclear", rollout_seed=0
            )
            truncated = score_external_response(items[0], "<think>unfinished", rollout_seed=0)
            if good.status != ExternalStatus.VALID_CORRECT:
                raise AssertionError(f"correct fixture did not score correctly: {good.status}")
            if wrong.status != ExternalStatus.VALID_WRONG:
                raise AssertionError(f"wrong fixture did not remain semantic wrong: {wrong.status}")
            if malformed.status != ExternalStatus.INVALID_FORMAT:
                raise AssertionError(f"malformed fixture status: {malformed.status}")
            if truncated.status != ExternalStatus.TRUNCATED_THINKING:
                raise AssertionError(f"truncated fixture status: {truncated.status}")
            row.update(
                {
                    "q0_status": "PASS" if spec.name != "RE2-Bench" else "BLOCKED_OFFICIAL_SOURCE",
                    "n_fixture_items": len(items),
                    "reason": (
                        "normalized contract/evaluator passes; official executable artifact "
                        "unresolved"
                        if spec.name == "RE2-Bench"
                        else "normalized contract, deterministic evaluator, and failure "
                        "taxonomy pass"
                    ),
                }
            )
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
            row["reason"] = str(exc)
        audit_rows.append(row)

    (output / "PROTOCOL.md").write_text(
        "# External benchmark qualification — Q0\n\n"
        "Q0 is model-free and offline. It validates normalized schemas, stable IDs,\n"
        "deterministic prompt records, objective evaluators, and separate statuses\n"
        "for valid answers, invalid format, truncated thinking, and runtime errors.\n\n"
        "No model, dataset, steering direction, activation, or holdout item was used.\n"
        "Fixtures are software tests, not benchmark results. Real dataset materialization\n"
        "must happen on the RunPod execution host and record an immutable source revision.\n\n"
        "Q1 smoke: at most 20 new items × 1 seed per practical candidate.\n"
        "Q2 qualification: only survivors, 50 new items × 2 independent seeds.\n"
        "No steering pilot is authorized in this campaign.\n",
        encoding="utf-8",
    )
    (output / "BENCHMARK_ADAPTER_AUDIT.md").write_text(
        "# Benchmark adapter audit\n\n"
        "The fixture audit is a model-free software gate. It does not establish model\n"
        "completion or accuracy. Official source availability is tracked separately.\n\n"
        + "\n".join(
            f"- **{row['candidate']} / {row['subtask']}**: {row['q0_status']} — {row['reason']}"
            for row in audit_rows
        )
        + "\n\nOfficial references:\n"
        + "\n".join(f"- {spec.name}: {spec.official_source}" for spec in candidate_specs())
        + "\n",
        encoding="utf-8",
    )
    with (output / "Q0_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    manifest = {
        "campaign": "external-benchmark-qualification",
        "stage": "Q0",
        "offline": True,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": git_metadata(ROOT),
        "fixtures": fixture_digests,
        "adapter_rows": audit_rows,
        "artifact_hash": stable_digest(
            "Q0-EXTERNAL", canonical_json(audit_rows), canonical_json(fixture_digests)
        ),
        "model_inference": False,
        "steering": False,
        "holdout_accessed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "FINAL_REPORT.md").write_text(
        "# External benchmark qualification — Q0\n\n"
        "Q0 completed locally without model or dataset downloads. This is not a\n"
        "qualification result. Q1 may proceed only on the RunPod host after each\n"
        "candidate's official data/evaluator is materialized and hashed.\n\n"
        + "\n".join(f"- {row['candidate']}: **{row['q0_status']}**" for row in audit_rows)
        + "\n\nNo steering was run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": audit_rows}, indent=2))
    return (
        0
        if all(row["q0_status"] in {"PASS", "BLOCKED_OFFICIAL_SOURCE"} for row in audit_rows)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
