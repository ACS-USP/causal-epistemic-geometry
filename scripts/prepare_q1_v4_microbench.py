#!/usr/bin/env python3
"""Prepare all model-free V4 manifests, postmortem, and review notes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.v4.character_count import (  # noqa: E402
    generate_character_count_manifest,
)
from epistemic_geometry.benchmarks.v4.geometry import generate_geometry_manifest  # noqa: E402
from epistemic_geometry.benchmarks.v4.postmortem import classify_postmortem  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    git_metadata,
    stable_digest,
)

OUT = ROOT / "review" / "q1_v4_microbench"
OLD_SMOKE = ROOT / "review" / "external_benchmark_qualification" / "cruxeval_q1_corrected_16384"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _postmortem() -> dict[str, object]:
    journal = OLD_SMOKE / "journal.jsonl"
    if not journal.exists():
        return {"status": "BLOCKED", "reason": f"missing artifact: {journal}"}
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
    diagnostics = []
    for row in rows:
        result = classify_postmortem(
            original_status=str(row["status"]),
            parsed_answer=row.get("parsed_answer"),
            reference_answer=str(row["reference_answer"]),
        )
        diagnostics.append(
            {
                "item_id": row["item_id"],
                "reference_answer": row["reference_answer"],
                "parsed_answer": row.get("parsed_answer"),
                **result.to_record(),
            }
        )
    counts = Counter(row["diagnostic_status"] for row in diagnostics)
    original = Counter(row["original_status"] for row in diagnostics)
    semantic_correct = counts["ORIGINAL_VALID_CORRECT"] + counts[
        "SEMANTIC_CORRECT_FORMAT_ERROR"
    ]
    semantic_wrong = counts["ORIGINAL_VALID_WRONG"] + counts["SEMANTIC_WRONG"]
    valid_semantic = semantic_correct + semantic_wrong
    labels = []
    if counts["SEMANTIC_CORRECT_FORMAT_ERROR"]:
        labels.append("FORMAT-SENSITIVE")
    if valid_semantic and semantic_correct / valid_semantic >= 0.9:
        labels.append("SEMANTICALLY SATURATED")
    if not labels:
        labels.append("NEITHER")
    payload = {
        "status": "COMPLETE",
        "original_counts": dict(original),
        "diagnostic_counts": dict(counts),
        "semantic_correct_count": semantic_correct,
        "semantic_wrong_count": semantic_wrong,
        "semantic_accuracy": semantic_correct / valid_semantic if valid_semantic else None,
        "diagnostic_labels": labels,
        "rows": diagnostics,
        "source_artifact": str(OLD_SMOKE),
    }
    _write_json(OUT / "cruxeval_postmortem.json", payload)
    report = [
        "# CRUXEval semantic postmortem (development-only)",
        "",
        "This diagnostic does not alter the frozen CRUXEval Q1 smoke result:",
        "11 VALID_CORRECT, 1 VALID_WRONG, 8 INVALID_FORMAT, valid completion 60%.",
        "",
        f"Original counts: {dict(original)}",
        f"Postmortem counts: {dict(counts)}",
        f"Type-aware semantic accuracy among assessed semantic outcomes: "
        f"{payload['semantic_accuracy']}",
        f"Interpretation labels: {', '.join(labels)}",
        "",
        "| item | original | parsed | diagnostic | mode |",
        "|---|---|---|---|---|",
    ]
    for row in diagnostics:
        report.append(
            f"| {row['item_id']} | {row['original_status']} | "
            f"{row['parsed_answer']!r} | {row['diagnostic_status']} | {row['comparison_mode']} |"
        )
    (OUT / "CRUXEVAL_SEMANTIC_POSTMORTEM.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return payload


def _dense_code_audit() -> None:
    report = """# Dense code failure-vector audit

## Current state

The normalized external-benchmark adapter deliberately stores one reference
answer and one final evaluator result per generated response. It does not claim
that the existing LiveCodeBench materialized file exposes a deterministic
per-test-case vector. No code-generation pilot was run in V4.

## Schema prepared

`epistemic_geometry.benchmarks.dense_code` now defines nested
`TestCaseOutcome` records and a `ProgramOutcome.failure_vector()` method. The
status vocabulary separates PASS, FAIL, COMPILE_ERROR, and RUNTIME_ERROR.
Tests remain nested under a problem/program and are not treated as independent
statistical units.

## Audit conclusion

`DENSE_CODE_VECTOR_NOT_READY`: the local repository has no verified official
per-test-case LiveCodeBench artifact/evaluator available offline. A future
model-free pilot should first materialize 3–5 problems using an official
evaluator and prove that individual test cases are accessible. Only then should
Qwen generation be considered.

Projected future pilot: 3–5 programs, one rollout each, deterministic local
execution, with GPU cost estimated only after prompt/output lengths are measured.
"""
    (OUT / "DENSE_CODE_FAILURE_VECTOR_AUDIT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    charcount = generate_character_count_manifest(seed=args.seed, per_stratum=10)
    geometry = generate_geometry_manifest()
    _write_json(OUT / "CHARCOUNT_MANIFEST.json", charcount)
    _write_json(OUT / "GEOMETRY_MANIFEST.json", geometry)
    postmortem = _postmortem()
    _dense_code_audit()
    protocol = f"""# Q1 V4 microbench protocol

This is development-only instrument reconnaissance. Historical V1–V3 and
CRUXEval outcomes are immutable. No steering, vector construction, geometry
causal test, code-generation pilot, LiveBench, LiveCodeBench generation, or
holdout access is authorized.

## Bench E

Thirty fresh procedural character-count items are frozen in
`CHARCOUNT_MANIFEST.json`: ten each in WORDLIKE_SHORT,
FRESH_PSEUDOWORD_MEDIUM, and FRESH_PSEUDOWORD_LONG. The Qwen generation cap is
fixed prospectively at 8192 tokens with the existing Qwen3-8B revision and
sampling policy. The only gate is completion plus genuine finished semantic
errors; formatting/truncation is never desired difficulty.

## Bench G

The geometry manifest contains 49 weekday-cycle prompts and 45 letter-sequence
prompts. The preset layer is zero-based block 31. Two diagnostic views are
separate: thinking prompt-boundary activation and a direct-answer positive
control with thinking disabled. The direct view is not a scientific answer
benchmark.

## Provenance

Generator hashes and item hashes are frozen before any model operation. Git
metadata at preparation: {git_metadata(ROOT)}.
Character manifest hash: `{charcount['manifest_hash']}`.
Geometry manifest hash: `{geometry['manifest_hash']}`.
CRUXEval postmortem status: `{postmortem.get('status')}`.
"""
    (OUT / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    (OUT / "LITERATURE_DESIGN_NOTES.md").write_text(
        """# Literature-inspired design notes

The V4 reset takes two methodological lessons as design constraints. Simple
exact tasks can provide a useful error regime without expensive reasoning
benchmarks, and simple conceptual domains make representation geometry
identifiable. This is an inspiration, not a reproduction, of the cited
Beaglehole et al. and Wurgaft et al. work.

Bench E therefore tests finished semantic errors with deterministic character
counting. Bench G separately tests known cyclic and sequential structure. They
must not be conflated, and neither receives steering in this micro-screen.
""",
        encoding="utf-8",
    )
    _write_json(
        OUT / "manifest.json",
        {
            "instrument": "Q1_V4_MICROBENCH",
            "source_commit": git_metadata(ROOT),
            "charcount_manifest_hash": charcount["manifest_hash"],
            "geometry_manifest_hash": geometry["manifest_hash"],
            "cruxeval_postmortem_digest": stable_digest(
                "V4-POSTMORTEM", canonical_json(postmortem)
            ),
            "steering": False,
            "holdout": False,
        },
    )
    _dense_code_audit()
    print(charcount["manifest_hash"])
    print(geometry["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
