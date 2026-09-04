#!/usr/bin/env python3
"""Freeze the executable Q3 fresh-instrument contract before scientific generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.q3_fresh.instrument import (
    ARCHETYPES,
    CONTAINER_BOUND,
    GENERATOR_VERSION,
    INTEGER_BOUND,
    LOOP_BOUND,
    OUTPUT_TYPES,
    PROMPT_TEMPLATE_VERSION,
    RECURSION_BOUND,
    REFERENCE_MEMORY_MB,
    REFERENCE_TIMEOUT_SECONDS,
    REFERENCE_VERSION,
    build_family,
    validate_family,
)
from epistemic_geometry.reproducibility import canonical_json

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "q3_fresh_instrument_qualification"
CANDIDATE = (
    ROOT / "review/q3_final_system_and_evaluation_supply/FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json"
)
DESIGN = ROOT / "review/q3_final_system_and_evaluation_supply/Q3_FRESH_INSTRUMENT_DESIGN_DRAFT.json"
EXPECTED_PARENT = "1bb3a16b81a2d11b5fa421c900907bad9cbcca66"
EXPECTED_CANDIDATE_SHA = "d8128e4ef4bf9459977cc46a3c9698b36c96afb8a2a388428f5daf03ac6e78f0"
EXPECTED_ROUTER_SHA = "269dc116c70b64dd47cf59340b07dbe558ec8c0f13be8410ed97017310ebad3d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-router", type=Path, required=True)
    args = parser.parse_args()
    if git("merge-base", "--is-ancestor", EXPECTED_PARENT, "HEAD") != "":
        raise AssertionError("unexpected git output")
    if sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise SystemExit("candidate-system hash mismatch")
    if sha256(args.private_router) != EXPECTED_ROUTER_SHA:
        raise SystemExit("private-router hash mismatch")

    fixture_indices = list(range(192))
    fixture_reports = []
    for index in fixture_indices:
        family = build_family("excluded-fixture", index, 0x5133)
        report = validate_family(family)
        fixture_reports.append(
            {
                "fixture_id": f"Q3-EXCLUDED-{index:04d}",
                "archetype": family.archetype,
                "complexity": family.complexity,
                "output_type": family.output_type,
                **report,
            }
        )
    fixture_report = {
        "schema_version": "q3-fresh-instrument-excluded-fixtures-v1",
        "status": "PASS",
        "scientific_population_eligible": False,
        "fixture_count": len(fixture_reports),
        "covered_archetypes": sorted({row["archetype"] for row in fixture_reports}),
        "covered_output_types": sorted({row["output_type"] for row in fixture_reports}),
        "all_dual_evaluator_agreement": all(
            row["dual_evaluator_agreement"] for row in fixture_reports
        ),
        "all_reference_repeat_determinism": all(
            row["reference_repeat_determinism"] for row in fixture_reports
        ),
        "all_parser_reference_roundtrip": all(
            row["parser_reference_roundtrip"] for row in fixture_reports
        ),
        "fixtures": fixture_reports,
    }
    REVIEW.mkdir(parents=True, exist_ok=True)
    fixture_path = REVIEW / "EXCLUDED_ENGINEERING_FIXTURE_REPORT.json"
    write_json(fixture_path, fixture_report)

    module_path = ROOT / "src/epistemic_geometry/benchmarks/q3_fresh/instrument.py"
    script_path = Path(__file__).resolve()
    evaluator_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    prelock = {
        "schema_version": "q3-fresh-instrument-generator-prelock-v1",
        "status": "FROZEN_BEFORE_SCIENTIFIC_GENERATION",
        "evidence_class": "INSTRUMENT_CONSTRUCTION_AND_QUALIFICATION_ONLY",
        "source_parent": EXPECTED_PARENT,
        "scientific_generation_started": False,
        "experimental_seeds_derived": False,
        "candidate_system": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": EXPECTED_CANDIDATE_SHA,
            "private_router_sha256": EXPECTED_ROUTER_SHA,
            "router_refit_or_modification": False,
        },
        "code_identity": {
            "generator_version": GENERATOR_VERSION,
            "reference_version": REFERENCE_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "generator_module": {
                "path": str(module_path.relative_to(ROOT)),
                "sha256": sha256(module_path),
            },
            "prelock_builder": {
                "path": str(script_path.relative_to(ROOT)),
                "sha256": sha256(script_path),
            },
            "qualification_parser": {
                "path": str(evaluator_path.relative_to(ROOT)),
                "version": "external-semantic-v3",
                "sha256": sha256(evaluator_path),
            },
            "excluded_fixture_report": {
                "path": str(fixture_path.relative_to(ROOT)),
                "sha256": sha256(fixture_path),
            },
        },
        "population": {
            "estimand": (
                "utility over independently generated families from this frozen "
                "restricted-Python generator law"
            ),
            "family": "one normalized source-program skeleton with one allocated typed input",
            "nested_not_independent": (
                "additional inputs, renamed variables, formatting changes, and "
                "literal-only changes to one skeleton"
            ),
            "allocation": {"qualification": 300, "confirmation": 1000, "reserve": 300},
            "namespace_split_before_generation": True,
            "candidate_order": "increasing candidate_index within each namespace",
            "maximum_candidates_per_namespace": 100_000,
            "archetype_cycle": list(ARCHETYPES),
            "complexity_cycle": [6, 8, 10, 12],
            "output_type_cycle": list(OUTPUT_TYPES),
            "operation_fill": (
                "required archetype operations plus uniform deterministic draws from "
                "eight operation kinds"
            ),
            "input_law": {
                "n": "discrete uniform integers [-50,50]",
                "values_length": "discrete uniform integers [3,8]",
                "values": "iid discrete uniform integers [-40,40]",
                "text": "uniform over the frozen eight-token vocabulary",
            },
        },
        "restricted_language": {
            "productions": [
                "typed scalar/container assignment",
                "bounded for and while loops",
                "nested bounded loops",
                "if/else",
                "integer arithmetic with Python floor/modulo semantics",
                "list aliasing and bounded indexed mutation",
                "bounded string slicing/reversal",
                "string-keyed dictionary updates",
                "bounded digit recursion represented by a depth-limited while production",
            ],
            "forbidden": [
                "imports",
                "attributes",
                "filesystem",
                "network",
                "reflection",
                "dynamic code",
                "randomness",
                "ambient process/environment",
                "unbounded loops",
            ],
            "bounds": {
                "integer_absolute_modulus": INTEGER_BOUND,
                "container_elements": CONTAINER_BOUND,
                "loop_iterations_total": LOOP_BOUND,
                "recursion_depth": RECURSION_BOUND,
                "reference_timeout_seconds": REFERENCE_TIMEOUT_SECONDS,
                "reference_memory_mb": REFERENCE_MEMORY_MB,
            },
        },
        "family_acceptance": {
            "exact_canonical_skeleton_overlap": 0,
            "canonicalization": (
                "alpha-normalized identifiers; all literal values removed; operation "
                "kind/order/variant and output type retained"
            ),
            "exact_normalized_token_overlap": 0,
            "exact_behavioral_signature_overlap": 0,
            "near_duplicate_rule": (
                "reject when same output type and operation-token multiset Jaccard "
                ">=0.90 and ordered operation-token similarity >=0.90"
            ),
            "numeric_near_duplicate_rule": (
                "input-only similarity never creates a new family; skeleton uniqueness "
                "is mandatory regardless of numeric distance"
            ),
            "near_duplicate_rate_gate": 0.01,
            "prior_content_overlap": (
                "reject exact source, prompt, or canonical skeleton hashes found in "
                "tracked prior project artifacts or excluded fixtures"
            ),
            "cross_split_checks": (
                "all exact and near-duplicate checks use the union of earlier accepted splits"
            ),
            "structural_rejections_logged": True,
            "no_manual_replacement": True,
        },
        "reference_contract": {
            "path_a": "direct IR interpreter; never compiles source",
            "path_b": (
                "isolated pinned CPython -I -S worker after independent AST whitelist validation"
            ),
            "agreement": "exact type and repr, twice",
            "worker": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable_sha256": sha256(Path(sys.executable).resolve()),
                "network": "blocked by absence of allowed import/attribute nodes plus audit hook",
                "filesystem": "empty non-writable cwd plus audit hook",
                "resource_policy": (
                    "CPU/file/fd rlimits; Linux AS cap; macOS 2ms fail-closed RSS watchdog"
                ),
            },
            "typed_serialization": (
                "Python repr for int/bool/str/list/tuple/string-keyed dict; exact "
                "ast.literal_eval roundtrip"
            ),
        },
        "seed_derivation": {
            "when": "only after the containing prelock commit is published",
            "formula": (
                "first 63 bits of SHA256('Q3-FRESH-V1' || prelock_file_sha256 || "
                "containing_commit || namespace)"
            ),
            "namespaces": ["qualification", "confirmation", "reserve"],
            "one_stream_per_namespace": True,
        },
        "qualification": {
            "families": 300,
            "rollouts": 2,
            "conditions": "exact eight bank policies, external champion, online routed system",
            "logical_generations": 6000,
            "exact_policy_sharing": False,
            "sharing_rationale": (
                "qualification conditions retain distinct frozen seeds and independently "
                "exercise the online routed engine; no post-hoc reuse"
            ),
            "pooled_row_aggregation": (
                "validity/evaluability/repetition rates pool 600 rows per condition"
            ),
            "champion_accuracy": "pooled correctness over 600 champion rows",
            "oracle_headroom": (
                "mean over 300 families of max across bank policies of two-rollout mean "
                "correctness minus champion two-rollout mean correctness"
            ),
            "gates": {
                "dual_evaluator_agreement": 1.0,
                "reference_repeat_determinism": 1.0,
                "parser_reference_roundtrip": 1.0,
                "cross_split_family_or_skeleton_collision": 0,
                "near_duplicate_rate_max": 0.01,
                "router_commitment_validity_min": 0.95,
                "champion_commitment_validity_min": 0.95,
                "router_semantic_evaluability_min": 0.95,
                "champion_semantic_evaluability_min": 0.95,
                "champion_accuracy_range": [0.25, 0.90],
                "frozen_bank_oracle_headroom_min": 0.05,
                "terminal_repetition_rate_max_per_condition": 0.10,
            },
            "routed_minus_champion_gain_is_gate": False,
            "routed_correctness_not_required_for_qualification": True,
        },
        "generation_contract": {
            "model": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "layer": 27,
            "dtype": "BF16",
            "attention": "SDPA",
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": 4096,
            "termination": "EXTREME_MECHANICAL_REPETITION_V1",
            "router_rng": "feature transform and policy selection consume no sampling RNG",
        },
        "confirmation_firewall": {
            "confirmation_qwen_forwards": 0,
            "reserve_qwen_forwards": 0,
            "confirmation_not_authorized": True,
        },
    }
    write_json(REVIEW / "Q3_FRESH_INSTRUMENT_GENERATOR_PRELOCK.json", prelock)
    print(canonical_json({"status": prelock["status"], "fixture_count": len(fixture_reports)}))


if __name__ == "__main__":
    main()
