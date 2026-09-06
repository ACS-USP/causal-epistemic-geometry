#!/usr/bin/env python3
"""Aggregate-only post-hoc behavioral audit of the closed Q3.4 qualification.

The program reads the sealed private journal, private frozen references, and
private score table, but emits no prompts, source programs, model text,
references, item identifiers, or literal answer values.  Its output is a
release-safe, deterministic aggregate artifact.  It performs no model load or
inference and cannot alter the historical qualification classification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q3_REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
Q3_MANIFEST = Q3_REVIEW / "QUALIFICATION_FAMILY_MANIFEST.json"
Q3_SCHEDULE = Q3_REVIEW / "Q3_FRESH_QUALIFICATION_SCHEDULE.json"
Q3_LOCK = Q3_REVIEW / "Q3_FRESH_QUALIFICATION_EXECUTION_LOCK.json"
CRUX_MANIFEST = ROOT / "review/q2_v4_1_prediction_lock/SEMANTIC_PANEL_MANIFEST.json"

EXPECTED = {
    "journal": "2194646bcf25ff9512c5e3aaf35d4c2d0ed922f1f86ba6480709a1958dc89431",
    "scores": "c3b4ab47cf2422afb311fa978496e2abfbe5485ac76040ee3dcead2986ace533",
    "dataset": "c791e38c29d36a43fbac8ce00412e4c77d533665e0b8cb9eef8fa12fb918ac1d",
    "q3_manifest": "9a01142e4825efad36c9ede99cacf88ec6c8cc42d37f24c2ba213bb6c4a790a1",
    "q3_schedule": "edba56fc8435cdc34b6f7551fc2d1b4a6d4cc3d87fc34127a5096526d670a635",
    "q3_lock": "3a51a8d6d9fe57722f9ca740e1c5281e0645031f1641cb824c469ab4dc36635f",
    "crux_manifest": "c127cf3594e8ea849dbd038492606b3afaaac406feb4146188769c04d6691187",
}
EXPECTED_ROWS = 6000
EXPECTED_FAMILIES = 300
ROUTER = "ONLINE_ROUTED"
CHAMPION = "V4_DIRECTION_02_MEDIUM"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, *, wrapper: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(value["row"] if wrapper else value)
    return rows


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["family_id"]), str(row["condition"]), int(row["rollout_index"])


def q(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {name: math.nan for name in ("mean", "median", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def value_fingerprint(canonical: str | None) -> str | None:
    if canonical is None:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def simple_constant(canonical: str | None) -> bool:
    if canonical is None:
        return False
    try:
        tag, value = json.loads(canonical)
    except (ValueError, TypeError):
        return False
    if tag == "bool":
        return True
    if tag == "int":
        return value in {-1, 0, 1}
    if tag == "str":
        return value == ""
    if tag in {"list", "tuple", "dict", "set", "frozenset"}:
        return value == []
    return False


def value_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["semantic_evaluable"] and row["canonical_value"]]
    counts = Counter(value_fingerprint(str(row["canonical_value"])) for row in valid)
    ordered = [count for _value, count in counts.most_common()]
    by_type: dict[str, Any] = {}
    for value_type in sorted({str(row["value_type"]) for row in valid}):
        subset = [row for row in valid if str(row["value_type"]) == value_type]
        type_counts = Counter(value_fingerprint(str(row["canonical_value"])) for row in subset)
        by_type[value_type] = {
            "rows": len(subset),
            "unique_values": len(type_counts),
            "top1_share": rate(type_counts.most_common(1)[0][1], len(subset)) if subset else 0.0,
            "top5_share": rate(sum(c for _, c in type_counts.most_common(5)), len(subset)),
            "simple_constant_share": rate(
                sum(simple_constant(str(row["canonical_value"])) for row in subset), len(subset)
            ),
        }
    return {
        "evaluable_rows": len(valid),
        "unique_values": len(counts),
        "top1_share": rate(sum(ordered[:1]), len(valid)),
        "top5_share": rate(sum(ordered[:5]), len(valid)),
        "top10_share": rate(sum(ordered[:10]), len(valid)),
        "simple_constant_share": rate(
            sum(simple_constant(str(row["canonical_value"])) for row in valid), len(valid)
        ),
        "by_output_type": by_type,
        "literal_values_released": False,
    }


def stratum_table(
    policies: list[str],
    scores_by_condition: dict[str, list[dict[str, Any]]],
    family_meta: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    levels = sorted({str(row[field]) for row in family_meta.values()})
    output: dict[str, Any] = {}
    for level in levels:
        by_policy: dict[str, Any] = {}
        for policy in policies:
            subset = [
                row
                for row in scores_by_condition[policy]
                if str(family_meta[str(row["family_id"])][field]) == level
            ]
            correct = sum(bool(row["correct"]) for row in subset)
            by_policy[policy] = {
                "rows": len(subset),
                "correct": correct,
                "accuracy": rate(correct, len(subset)),
            }
        output[level] = by_policy
    return output


def ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((ast_depth(child) for child in children), default=0)


def ast_features(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    return {
        "source_chars": len(source),
        "source_lines": source.count("\n") + int(bool(source) and not source.endswith("\n")),
        "ast_nodes": len(nodes),
        "ast_depth": ast_depth(tree),
        "statements": sum(isinstance(node, ast.stmt) for node in nodes),
        "branches": sum(isinstance(node, ast.If) for node in nodes),
        "loops": sum(isinstance(node, (ast.For, ast.While)) for node in nodes),
        "calls": sum(isinstance(node, ast.Call) for node in nodes),
        "assignments": sum(
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for node in nodes
        ),
        "subscripts": sum(isinstance(node, ast.Subscript) for node in nodes),
        "arithmetic_ops": sum(isinstance(node, (ast.BinOp, ast.UnaryOp)) for node in nodes),
    }


def summarize_features(rows: list[dict[str, int]]) -> dict[str, dict[str, float]]:
    return {field: q(row[field] for row in rows) for field in rows[0]}


def split_cruxeval_prompt(prompt: str) -> tuple[str, str]:
    prefix = "Function:\n```python\n"
    middle = "\n```\n\nInput: "
    suffix = "\n\nReturn exactly one final line in this form:\n"
    start = prompt.index(prefix) + len(prefix)
    end = prompt.index(middle, start)
    input_start = end + len(middle)
    input_end = prompt.index(suffix, input_start)
    return prompt[start:end], prompt[input_start:input_end]


def reference_type(reference: str) -> str:
    try:
        value = ast.literal_eval(reference)
    except (ValueError, SyntaxError):
        return "str_fallback"
    return type(value).__name__


def distribution_comparison(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    crux = read_json(CRUX_MANIFEST)
    if len(crux["items"]) != EXPECTED_FAMILIES:
        raise RuntimeError("CRUXEVAL_PANEL_CARDINALITY_MISMATCH")
    q3_features = [ast_features(str(row["source"])) for row in dataset]
    crux_sources: list[str] = []
    crux_inputs: list[str] = []
    for row in crux["items"]:
        source, input_value = split_cruxeval_prompt(str(row["prompt"]))
        crux_sources.append(source)
        crux_inputs.append(input_value)
    crux_features = [ast_features(source) for source in crux_sources]
    q3_ops = Counter(str(operation["kind"]) for row in dataset for operation in row["operations"])
    crux_node_categories = {
        "BRANCH": ast.If,
        "LOOP": (ast.For, ast.While),
        "CALL": ast.Call,
        "MUTATION_OR_ASSIGNMENT": (ast.Assign, ast.AnnAssign, ast.AugAssign),
        "SUBSCRIPT": ast.Subscript,
        "ARITHMETIC": (ast.BinOp, ast.UnaryOp),
    }
    crux_category_counts = Counter()
    for source in crux_sources:
        nodes = list(ast.walk(ast.parse(source)))
        for label, types in crux_node_categories.items():
            crux_category_counts[label] += sum(isinstance(node, types) for node in nodes)
    return {
        "q3_fresh": {
            "families": len(dataset),
            "prompt_chars": q(len(str(row["prompt"])) for row in dataset),
            "input_chars": q(len(repr(row["input_value"])) for row in dataset),
            "code": summarize_features(q3_features),
            "declared_complexity": q(int(row["complexity"]) for row in dataset),
            "archetype_counts": dict(
                sorted(Counter(str(row["archetype"]) for row in dataset).items())
            ),
            "output_type_counts": dict(
                sorted(Counter(str(row["output_type"]) for row in dataset).items())
            ),
            "generator_operation_counts": dict(sorted(q3_ops.items())),
        },
        "closed_cruxeval": {
            "families": len(crux_sources),
            "prompt_chars": q(len(str(row["prompt"])) for row in crux["items"]),
            "input_chars": q(len(value) for value in crux_inputs),
            "code": summarize_features(crux_features),
            "output_type_counts": dict(
                sorted(
                    Counter(
                        reference_type(str(row["reference_answer"])) for row in crux["items"]
                    ).items()
                )
            ),
            "ast_operation_proxy_counts": dict(sorted(crux_category_counts.items())),
        },
        "comparability_limit": (
            "Q3 operation kinds are generator IR labels; CRUXEval operation composition is an "
            "AST-node proxy and is not a one-to-one semantic taxonomy."
        ),
        "prompt_representations_captured": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed_hashes = {
        "journal": sha256_file(args.journal),
        "scores": sha256_file(args.scores),
        "dataset": sha256_file(args.dataset),
        "q3_manifest": sha256_file(Q3_MANIFEST),
        "q3_schedule": sha256_file(Q3_SCHEDULE),
        "q3_lock": sha256_file(Q3_LOCK),
        "crux_manifest": sha256_file(CRUX_MANIFEST),
    }
    if observed_hashes != EXPECTED:
        raise RuntimeError(f"FROZEN_INPUT_HASH_MISMATCH: {observed_hashes}")

    raw = read_jsonl(args.journal, wrapper=True)
    scores = read_jsonl(args.scores)
    dataset = read_jsonl(args.dataset)
    schedule = list(read_json(Q3_SCHEDULE)["rows"])
    manifest = read_json(Q3_MANIFEST)
    lock = read_json(Q3_LOCK)
    if not (len(raw) == len(scores) == len(schedule) == EXPECTED_ROWS):
        raise RuntimeError("ROW_COUNT_MISMATCH")
    if len(dataset) != EXPECTED_FAMILIES or len(manifest["families"]) != EXPECTED_FAMILIES:
        raise RuntimeError("FAMILY_COUNT_MISMATCH")

    raw_by_key = {key(row): row for row in raw}
    score_by_key = {key(row): row for row in scores}
    schedule_by_key = {key(row): row for row in schedule}
    if (
        not (set(raw_by_key) == set(score_by_key) == set(schedule_by_key))
        or len(raw_by_key) != EXPECTED_ROWS
    ):
        raise RuntimeError("LOGICAL_KEY_COVERAGE_MISMATCH")

    family_meta = {str(row["family_id"]): row for row in dataset}
    public_meta = {str(row["family_id"]): row for row in manifest["families"]}
    if set(family_meta) != set(public_meta) or len(family_meta) != EXPECTED_FAMILIES:
        raise RuntimeError("PRIVATE_PUBLIC_FAMILY_MISMATCH")

    scores_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        scores_by_condition[str(row["condition"])].append(row)
    for row in raw:
        raw_by_condition[str(row["condition"])].append(row)

    correct_counts = {
        condition: sum(bool(row["correct"]) for row in rows)
        for condition, rows in scores_by_condition.items()
    }
    seven = sorted(
        condition
        for condition, count in correct_counts.items()
        if condition != ROUTER and count == 80
    )
    if len(seven) != 7 or CHAMPION not in seven:
        raise RuntimeError(f"EXPECTED_SEVEN_POLICY_PATTERN_MISSING: {seven}")
    correct_sets = {
        policy: {
            (str(row["family_id"]), int(row["rollout_index"]))
            for row in scores_by_condition[policy]
            if row["correct"]
        }
        for policy in seven
    }
    intersection = set.intersection(*(correct_sets[policy] for policy in seven))
    union = set.union(*(correct_sets[policy] for policy in seven))
    family_sets = {
        policy: {item for item, _rollout in values} for policy, values in correct_sets.items()
    }
    family_intersection = set.intersection(*(family_sets[policy] for policy in seven))
    family_union = set.union(*(family_sets[policy] for policy in seven))
    per_policy_family_coverage = {}
    for policy in seven:
        counts = Counter(family_id for family_id, _rollout in correct_sets[policy])
        per_policy_family_coverage[policy] = {
            "families_correct_at_least_once": len(counts),
            "families_correct_in_both_rollouts": sum(value == 2 for value in counts.values()),
            "families_correct_in_one_rollout": sum(value == 1 for value in counts.values()),
        }
    pairwise = {}
    for left, right in combinations(seven, 2):
        overlap = len(correct_sets[left] & correct_sets[right])
        combined = len(correct_sets[left] | correct_sets[right])
        pairwise[f"{left}__{right}"] = {
            "intersection": overlap,
            "union": combined,
            "jaccard": rate(overlap, combined),
        }
    row_policy_multiplicity = Counter()
    for logical in set.union(*(correct_sets[policy] for policy in seven)):
        row_policy_multiplicity[str(sum(logical in correct_sets[policy] for policy in seven))] += 1

    q3_prompt_template = (
        "Predict the exact output of this deterministic Python function.\n\n"
        "```python\n{source}```\n\n"
        "Input: {input_value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )
    prompt_reconstruction_failures = 0
    manifest_hash_failures = 0
    schedule_prompt_failures = 0
    raw_prompt_failures = 0
    for family_id, item in family_meta.items():
        frozen_input = {
            "n": item["input_value"]["n"],
            "values": item["input_value"]["values"],
            "text": item["input_value"]["text"],
        }
        expected_prompt = q3_prompt_template.format(
            source=item["source"], input_value=repr(frozen_input)
        )
        if (
            expected_prompt != item["prompt"]
            or item["source"] not in item["prompt"]
            or repr(frozen_input) not in item["prompt"]
        ):
            prompt_reconstruction_failures += 1
        prompt_hash = hashlib.sha256(str(item["prompt"]).encode()).hexdigest()
        if prompt_hash != public_meta[family_id]["prompt_sha256"]:
            manifest_hash_failures += 1
    schedule_indices = {key(row): index for index, row in enumerate(schedule)}
    for logical, planned in schedule_by_key.items():
        if planned["prompt_sha256"] != public_meta[logical[0]]["prompt_sha256"]:
            schedule_prompt_failures += 1
        row = raw_by_key[logical]
        if (
            row["prompt_sha256"] != planned["prompt_sha256"]
            or row["seed"] != planned["seed"]
            or row["schedule_index"] != schedule_indices[logical]
        ):
            raw_prompt_failures += 1

    rendered_hashes_by_family: dict[str, set[str]] = defaultdict(set)
    for row in raw:
        rendered_hashes_by_family[str(row["family_id"])].add(
            str(row["condition_metadata"]["prompt_hash"])
        )
    rendered_nondeterministic_families = sum(
        len(values) != 1 for values in rendered_hashes_by_family.values()
    )
    rendered_hash_collisions = EXPECTED_FAMILIES - len(
        {next(iter(values)) for values in rendered_hashes_by_family.values()}
    )

    status_counts = Counter(str(row["status"]) for row in scores)
    typed_same = 0
    typed_mismatch = 0
    ref_types = {family_id: str(row["reference_type"]) for family_id, row in family_meta.items()}
    for row in scores:
        if row["semantic_evaluable"]:
            if str(row["value_type"]) == ref_types[str(row["family_id"])]:
                typed_same += 1
            else:
                typed_mismatch += 1

    runtime_by_condition = {}
    for condition, rows in sorted(raw_by_condition.items()):
        runtime_by_condition[condition] = {
            "rows": len(rows),
            "tokens": q(int(row["generated_token_count"]) for row in rows),
            "elapsed_seconds": q(float(row["elapsed_seconds"]) for row in rows),
            "total_tokens": sum(int(row["generated_token_count"]) for row in rows),
            "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "repetition_stops": sum(
                row["terminal_reason"] == "EXTREME_MECHANICAL_REPETITION_V1" for row in rows
            ),
            "hard_caps": sum(row["terminal_reason"] == "max_new_tokens" for row in rows),
        }

    routed_raw = raw_by_condition[ROUTER]
    routed_scores = {key(row): row for row in scores_by_condition[ROUTER]}
    selection_counts = Counter(str(row["hook_trace"]["selected_policy"]) for row in routed_raw)
    selection_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in routed_raw:
        selection_rows[str(row["hook_trace"]["selected_policy"])].append(row)
    selected_policy_detail = {}
    matched_totals = Counter()
    for selected, selected_rows in sorted(selection_rows.items()):
        contingency = Counter()
        exact_value_agreement = 0
        both_evaluable = 0
        routed_subset_scores = []
        for routed_row in selected_rows:
            routed_key = key(routed_row)
            fixed_key = (routed_key[0], selected, routed_key[2])
            routed_score = routed_scores[routed_key]
            fixed_score = score_by_key[fixed_key]
            routed_subset_scores.append(routed_score)
            label = (
                "both_correct"
                if routed_score["correct"] and fixed_score["correct"]
                else "router_only_correct"
                if routed_score["correct"]
                else "fixed_only_correct"
                if fixed_score["correct"]
                else "neither_correct"
            )
            contingency[label] += 1
            matched_totals[label] += 1
            if routed_score["semantic_evaluable"] and fixed_score["semantic_evaluable"]:
                both_evaluable += 1
                exact_value_agreement += (
                    routed_score["canonical_value"] == fixed_score["canonical_value"]
                )
        selected_policy_detail[selected] = {
            "selected_rows": len(selected_rows),
            "selection_share": rate(len(selected_rows), len(routed_raw)),
            "fallback_rows": len(selected_rows) if selected == CHAMPION else 0,
            "correct": sum(bool(row["correct"]) for row in routed_subset_scores),
            "valid": sum(bool(row["commitment_valid"]) for row in routed_subset_scores),
            "evaluable": sum(bool(row["semantic_evaluable"]) for row in routed_subset_scores),
            "invalid_or_unevaluable": sum(
                not bool(row["semantic_evaluable"]) for row in routed_subset_scores
            ),
            "tokens": q(int(row["generated_token_count"]) for row in selected_rows),
            "elapsed_seconds": q(float(row["elapsed_seconds"]) for row in selected_rows),
            "matched_fixed_policy_correctness": dict(contingency),
            "both_evaluable_rows": both_evaluable,
            "exact_canonical_value_agreement_rows": exact_value_agreement,
            "exact_canonical_value_agreement_rate_when_both_evaluable": rate(
                exact_value_agreement, both_evaluable
            ),
        }

    router_trace_failures = sum(
        row["hook_trace"].get("selection_count") != 1
        or row["hook_trace"].get("same_policy_throughout_decode") is not True
        for row in routed_raw
    )
    bank_order = [row["policy_id"] for row in lock["policies"] if row["role"] == "BANK"]
    allowed_router_selection = set(bank_order) | {CHAMPION}
    if set(selection_counts) - allowed_router_selection:
        raise RuntimeError("UNEXPECTED_ROUTER_SELECTION")

    output = {
        "schema_version": "q3-fresh-qualification-behavioral-postmortem-v1",
        "label": "POST_HOC_DESCRIPTIVE_ONLY",
        "historical_status": "Q3_FRESH_INSTRUMENT_NOT_QUALIFIED",
        "historical_forensic_status": "Q3_FRESH_INSTRUMENT_QUALIFICATION_FORENSIC_CLEAN",
        "historical_classification_modified": False,
        "source_hashes": observed_hashes,
        "correctness_overlap": {
            "policies_with_exactly_80_of_600_correct": seven,
            "all_seven_correct_sets_identical": len(intersection) == len(union) == 80,
            "row_intersection": len(intersection),
            "row_union": len(union),
            "family_intersection": len(family_intersection),
            "family_union": len(family_union),
            "per_policy_family_coverage": per_policy_family_coverage,
            "correct_rows_by_number_of_policies": dict(sorted(row_policy_multiplicity.items())),
            "pairwise": pairwise,
            "by_output_type": stratum_table(seven, scores_by_condition, family_meta, "output_type"),
            "by_archetype": stratum_table(seven, scores_by_condition, family_meta, "archetype"),
            "by_complexity": stratum_table(seven, scores_by_condition, family_meta, "complexity"),
        },
        "answer_value_concentration": {
            condition: value_concentration(rows)
            for condition, rows in sorted(scores_by_condition.items())
        },
        "router": {
            "selection_counts": dict(sorted(selection_counts.items())),
            "fallback_definition": (
                "selected frozen champion, which is outside the eight-policy router argmax order"
            ),
            "fallback_count": selection_counts.get(CHAMPION, 0),
            "fallback_rate": rate(selection_counts.get(CHAMPION, 0), len(routed_raw)),
            "trace_invariant_failures": router_trace_failures,
            "by_selected_policy": selected_policy_detail,
            "matched_fixed_policy_overall": dict(matched_totals),
            "distinct_sampling_seeds": True,
            "individual_response_identity_required": False,
            "strong_policy": "V4_DIRECTION_35_STRONG",
        },
        "cost": {
            "by_condition": runtime_by_condition,
            "all_rows": {
                "tokens": q(int(row["generated_token_count"]) for row in raw),
                "elapsed_seconds": q(float(row["elapsed_seconds"]) for row in raw),
                "total_tokens": sum(int(row["generated_token_count"]) for row in raw),
                "summed_generation_seconds": sum(float(row["elapsed_seconds"]) for row in raw),
            },
        },
        "interface_and_evaluation": {
            "exact_prompt_reconstruction_failures": prompt_reconstruction_failures,
            "private_public_prompt_hash_failures": manifest_hash_failures,
            "schedule_prompt_hash_failures": schedule_prompt_failures,
            "journal_schedule_prompt_seed_or_index_failures": raw_prompt_failures,
            "rendered_prompt_hash_nondeterministic_families": rendered_nondeterministic_families,
            "rendered_prompt_hash_cross_family_collisions": rendered_hash_collisions,
            "router_trace_invariant_failures": router_trace_failures,
            "output_status_counts": dict(sorted(status_counts.items())),
            "evaluable_rows_with_reference_type_match": typed_same,
            "evaluable_rows_with_reference_type_mismatch": typed_mismatch,
            "terminal_rows_forced_incorrect": sum(
                row["status"] in {"REPETITION_STOP", "HARD_CAP", "RUNTIME_ERROR"}
                and not row["correct"]
                for row in scores
            ),
            "model_input_token_counts_persisted": False,
            "prompt_token_truncation_directly_auditable": False,
            "code_path_tokenizes_without_truncation_argument": True,
            "manual_answer_adjudication": False,
            "shared_scorer_error_ruled_out_by_agreement": False,
        },
        "distribution_comparison": distribution_comparison(dataset),
        "firewall": {
            "new_model_inference": 0,
            "new_semantic_trajectories": 0,
            "confirmation_qwen_access": 0,
            "reserve_qwen_access": 0,
            "raw_model_text_released": False,
            "benchmark_content_released": False,
            "spark2_used": False,
            "runpod_used": False,
            "q3_confirmatory_result": "NOT_RUN",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "POST_HOC_DESCRIPTIVE_ONLY", "output": str(args.output)}))


if __name__ == "__main__":
    main()
