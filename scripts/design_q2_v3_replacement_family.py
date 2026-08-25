#!/usr/bin/env python3
"""Outcome-free static inventory for the Q2 V3 replacement-family sprint.

The input JSONL contains benchmark references, but this module deliberately
projects every record onto an allowlist that excludes the ``output`` field.
No model result, correctness label, or semantic metric is imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tokenize
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

STATIC_FIELDS = (
    "id",
    "official_index",
    "code",
    "input",
    "dataset_repo",
    "dataset_revision",
)

TYPE_CALLS = {
    "bool",
    "bytes",
    "decode",
    "dict",
    "encode",
    "float",
    "int",
    "isinstance",
    "list",
    "set",
    "str",
    "tuple",
    "type",
}
MUTATING_CALLS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}
INVARIANT_CALLS = {
    "count",
    "dict",
    "items",
    "keys",
    "len",
    "list",
    "reverse",
    "reversed",
    "set",
    "sort",
    "sorted",
    "tuple",
    "values",
}
BUILTIN_NAMES = {
    "all",
    "any",
    "bool",
    "bytes",
    "dict",
    "enumerate",
    "filter",
    "float",
    "id",
    "input",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_static_records(path: Path) -> list[dict[str, Any]]:
    """Read only task text and static metadata; benchmark outputs are ignored."""

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        records.append({field: raw[field] for field in STATIC_FIELDS})
    return records


def read_item_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload["item_ids"])


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _has_nested_call(tree: ast.AST) -> bool:
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if any(isinstance(child, ast.Call) for child in ast.walk(call) if child is not call):
            return True
    return False


def _has_scope_binding_feature(tree: ast.AST) -> bool:
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    outer_bound: set[str] = set()
    if functions:
        outer_bound.update(argument.arg for argument in functions[0].args.args)
    outer_bound.update(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    builtin_shadow = bool(outer_bound & BUILTIN_NAMES)
    nested_scope = any(
        isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        for node in ast.walk(tree)
    )
    comprehension_targets = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        for generator in node.generators
        for target in ast.walk(generator.target)
        if isinstance(target, ast.Name)
    }
    return builtin_shadow or (nested_scope and bool(comprehension_targets & outer_bound))


def static_features(record: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(record["code"])
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = [call_name(node) for node in calls]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function_name = functions[0].name if functions else None
    recursion = bool(
        function_name
        and any(
            isinstance(node.func, ast.Name) and node.func.id == function_name
            for node in calls
        )
    )
    token_count = sum(
        1
        for token in tokenize.tokenize(BytesIO(record["code"].encode("utf-8")).readline)
        if token.type
        not in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
        }
    )
    return {
        "item_id": record["id"],
        "official_index": record["official_index"],
        "code_bytes": len(record["code"].encode("utf-8")),
        "input_bytes": len(record["input"].encode("utf-8")),
        "code_lines": len(record["code"].splitlines()),
        "python_tokens": token_count,
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "function_calls": len(calls),
        "branches": sum(
            isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)
        ),
        "loops": sum(
            isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree)
        ),
        "INTERMEDIATE_DATAFLOW_COMPOSITION": _has_nested_call(tree),
        "SCOPE_BINDING_SHADOWING": _has_scope_binding_feature(tree),
        "SHORT_CIRCUIT_EVALUATION": (
            any(isinstance(node, ast.BoolOp) for node in ast.walk(tree))
            or any(name in {"all", "any"} for name in call_names)
        ),
        "TYPE_COERCION_SEMANTICS": any(name in TYPE_CALLS for name in call_names),
        "EXCEPTION_ERROR_PATH": any(
            isinstance(node, (ast.Try, ast.Raise, ast.Assert)) for node in ast.walk(tree)
        ),
        "RECURSION_BASE_CASE": recursion,
        "ORDERING_SIDE_EFFECT_DEPENDENCE": sum(name in MUTATING_CALLS for name in call_names)
        >= 2,
        "DATA_STRUCTURE_INVARIANT": sum(name in INVARIANT_CALLS for name in call_names) >= 2,
    }


def summarize_candidates(features: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(features)
    candidates = [
        "INTERMEDIATE_DATAFLOW_COMPOSITION",
        "SCOPE_BINDING_SHADOWING",
        "SHORT_CIRCUIT_EVALUATION",
        "TYPE_COERCION_SEMANTICS",
        "EXCEPTION_ERROR_PATH",
        "RECURSION_BASE_CASE",
        "ORDERING_SIDE_EFFECT_DEPENDENCE",
        "DATA_STRUCTURE_INVARIANT",
    ]
    return {
        candidate: {
            "static_positive_count": sum(bool(row[candidate]) for row in rows),
            "static_negative_count": sum(not bool(row[candidate]) for row in rows),
            "positive_item_ids": [row["item_id"] for row in rows if row[candidate]],
        }
        for candidate in candidates
    }


def build_inventory(repo: Path) -> dict[str, Any]:
    review = repo / "review"
    records = read_static_records(
        review / "q2_v3_provenance_reconciliation" / "OFFICIAL_SOURCE_RECORDS.jsonl"
    )
    primary = read_item_ids(
        review / "q2_v3_amendment1_freeze" / "PRIMARY_PANEL_MANIFEST.json"
    )
    construction = read_item_ids(
        review / "q2_v3_amendment1_freeze" / "SOURCE_CONSTRUCTION_MANIFEST.json"
    )
    validation = read_item_ids(
        review / "q2_v3_amendment1_freeze" / "SOURCE_VALIDATION_MANIFEST.json"
    )
    excluded = primary | construction | validation
    available_records = sorted(
        (record for record in records if record["id"] not in excluded),
        key=lambda record: record["official_index"],
    )
    features = [static_features(record) for record in available_records]
    static_projection = [
        {field: record[field] for field in STATIC_FIELDS} for record in available_records
    ]
    return {
        "schema_version": "q2-v3-replacement-family-static-inventory-v1",
        "analysis_mode": "CPU_ONLY_DESIGN_ONLY",
        "input_records": len(records),
        "excluded_primary_panel_ids": len(primary),
        "excluded_prior_source_construction_ids": len(construction),
        "excluded_prior_source_validation_ids": len(validation),
        "available_disjoint_records": len(available_records),
        "available_item_ids": [record["id"] for record in available_records],
        "available_static_projection_sha256": sha256_json(static_projection),
        "output_field_used": False,
        "model_behavior_used": False,
        "correctness_used": False,
        "candidate_counts": summarize_candidates(features),
        "static_feature_rows": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    inventory = build_inventory(args.repo)
    print(json.dumps(inventory, indent=None if args.compact else 2, sort_keys=True))


if __name__ == "__main__":
    main()
