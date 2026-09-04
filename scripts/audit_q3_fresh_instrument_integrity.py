#!/usr/bin/env python3
"""Independent, source-only structural audit of the sealed Q3 fresh instrument.

The reader deliberately decodes only allow-listed top-level JSON fields.  It
never constructs prompts, references, model outputs, or correctness fields and
never executes generated programs.
"""

from __future__ import annotations

import argparse
import ast
import collections
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_fresh_instrument_qualification"
PRECHECK = REVIEW / "Q3_EXTERNAL_REVIEW_INTEGRITY_AUDIT_PRECHECK.json"
INSTRUMENT = ROOT / "src/epistemic_geometry/benchmarks/q3_fresh/instrument.py"
EXPECTED_PRECHECK_STATUS = "FROZEN_BEFORE_PRIVATE_STRUCTURAL_AUDIT"
ALLOWED_FIELDS = {
    "namespace",
    "family_id",
    "source",
    "canonical_skeleton_sha256",
    "normalized_token_sha256",
    "operations",
    "output_type",
}
FORBIDDEN_FIELDS = {"reference_repr", "prompt", "model_output", "correctness"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skip_ws(line: str, offset: int) -> int:
    while offset < len(line) and line[offset].isspace():
        offset += 1
    return offset


def _value_end(line: str, offset: int) -> int:
    """Return the end of one JSON value without decoding it."""

    offset = _skip_ws(line, offset)
    if offset >= len(line):
        raise ValueError("missing JSON value")
    if line[offset] == '"':
        index, escaped = offset + 1, False
        while index < len(line):
            char = line[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return index + 1
            index += 1
        raise ValueError("unterminated JSON string")
    if line[offset] in "[{":
        opening = line[offset]
        closing = "]" if opening == "[" else "}"
        stack = [closing]
        index, in_string, escaped = offset + 1, False, False
        while index < len(line):
            char = line[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                stack.append("]" if char == "[" else "}")
            elif char in "]}":
                if not stack or char != stack.pop():
                    raise ValueError("unbalanced JSON container")
                if not stack:
                    return index + 1
            index += 1
        raise ValueError("unterminated JSON container")
    index = offset
    while index < len(line) and line[index] not in ",}":
        index += 1
    return index


def allowed_top_level_fields(line: str) -> tuple[dict[str, Any], set[str]]:
    """Decode only allow-listed fields and report names skipped as opaque spans."""

    decoder = json.JSONDecoder()
    offset = _skip_ws(line, 0)
    if offset >= len(line) or line[offset] != "{":
        raise ValueError("row is not a JSON object")
    offset += 1
    selected: dict[str, Any] = {}
    skipped: set[str] = set()
    while True:
        offset = _skip_ws(line, offset)
        if offset < len(line) and line[offset] == "}":
            break
        key, offset = decoder.raw_decode(line, offset)
        if not isinstance(key, str):
            raise ValueError("non-string top-level key")
        offset = _skip_ws(line, offset)
        if offset >= len(line) or line[offset] != ":":
            raise ValueError("missing JSON colon")
        value_start = _skip_ws(line, offset + 1)
        value_end = _value_end(line, value_start)
        if key in ALLOWED_FIELDS:
            selected[key] = json.loads(line[value_start:value_end])
        else:
            skipped.add(key)
        offset = _skip_ws(line, value_end)
        if offset < len(line) and line[offset] == ",":
            offset += 1
            continue
        if offset < len(line) and line[offset] == "}":
            break
        raise ValueError("malformed top-level object")
    missing = ALLOWED_FIELDS - selected.keys()
    if missing:
        raise ValueError(f"missing allow-listed fields: {sorted(missing)}")
    return selected, skipped


class ScopeNormalizer(ast.NodeTransformer):
    """Normalize bound identifiers while preserving scope and use-definition links."""

    def __init__(self) -> None:
        self.scopes: list[dict[str, str]] = [{}]
        self.next_binding = 0
        self.next_function = 0

    def _lookup(self, name: str) -> str | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _bind(self, name: str) -> str:
        current = self.scopes[-1]
        if name not in current:
            current[name] = f"B{self.next_binding}"
            self.next_binding += 1
        return current[name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        node.name = f"F{self.next_function}"
        self.next_function += 1
        self.scopes.append({})
        node.args = self.visit(node.args)
        node.body = [self.visit(child) for child in node.body]
        node.decorator_list = [self.visit(child) for child in node.decorator_list]
        self.scopes.pop()
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        node.arg = self._bind(node.arg)
        node.annotation = self.visit(node.annotation) if node.annotation else None
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            node.id = self._lookup(node.id) or self._bind(node.id)
        else:
            node.id = self._lookup(node.id) or node.id
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:  # noqa: N802
        self.scopes.append({})
        for generator in node.generators:
            generator.iter = self.visit(generator.iter)
            generator.target = self.visit(generator.target)
            generator.ifs = [self.visit(child) for child in generator.ifs]
        node.elt = self.visit(node.elt)
        self.scopes.pop()
        return node


def ast_object(value: Any) -> Any:
    if isinstance(value, ast.Constant):
        return {"node": "Constant", "literal_type": type(value.value).__name__}
    if isinstance(value, ast.AST):
        result: dict[str, Any] = {"node": type(value).__name__}
        for field, child in ast.iter_fields(value):
            if field in {"type_comment"}:
                continue
            result[field] = ast_object(child)
        return result
    if isinstance(value, list):
        return [ast_object(child) for child in value]
    return value


def normalized_ast(source: str) -> Any:
    tree = ast.parse(source, mode="exec")
    normalized = ScopeNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast_object(normalized)


def ast_fingerprint(source: str) -> tuple[str, tuple[str, ...], collections.Counter[str]]:
    parsed = ast.parse(source, mode="exec")
    normalized = ScopeNormalizer().visit(parsed)
    ast.fix_missing_locations(normalized)
    obj = ast_object(normalized)
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    nodes = tuple(type(node).__name__ for node in ast.walk(parsed))
    features = collections.Counter(nodes)
    return sha256_bytes(encoded), nodes, features


def multiset_jaccard(left: collections.Counter[str], right: collections.Counter[str]) -> float:
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection / union if union else 1.0


def frozen_identity_functions() -> tuple[Any, Any]:
    """Load only the two audited pure definitions from the hash-pinned source."""

    tree = ast.parse(INSTRUMENT.read_text(encoding="utf-8"), mode="exec")
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"render_program", "canonical_skeleton"}
    ]
    if {node.name for node in wanted} != {"render_program", "canonical_skeleton"}:
        raise RuntimeError("frozen identity functions not found")
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "GENERATOR_VERSION": "q3-restricted-python-generator-v1",
        "INTEGER_BOUND": 1_000_000_000,
        "RECURSION_BOUND": 32,
    }
    exec(compile(module, str(INSTRUMENT), "exec"), namespace, namespace)  # noqa: S102
    return namespace["render_program"], namespace["canonical_skeleton"]


def synthetic_tests() -> dict[str, bool]:
    render_program, canonical_skeleton = frozen_identity_functions()
    base = ({"kind": "AFFINE", "a": 3, "b": 5, "c": 7, "variant": 0},)
    variant = ({**base[0], "variant": 1},)
    structural_option = ({**base[0], "kind": "BRANCH"},)
    source_a, source_b = render_program(base, "int"), render_program(variant, "int")
    source_structural = render_program(structural_option, "int")
    fp_a = ast_fingerprint(source_a)[0]
    fp_b = ast_fingerprint(source_b)[0]
    renamed = source_a.replace("acc", "state").replace("value", "entry")
    literal_changed = source_a.replace(" * 3 + 5 + ", " * 13 + 29 + ")
    operator_changed = source_a.replace(" * 3 + 5 + ", " * 3 - 5 + ")
    return {
        "affine_variant_source_identical": source_a == source_b,
        "affine_variant_current_identity_differs": (
            canonical_skeleton(base, "int") != canonical_skeleton(variant, "int")
        ),
        "affine_variant_strict_ast_equal": fp_a == fp_b,
        "structure_option_current_identity_differs": (
            canonical_skeleton(base, "int") != canonical_skeleton(structural_option, "int")
        ),
        "structure_option_strict_ast_differs": fp_a != ast_fingerprint(source_structural)[0],
        "consistent_bound_rename_equal": fp_a == ast_fingerprint(renamed)[0],
        "literal_only_change_equal": fp_a == ast_fingerprint(literal_changed)[0],
        "operator_change_differs": fp_a != ast_fingerprint(operator_changed)[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    precheck = json.loads(PRECHECK.read_text(encoding="utf-8"))
    if precheck.get("status") != EXPECTED_PRECHECK_STATUS:
        raise SystemExit("audit precheck is not frozen")
    for value in precheck["frozen_source_artifacts"].values():
        if isinstance(value, dict) and "path" in value and "sha256" in value:
            path = ROOT / value["path"]
            if path.exists() and sha256_path(path) != value["sha256"]:
                raise SystemExit(f"frozen source hash mismatch: {value['path']}")

    rows: list[dict[str, Any]] = []
    skipped_fields: collections.Counter[str] = collections.Counter()
    split_file_hashes: dict[str, str] = {}
    for namespace, expected in (("qualification", 300), ("confirmation", 1000), ("reserve", 300)):
        path = args.private_dir / f"{namespace}.jsonl"
        split_file_hashes[namespace] = sha256_path(path)
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                selected, skipped = allowed_top_level_fields(line)
                skipped_fields.update(skipped)
                if selected["namespace"] != namespace:
                    raise SystemExit("namespace mismatch")
                if not {"reference_repr", "prompt"}.issubset(skipped):
                    raise SystemExit("expected private content fields were not skipped opaquely")
                fingerprint, nodes, features = ast_fingerprint(str(selected["source"]))
                rows.append(
                    {
                        "namespace": namespace,
                        "family_id": selected["family_id"],
                        "output_type": selected["output_type"],
                        "current": selected["canonical_skeleton_sha256"],
                        "normalized_token": selected["normalized_token_sha256"],
                        "strict": fingerprint,
                        "nodes": nodes,
                        "features": features,
                        "source_sha256": sha256_bytes(str(selected["source"]).encode()),
                    }
                )
                count += 1
        if count != expected:
            raise SystemExit(f"{namespace} row count mismatch: {count}/{expected}")

    strict_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    current_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    source_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        strict_groups[row["strict"]].append(row)
        current_groups[row["current"]].append(row)
        source_groups[row["source_sha256"]].append(row)
    duplicates = [group for group in strict_groups.values() if len(group) > 1]
    cross_split = [group for group in duplicates if len({row["namespace"] for row in group}) > 1]
    current_splits_strict = sum(
        1 for group in current_groups.values() if len({row["strict"] for row in group}) > 1
    )
    strict_splits_current = sum(
        1 for group in strict_groups.values() if len({row["current"] for row in group}) > 1
    )

    relaxed_pairs = 0
    relaxed_cross_split_pairs = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["output_type"] != right["output_type"]:
                continue
            if min(len(left["nodes"]), len(right["nodes"])) / max(
                len(left["nodes"]), len(right["nodes"])
            ) < 0.90:
                continue
            if multiset_jaccard(left["features"], right["features"]) < 0.95:
                continue
            ordered = difflib.SequenceMatcher(
                a=left["nodes"], b=right["nodes"], autojunk=False
            ).ratio()
            if ordered < 0.95:
                continue
            relaxed_pairs += 1
            if left["namespace"] != right["namespace"]:
                relaxed_cross_split_pairs += 1

    sizes = collections.Counter(len(group) for group in duplicates)
    duplicate_members = sum(len(group) for group in duplicates)
    split_members = {
        namespace: sum(1 for group in duplicates for row in group if row["namespace"] == namespace)
        for namespace in ("qualification", "confirmation", "reserve")
    }
    synthetic = synthetic_tests()
    synthetic_pass = all(synthetic.values())
    material = bool(duplicates)
    ruling = (
        "Q3_FRESH_INSTRUMENT_MATERIAL_FAMILY_VIOLATION_QUALIFICATION_BLOCKED"
        if material
        else "Q3_FRESH_INSTRUMENT_STRUCTURALLY_CLEAN_QUALIFICATION_RESUMED"
    )
    result = {
        "schema_version": "q3-fresh-external-review-structural-audit-v1",
        "classification": "ADDITIVE_MODEL_FREE_SOURCE_ONLY_AUDIT",
        "status": ruling,
        "precheck_sha256": sha256_path(PRECHECK),
        "private_split_hashes": split_file_hashes,
        "content_access": {
            "source_fields_decoded": True,
            "references_decoded": False,
            "prompts_decoded": False,
            "model_outputs_decoded": False,
            "correctness_decoded": False,
            "programs_executed": 0,
            "forbidden_field_occurrences_skipped_opaquely": dict(sorted(skipped_fields.items())),
            "examples_emitted": 0,
        },
        "synthetic_tests": {"checks": synthetic, "all_pass": synthetic_pass},
        "actual_dataset": {
            "rows": len(rows),
            "current_unique_family_ids": len({row["family_id"] for row in rows}),
            "current_unique_canonical_skeletons": len(current_groups),
            "strict_ast_unique_skeletons": len(strict_groups),
            "strict_duplicate_groups": len(duplicates),
            "strict_duplicate_members": duplicate_members,
            "strict_duplicate_excess_rows": duplicate_members - len(duplicates),
            "strict_group_size_distribution": dict(sorted(sizes.items())),
            "strict_cross_split_groups": len(cross_split),
            "strict_cross_split_members": sum(len(group) for group in cross_split),
            "duplicate_members_by_split": split_members,
            "exact_rendered_source_duplicate_groups": sum(
                1 for group in source_groups.values() if len(group) > 1
            ),
            "strict_identity_split_across_current_ids": strict_splits_current,
            "current_identity_merges_distinct_strict_ids": current_splits_strict,
            "relaxed_ast_candidate_pairs": relaxed_pairs,
            "relaxed_ast_cross_split_candidate_pairs": relaxed_cross_split_pairs,
        },
        "material_violation": material,
        "qualification_resume_permitted": not material,
        "historical_artifacts_modified": False,
        "qualification_partial_rows_scored": False,
        "qwen_inference": 0,
    }
    out = REVIEW / "Q3_EXTERNAL_REVIEW_STRUCTURAL_AUDIT.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
