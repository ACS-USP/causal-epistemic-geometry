from __future__ import annotations

import ast
from pathlib import Path

import pytest

from epistemic_geometry.publication.q1 import loaders


def _require_private_q1_sources() -> None:
    missing = [
        path
        for path in loaders.expected_source_hashes()
        if not (loaders.ROOT / path).is_file()
    ]
    if missing:
        pytest.skip(
            "private/hash-pinned Q1 publication sources are not present in this clone"
        )


def test_frozen_q1_source_hashes_and_controller_identities_validate() -> None:
    _require_private_q1_sources()
    observed = loaders.validate_frozen_sources()
    assert observed == loaders.expected_source_hashes()
    assert loaders.validate_controller_identities() == {
        "Qwen": "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838",
        "Ministral": "0c467b7a452619d058afb07c96fd0cd8e20abb19a58d89674ab0a42e00ef2b94",
    }


def test_holdout_membership_order_and_confirmatory_journals() -> None:
    _require_private_q1_sources()
    item_ids = loaders.holdout_item_ids()
    assert len(item_ids) == 57
    assert len(set(item_ids)) == 57
    assert item_ids[:3] == ["sample_58", "sample_54", "sample_140"]
    for model in ("Qwen", "Ministral"):
        rows = loaders.load_confirmatory_journal(model)
        assert len(rows) == 798
        keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in rows}
        assert len(keys) == 798
        assert {row["rollout_index"] for row in rows} == {0, 1}
        assert {row["condition"] for row in rows} == loaders.REQUIRED_CONDITIONS
        assert all("raw_output" not in row for row in rows)


def test_q1_publication_code_has_no_q2_imports_or_sources() -> None:
    package = loaders.ROOT / "src/epistemic_geometry/publication/q1"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any("q2" in name.lower() for name in imports), path
    spec = loaders.figure_spec()
    assert spec["scientific_firewall"]["q2_semantic_sources_allowed"] is False
    assert all("/q2" not in source.lower() for source in spec["expected_source_sha256"])


def test_publication_package_never_carries_raw_text_fields() -> None:
    source = Path(loaders.__file__).read_text()
    assert "raw_output" not in loaders.JOURNAL_FIELDS
    assert "Q1 publication firewall" in source
