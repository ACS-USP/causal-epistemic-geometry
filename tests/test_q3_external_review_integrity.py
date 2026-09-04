from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_q3_fresh_instrument_integrity.py"
SPEC = importlib.util.spec_from_file_location("q3_external_integrity", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_source_only_reader_does_not_decode_forbidden_values(monkeypatch) -> None:
    row = {
        "namespace": "qualification",
        "family_id": "fixture",
        "source": "def solve(data):\n    return 1\n",
        "canonical_skeleton_sha256": "a" * 64,
        "normalized_token_sha256": "b" * 64,
        "operations": [],
        "output_type": "int",
        "prompt": "PRIVATE_PROMPT_SENTINEL",
        "reference_repr": "PRIVATE_REFERENCE_SENTINEL",
    }
    original = json.loads

    def guarded_loads(value, *args, **kwargs):
        assert "PRIVATE_PROMPT_SENTINEL" not in value
        assert "PRIVATE_REFERENCE_SENTINEL" not in value
        return original(value, *args, **kwargs)

    monkeypatch.setattr(AUDIT.json, "loads", guarded_loads)
    selected, skipped = AUDIT.allowed_top_level_fields(json.dumps(row, sort_keys=True))
    assert selected["source"] == row["source"]
    assert {"prompt", "reference_repr"}.issubset(skipped)


def test_independent_ast_normalizer_preserves_use_relations() -> None:
    original = "def solve(data):\n    x = data['n']\n    y = x + 1\n    return y\n"
    renamed = (
        "def solve(payload):\n"
        "    first = payload['n']\n"
        "    second = first + 9\n"
        "    return second\n"
    )
    changed_use = (
        "def solve(payload):\n"
        "    first = payload['n']\n"
        "    second = first + 9\n"
        "    return first\n"
    )
    assert AUDIT.ast_fingerprint(original)[0] == AUDIT.ast_fingerprint(renamed)[0]
    assert AUDIT.ast_fingerprint(original)[0] != AUDIT.ast_fingerprint(changed_use)[0]


def test_required_synthetic_identity_checks_all_pass() -> None:
    checks = AUDIT.synthetic_tests()
    assert checks
    assert all(checks.values())
