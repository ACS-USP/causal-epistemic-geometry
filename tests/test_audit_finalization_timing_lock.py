from __future__ import annotations

from pathlib import Path

import pytest
from scripts.audit_finalization_timing_lock import audit
from tests.test_freeze_finalization_timing_lock import _fixture_lock


def test_audits_synthetic_frozen_lock(tmp_path: Path) -> None:
    assert audit(_fixture_lock(tmp_path))["status"] == "FINALIZATION_TIMING_LOCK_AUDIT_PASS"


def test_rejects_semantic_field(tmp_path: Path) -> None:
    lock = _fixture_lock(tmp_path)
    import json

    p = lock.parent / "MANIFEST.json"
    rows = json.loads(p.read_text())
    rows[0]["correct"] = True
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        audit(lock)
