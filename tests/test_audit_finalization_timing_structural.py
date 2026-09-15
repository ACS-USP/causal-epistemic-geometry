from __future__ import annotations

import json

import pytest
from scripts.analyze_finalization_timing_structural import analyze
from scripts.audit_finalization_timing_structural import StructuralAuditError, audit
from tests.test_analyze_finalization_timing_structural import _seal


def test_independent_structural_audit_passes(tmp_path) -> None:
    seal_path = _seal(tmp_path, d75_marker=True)
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analyze(seal_path)))
    payload = audit(seal_path, analysis_path)
    assert payload["classification"] == "STRUCTURAL_TIMING_ANALYSIS_INDEPENDENT_AUDIT_PASS"
    assert payload["max_absolute_difference"] == 0.0


def test_independent_structural_audit_rejects_numeric_tampering(tmp_path) -> None:
    seal_path = _seal(tmp_path, d75_marker=False)
    analysis_path = tmp_path / "analysis.json"
    result = analyze(seal_path)
    result["evalues"]["d75_earlier_by_delta"] = 999.0
    analysis_path.write_text(json.dumps(result))
    with pytest.raises(StructuralAuditError, match="numeric"):
        audit(seal_path, analysis_path)
