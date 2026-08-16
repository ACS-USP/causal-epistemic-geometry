from __future__ import annotations

from pathlib import Path

from epistemic_geometry.storage import storage_report


def test_storage_report_is_informational_and_handles_missing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "missing-workspace"
    report = storage_report(workspace, threshold_gib=1.0)
    assert report["workspace"]["exists"] is False
    assert report["warning"] == "workspace is missing"
    assert not workspace.exists()
