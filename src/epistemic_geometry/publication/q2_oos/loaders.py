"""Hash-validated loaders for release-safe Q2 OOS figure data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_PATH = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
    "Q2_OOS_V2_SEMANTIC_ANALYSIS.json"
)
DIAGNOSTIC_PATH = (
    ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
    "item_bootstrap_diagnostic/Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_RESULT.json"
)
FIGURE_SPEC_PATH = ROOT / "manuscript/figures/paper1_q2_oos/FIGURE_SPEC.json"
TABLE_DIR = ROOT / "manuscript/data/paper1_q2_oos/derived_figure_tables"

EXPECTED_SOURCES = {
    (
        "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
        "Q2_OOS_V2_SEMANTIC_ANALYSIS.json"
    ): "97913256d32dcbdfd30fb247bdf925ed0ad0d6a8d39da29a0195bfd7845987c5",
    (
        "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
        "item_bootstrap_diagnostic/Q2_OOS_V2_ITEM_BOOTSTRAP_DIAGNOSTIC_RESULT.json"
    ): "7465e81d568854b92a0a7e6f0e46ec4a800577459df663b044c1975ef5be1573",
    (
        "review/q2_oos_fresh_controller_design/v2_presemantic_closeout/PREDICTION_MATRICES.npz"
    ): "b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703",
}
TABLE_NAMES = (
    "controller_associations",
    "global_associations",
    "fresh_fresh_pairs",
    "fresh_fresh_summary",
    "lofo",
    "bootstrap_diagnostic",
    "runtime_summary",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sources() -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in EXPECTED_SOURCES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Q2 OOS source hash mismatch: {relative}")
        verified[relative] = actual
    analysis = read_json(ANALYSIS_PATH)
    diagnostic = read_json(DIAGNOSTIC_PATH)
    if analysis["primary"]["classification"] != "Q2_OOS_V2_A0_PASS":
        raise RuntimeError("unexpected Q2 OOS primary classification")
    if analysis["forensic"]["status"] != "Q2_OOS_V2_FORENSIC_CLEAN":
        raise RuntimeError("unexpected Q2 OOS forensic status")
    if diagnostic["ruling"] != "Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED":
        raise RuntimeError("unexpected item-bootstrap diagnostic ruling")
    return verified


def load_sources() -> dict[str, Any]:
    hashes = validate_sources()
    tables = {name: pd.read_csv(TABLE_DIR / f"{name}.csv") for name in TABLE_NAMES}
    controllers = tables["controller_associations"]
    if controllers.shape[0] != 16 or controllers["controller_order"].tolist() != list(range(1, 17)):
        raise RuntimeError("controller table is not in frozen order")
    if not controllers["primary_positive"].all():
        raise RuntimeError("controller table disagrees with frozen 16/16 result")
    if tables["fresh_fresh_pairs"].shape[0] != 120:
        raise RuntimeError("fresh-by-fresh pair table must contain C(16,2)=120 rows")
    return {
        "analysis": read_json(ANALYSIS_PATH),
        "diagnostic": read_json(DIAGNOSTIC_PATH),
        "spec": read_json(FIGURE_SPEC_PATH),
        "tables": tables,
        "source_hashes": hashes,
    }


__all__ = ["ROOT", "TABLE_DIR", "load_sources", "sha256", "validate_sources"]
