from __future__ import annotations

import json
from pathlib import Path

from epistemic_geometry.benchmarks.external.reanalysis import reanalyze_gate1

ROOT = Path(__file__).parents[1]


def test_gate1_reanalysis_uses_preserved_rows_and_corrects_semantic_accounting(tmp_path) -> None:
    output = tmp_path / "reanalysis"
    summary = reanalyze_gate1(
        ROOT / "review/full_nonthinking_smoke/journal.jsonl",
        output,
        historical_source_commit="2d03624",
        reanalysis_source_commit="test",
    )
    assert summary["summaries"]["FRESH_PSEUDOWORD_LONG"]["valid"] == 20
    assert summary["summaries"]["FRESH_PSEUDOWORD_LONG"]["correct"] == 15
    assert summary["summaries"]["FRESH_PSEUDOWORD_LONG"]["wrong"] == 5
    assert summary["summaries"]["FRESH_PSEUDOWORD_LONG"]["classification"] == "PROMISING"
    assert summary["summaries"]["CRUXEVAL_SEMANTIC"]["valid"] == 20
    assert summary["summaries"]["CRUXEVAL_SEMANTIC"]["correct"] == 8
    assert summary["summaries"]["CRUXEVAL_SEMANTIC"]["wrong"] == 12
    assert (output / "SEMANTIC_REANALYSIS.csv").exists()
    assert (output / "SEMANTIC_REANALYSIS.md").exists()
    version = json.loads((output / "parser_version.json").read_text())
    assert version["model_inference"] is False
    assert version["rules"]["item_specific_repairs"] is False
