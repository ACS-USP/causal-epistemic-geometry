from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from epistemic_geometry.benchmarks.e3.qualification import (
    CalibrationScoreRow,
    score_row,
    select_cells,
    semantic_probabilities,
    summarize_cell,
)


def _rows() -> list[CalibrationScoreRow]:
    rows: list[CalibrationScoreRow] = []
    for target in range(10):
        for replicate in range(2):
            prediction = target if replicate == 0 else (target + 1) % 10
            scores = [0.0] * 10
            scores[prediction] = 4.0
            latent_id = f"latent-{target}-{replicate}"
            for surface, channel in (
                ("canonical", "decimal"),
                ("surface_twin", "decimal"),
                ("canonical", "number_word"),
            ):
                rows.append(
                    CalibrationScoreRow(
                        latent_id=latent_id,
                        family="FSM10",
                        cell="length_4",
                        surface=surface,
                        response_channel=channel,
                        target=target,
                        scores=tuple(scores),
                    )
                )
    return rows


def test_semantic_scoring_computes_direct_observables() -> None:
    probabilities = semantic_probabilities([0.0] * 10)
    assert np.allclose(probabilities, np.full(10, 0.1))
    result = score_row(_rows()[0])
    assert result["prediction"] == 0
    assert result["correct"] is True
    assert result["true_answer_margin"] == 4.0
    assert result["brier"] >= 0


def test_qualification_and_selection_are_mechanical() -> None:
    summary = summarize_cell(_rows())
    assert summary["accuracy"] == 0.5
    assert summary["decimal_word_agreement"] == 1.0
    assert summary["surface_twin_agreement"] == 1.0
    assert np.isclose(summary["normalized_prediction_entropy"], 1.0)
    assert summary["qualification"]["qualified"] is True
    summaries = [summary, {**summary, "cell": "length_8", "accuracy": 0.6}]
    selected = select_cells(summaries)
    assert selected["FSM10"]["cell"] == "length_4"


def test_independent_recomputation_script_reads_only_score_vectors(tmp_path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "".join(json.dumps(row.to_record()) + "\n" for row in _rows()), encoding="utf-8"
    )
    output_path = tmp_path / "recomputed.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/recompute_q1_v2_instrument.py",
            str(scores_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "families_selected" in result.stdout
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["selected"]["FSM10"]["cell"] == "length_4"
