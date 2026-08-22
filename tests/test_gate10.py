from __future__ import annotations

from pathlib import Path

import numpy as np

from epistemic_geometry.benchmarks.v4.character_semantic_v3 import (
    evaluate_character_count_answer_v3,
)
from epistemic_geometry.experiments import gate10


def test_character_semantic_v3_is_typed_and_condition_blind() -> None:
    assert evaluate_character_count_answer_v3("FINAL: 7", "7").correct
    wrong_type = evaluate_character_count_answer_v3('FINAL: "7"', "7")
    assert wrong_type.commitment_valid and wrong_type.semantic_evaluable and not wrong_type.correct
    assert not evaluate_character_count_answer_v3("FINAL: 7\nFINAL: 8", "7").commitment_valid
    assert not evaluate_character_count_answer_v3("FINAL: 7", "7", truncated=True).commitment_valid


def test_gate10_generator_is_fresh_deterministic_and_exact(tmp_path: Path) -> None:
    historical = {
        "item_ids": ["old"],
        "generator_seeds": [1],
        "exact_strings": ["old"],
        "item_hashes": ["old"],
    }
    first = gate10.generate_fresh_manifest(historical, 200)
    second = gate10.generate_fresh_manifest(historical, 200)
    assert first == second and len(first["items"]) == 200
    assert len({x["text"] for x in first["items"]}) == 200
    assert all(
        31 <= len(x["text"]) <= 60 and x["text"].count(x["target_character"]) == x["answer"]
        for x in first["items"]
    )


def test_gate10_schedule_has_2800_unique_keys_and_seeds() -> None:
    rows = gate10.build_schedule([f"i{i}" for i in range(200)])
    assert len(rows) == 2800
    assert len({(r["item_id"], r["condition"], r["rollout_index"]) for r in rows}) == 2800
    assert len({r["seed"] for r in rows}) == 2800


def test_opportunity_gate() -> None:
    errors = np.zeros((200, 2), dtype=int)
    errors[:20] = 1
    errors[20:100, 0] = 1
    summary = {"commitment_validity": 0.99, "semantic_evaluability": 0.99, "accuracy": 0.70}
    assert gate10.opportunity(errors, summary)["pass"]


def test_gate10_strong_classification() -> None:
    base = {"commitment_validity": 0.99, "semantic_evaluability": 0.99, "accuracy": 0.75}
    ctl = {"commitment_validity": 0.98, "semantic_evaluability": 0.98, "accuracy": 0.80}
    point = {"G": 0.05, "C": 0.03, "D": 0.06, "G_norm": 0.25, "rescue": 0.08, "damage": 0.03}
    random = {m: {"mean": 0.005, "max": 0.01} for m in ("G", "C", "D")}
    bootstrap = {
        k: {"q025": 0.001}
        for k in (
            "meaningful:G",
            "meaningful:C",
            "meaningful:D",
            "meaningful:G_minus_random_mean",
            "meaningful:C_minus_random_mean",
        )
    }
    classification, _ = gate10.classify_gate10(
        baseline=base,
        controller=ctl,
        point=point,
        random_summary=random,
        bootstrap=bootstrap,
        loo={"G": True, "C": True, "D": True},
        opportunity_pass=True,
        style_transfer=False,
        accuracy_bootstrap_positive=True,
    )
    assert classification == "GATE10_STRONG_CROSS_DOMAIN_USEFUL_COMPLEMENTARITY"
