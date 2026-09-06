from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts")

import analyze_q3_fresh_qualification_behavioral_postmortem as postmortem  # noqa: E402


def test_value_concentration_releases_counts_not_literals() -> None:
    rows = [
        {
            "semantic_evaluable": True,
            "canonical_value": json.dumps(["int", value], separators=(",", ":")),
            "value_type": "int",
        }
        for value in (0, 0, 1, 9)
    ]
    result = postmortem.value_concentration(rows)
    assert result["evaluable_rows"] == 4
    assert result["unique_values"] == 3
    assert result["top1_share"] == 0.5
    assert result["simple_constant_share"] == 0.75
    assert result["literal_values_released"] is False
    assert "9" not in json.dumps(result)


def test_cruxeval_prompt_split_is_exact() -> None:
    source = "def f(x):\n    return x + 1"
    input_value = "[3]"
    prompt = (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{source}\n```\n\n"
        f"Input: {input_value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )
    assert postmortem.split_cruxeval_prompt(prompt) == (source, input_value)


def test_ast_features_capture_structure() -> None:
    result = postmortem.ast_features("def f(x):\n    if x:\n        return x + 1\n    return 0\n")
    assert result["branches"] == 1
    assert result["arithmetic_ops"] == 1
    assert result["statements"] >= 4
