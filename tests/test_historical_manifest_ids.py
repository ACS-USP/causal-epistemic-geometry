from __future__ import annotations

import ast
import inspect

import pytest

import epistemic_geometry.benchmarks.reasoning.budget_interaction as budget_interaction


def _manifest_payload() -> dict[str, object]:
    items = [{"latent_id": "latent-a"}, {"latent_id": "latent-b"}]
    return {
        "manifests": {
            "cell/512": {"items": items},
            # Stage A repeats one item set across paired budgets.
            "cell/1024": {"items": list(items)},
        },
        "paired_budget_groups": {
            "cell": {"latent_ids": ["latent-a", "latent-b"]},
        },
    }


def test_extracts_the_same_ids_from_both_supported_representations() -> None:
    payload = _manifest_payload()

    assert budget_interaction.extract_historical_latent_ids(payload) == {
        "latent-a",
        "latent-b",
    }
    assert budget_interaction.extract_historical_latent_ids(
        [{"items": [{"latent_id": "latent-a"}, {"latent_id": "latent-b"}]}]
    ) == {"latent-a", "latent-b"}


def test_unions_partial_overlaps_across_manifest_containers_and_groups() -> None:
    assert budget_interaction.extract_historical_latent_ids(
        {
            "manifests": {
                "first": {
                    "items": [
                        {"latent_id": "latent-a"},
                        {"latent_id": "latent-b"},
                    ]
                },
                "second": {
                    "items": [
                        {"latent_id": "latent-b"},
                        {"latent_id": "latent-c"},
                    ]
                },
            }
        }
    ) == {"latent-a", "latent-b", "latent-c"}
    assert budget_interaction.extract_historical_latent_ids(
        {
            "paired_budget_groups": {
                "first": {"latent_ids": ["latent-a", "latent-b"]},
                "second": {"latent_ids": ["latent-b", "latent-c"]},
            }
        }
    ) == {"latent-a", "latent-b", "latent-c"}


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"manifests": [{"items": [{"latent_id": "latent-a"}, {"latent_id": "latent-a"}]}]},
            "duplicate",
        ),
        (
            {"paired_budget_groups": {"one": {"latent_ids": ["latent-a", "latent-a"]}}},
            "duplicate",
        ),
        (
            {"manifests": [{"items": [{"latent_id": ""}]}]},
            "non-empty",
        ),
        (
            {"manifests": [{"items": [{"latent_id": 7}]}]},
            "non-empty",
        ),
        (
            {
                "manifests": [{"items": [{"latent_id": "latent-a"}]}],
                "paired_budget_groups": {"cell": {"latent_ids": ["latent-b"]}},
            },
            "inconsistent",
        ),
    ],
)
def test_rejects_duplicate_malformed_or_inconsistent_payloads(
    payload: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        budget_interaction.extract_historical_latent_ids(payload)  # type: ignore[arg-type]


def test_extractor_source_has_no_file_or_journal_io() -> None:
    tree = ast.parse(inspect.getsource(budget_interaction))
    forbidden_calls = {
        "open",
        "read_text",
        "read_bytes",
        "loads",
        "load",
        "glob",
        "rglob",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint(forbidden_calls)
    extractor_source = inspect.getsource(budget_interaction.extract_historical_latent_ids)
    assert "journal" not in extractor_source.lower()
