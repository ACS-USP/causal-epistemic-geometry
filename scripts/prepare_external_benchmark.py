#!/usr/bin/env python3
"""Materialize an official external benchmark into normalized JSONL on RunPod.

This script is deliberately remote-only.  It never runs on the Mac because the
first data-loading operation calls the repository HF location guard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.adapters import adapter_for  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402


def _cruxeval_prompt(code: str, value: str) -> str:
    return (
        "Solve the following code-output prediction problem.\n\n"
        "Python function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Reason carefully, then end with exactly one line in the form "
        "FINAL: <the exact Python output>. Do not add text after FINAL."
    )


def _final_contract(prompt: str, answer_description: str) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "For this evaluation, do not use a judge or explanatory paraphrase. "
        f"End with exactly one line in the form FINAL: <{answer_description}>. "
        "Do not add text after FINAL."
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_dataset_revision(repo_id: str, requested: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=requested or "main")
    resolved = getattr(info, "sha", None)
    if not resolved:
        raise RuntimeError(f"Could not resolve immutable revision for {repo_id}")
    return str(resolved)


def _prepare_cruxeval(
    output: Path, requested_revision: str | None, limit: int | None
) -> dict[str, object]:
    require_remote_hf_execution("CRUXEval dataset loading")
    from datasets import load_dataset

    repo_id = "cruxeval-org/cruxeval"
    revision = _resolve_dataset_revision(repo_id, requested_revision)
    dataset = load_dataset(repo_id, split="test", revision=revision)
    records: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        records.append(
            {
                "item_id": str(row["id"]),
                "benchmark": "CRUXEval",
                "subtask": "output_prediction",
                "prompt": _cruxeval_prompt(str(row["code"]), str(row["input"])),
                "reference_answer": str(row["output"]),
                "evaluator": "python_literal",
                "source_revision": revision,
                "metadata": {
                    "dataset_repo": repo_id,
                    "dataset_revision": revision,
                    "native_fields": ["code", "input", "output", "id"],
                    "official_evaluator": "facebookresearch/cruxeval/evaluation",
                },
            }
        )
    _write_jsonl(output, records)
    return {"dataset": repo_id, "revision": revision, "n_items": len(records)}


def _prepare_livecodebench(
    output: Path, requested_revision: str | None, limit: int | None
) -> dict[str, object]:
    require_remote_hf_execution("LiveCodeBench dataset loading")
    from datasets import load_dataset

    repo_id = "livecodebench/test_generation"
    revision = _resolve_dataset_revision(repo_id, requested_revision)
    dataset = load_dataset(repo_id, split="test", revision=revision)
    records: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        tests = json.loads(str(row["test"]))
        if not tests or not isinstance(tests[0], dict):
            raise ValueError(f"LiveCodeBench row {row['question_id']} has no first test case")
        test = tests[0]
        testcase = (
            f"assert {row['function_name']}({test['input']}) == # TODO"
        )
        prompt = _final_contract(
            "Problem:\n"
            + str(row["question_content"])
            + "\nFunction:\n```\n"
            + str(row["starter_code"])
            + "\n```\n\nPlease complete this test case:\n```\n"
            + testcase
            + "\n```",
            "the exact Python output expression",
        )
        records.append(
            {
                "item_id": f"{row['question_id']}:{row['test_id']}",
                "benchmark": "LiveCodeBench",
                "subtask": "test_output_prediction",
                "prompt": prompt,
                "reference_answer": str(test["output"]),
                "evaluator": "python_literal",
                "source_revision": revision,
                "metadata": {
                    "dataset_repo": repo_id,
                    "dataset_revision": revision,
                    "official_scenario": "testoutputprediction",
                    "official_evaluator": (
                        "lcb_runner.evaluation.compute_test_output_prediction_metrics"
                    ),
                    "question_id": str(row["question_id"]),
                    "test_id": int(row["test_id"]),
                },
            }
        )
    _write_jsonl(output, records)
    return {"dataset": repo_id, "revision": revision, "n_items": len(records)}


def _prepare_livebench(
    output: Path, requested_revision: str | None, limit: int | None
) -> dict[str, object]:
    require_remote_hf_execution("LiveBench objective dataset loading")
    from datasets import load_dataset

    repo_id = "livebench/reasoning"
    revision = _resolve_dataset_revision(repo_id, requested_revision)
    dataset = load_dataset(repo_id, split="test", revision=revision)
    records: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        turns = row.get("turns")
        if not isinstance(turns, list) or len(turns) != 1:
            raise ValueError(f"LiveBench row {row['question_id']} is not a single-turn item")
        prompt = _final_contract(str(turns[0]), "the exact comma-separated answer payload")
        records.append(
            {
                "item_id": str(row["question_id"]),
                "benchmark": "LiveBench",
                "subtask": "objective_subtask",
                "prompt": prompt,
                "reference_answer": str(row["ground_truth"]),
                "evaluator": "exact",
                "source_revision": revision,
                "metadata": {
                    "dataset_repo": repo_id,
                    "dataset_revision": revision,
                    "official_task": str(row["task"]),
                    "official_evaluator": (
                        "LiveBench deterministic task scorer; exact payload adapter"
                    ),
                    "llm_judge": False,
                },
            }
        )
    _write_jsonl(output, records)
    return {"dataset": repo_id, "revision": revision, "n_items": len(records)}


def _prepare_generic(
    candidate: str,
    source: Path,
    output: Path,
    revision: str,
    evaluator: str,
    limit: int | None,
) -> dict[str, object]:
    """Normalize a source JSONL exported by the candidate's official loader.

    This path intentionally requires explicit evaluator and revision values.
    It will not infer an evaluator from model outputs or use an LLM judge.
    """

    adapter = adapter_for(candidate)
    records: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if limit is not None and len(records) >= limit:
                break
            raw = json.loads(line)
            prompt = raw.get("prompt") or raw.get("question_content") or raw.get("question")
            reference = (
                raw.get("reference_answer")
                or raw.get("output")
                or raw.get("answer")
                or raw.get("expected_output")
            )
            item_id = raw.get("item_id") or raw.get("id") or raw.get("question_id")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (prompt, reference, item_id)
            ):
                raise ValueError(
                    f"{source}:{index + 1}: explicit prompt/id/reference fields are required"
                )
            records.append(
                {
                    "item_id": str(item_id),
                    "benchmark": candidate,
                    "subtask": adapter.spec.subtask,
                    "prompt": str(prompt),
                    "reference_answer": str(reference),
                    "evaluator": evaluator,
                    "source_revision": revision,
                    "metadata": {
                        "native_record_fields": sorted(raw),
                        "official_source_revision": revision,
                    },
                }
            )
    _write_jsonl(output, records)
    items = adapter.load_items(output)
    return {"candidate": candidate, "revision": revision, "n_items": len(items)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=[spec.name for spec in __import__(
        "epistemic_geometry.benchmarks.external.adapters", fromlist=["candidate_specs"]
    ).candidate_specs()])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--evaluator", choices=["exact", "python_literal", "json"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.candidate == "CRUXEval":
        summary = _prepare_cruxeval(args.output, args.revision, args.limit)
    elif args.candidate == "LiveCodeBench":
        summary = _prepare_livecodebench(args.output, args.revision, args.limit)
    elif args.candidate == "LiveBench" and args.source_jsonl is None:
        summary = _prepare_livebench(args.output, args.revision, args.limit)
    else:
        if args.source_jsonl is None or args.evaluator is None or not args.revision:
            parser.error("non-CRUX candidates require --source-jsonl, --revision, and --evaluator")
        require_remote_hf_execution(f"{args.candidate} source validation")
        summary = _prepare_generic(
            args.candidate,
            args.source_jsonl,
            args.output,
            args.revision,
            args.evaluator,
            args.limit,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
