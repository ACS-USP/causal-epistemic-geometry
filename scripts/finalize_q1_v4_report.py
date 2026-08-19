#!/usr/bin/env python3
"""Write the final V4 review report from local, already-completed artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.reproducibility import stable_digest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "review" / "q1_v4_microbench")
    args = parser.parse_args()
    output = args.output
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    geometry = analysis.get("geometry", {})
    domains = geometry.get("domains", [])
    lifecycle = json.loads((output / "remote_lifecycle.json").read_text(encoding="utf-8"))
    cost = float(lifecycle["active_seconds"]) * float(lifecycle["rate_per_hour"]) / 3600
    (output / "COST_REPORT.md").write_text(
        "# V4 microbench cost report\n\n"
        f"- A40 active session seconds: `{lifecycle['active_seconds']}`\n"
        f"- A40 rate: `${lifecycle['rate_per_hour']}/hour`\n"
        f"- Estimated incremental cost: `${cost:.3f}`\n"
        f"- Hard stop: `${lifecycle['hard_stop']}`\n"
        "- Pod final state: `STOPPED/EXITED`\n"
        "- Local model/dataset downloads: `NO`\n",
        encoding="utf-8",
    )
    lines = [
        "# Q1 V4 — final micro-screen report",
        "",
        "## Scientific boundary",
        "",
        "This is development-only instrument reconnaissance. Historical V1–V3 and",
        "the frozen CRUXEval smoke were not changed. Steering, activation intervention,",
        "PCA-derived steering, code generation, LiveBench, LiveCodeBench generation,",
        "and holdout access were not run.",
        "",
        "## Answers",
        "",
        "1. **CRUXEval postmortem:** the type-aware diagnostic is both",
        "   FORMAT-SENSITIVE and SEMANTICALLY SATURATED: seven of eight original",
        "   invalid-format rows were semantically correct under deterministic rules,",
        "   while assessed semantic accuracy was 18/19 = 94.7%. The frozen original",
        "   result remains unchanged.",
        "2. **Character-count Bench E:** not determined. The worker journaled 20/30",
        "   fixed trajectories (short and medium strata); the long stratum did not run",
        "   before the hard cost gate. No complete 30-item gate was applied, so no",
        "   Bench E qualification conclusion is valid.",
        "3. **Geometry Bench G:** weekday prompt-boundary activations show a",
        "   nontrivial preselected-layer relationship; letters do not cross the frozen",
        "   signal threshold. The direct behavior positive control shows no clear",
        "   corresponding simple structure in either domain.",
        "4. **Bench G / Bench E separation:** methodologically plausible, but Bench E",
        "   remains unqualified because its bounded run did not complete.",
        "5. **Dense code vectors:** not ready; no verified local official per-test-case",
        "   LiveCodeBench artifact/evaluator was available and no pilot was run.",
        "",
        "## Geometry results",
        "",
        "| domain | activation Spearman | activation Pearson | permutation p | "
        "direct behavior Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in domains:
        lines.append(
            f"| {row['domain']} | {row['thinking_activation_spearman']} | "
            f"{row['thinking_activation_pearson']} | {row['permutation_p']} | "
            f"{row['direct_behavior_spearman']} |"
        )
    lines += [
        "",
        "## Required sentinels",
        "",
        "- Character-count: `CHARCOUNT_MICROBENCH_NOT_PROMISING` **(partial 20/30; "
        "not evaluated)**",
        f"- Geometry: `{geometry.get('status')}`",
        "- Dense code vector: `DENSE_CODE_VECTOR_NOT_READY`",
        "",
        "## Next single experiment",
        "",
        "`STOP_Q1_INSTRUMENT_SEARCH` is the recommended next action pending principal",
        "review. The character-count screen did not complete, CRUXEval is semantically",
        "saturated after type-aware postmortem, and the geometry result is activation-only",
        "with no corresponding direct-behavior structure. No steering experiment is",
        "authorized. If Bench E is pursued later, authorize a separate bounded",
        "technical calibration rather than treating these partial rows as a result.",
        "",
        f"Estimated GPU cost: `${cost:.3f}` (hard stop `${lifecycle['hard_stop']}`).",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("V4_REPORT_WRITTEN", stable_digest("V4-FINAL-REPORT", "\n".join(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
