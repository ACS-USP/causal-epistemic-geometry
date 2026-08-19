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
        "2. **Fresh character counting:** the authorized worker journaled 20/30",
        "   fixed trajectories (short and medium strata); the long stratum did not run",
        "   before the hard cost gate. The partial rows had 14 valid correct answers,",
        "   zero valid wrong answers, and six format failures. No complete gate was",
        "   applied, so this is not a scientific qualification result.",
        "3. **Character-count candidate:** none. WORDLIKE_SHORT and",
        "   FRESH_PSEUDOWORD_MEDIUM were observed only partially and the long stratum",
        "   was not run; no stratum was selected.",
        "4. **Geometry Bench G:** at the preselected zero-based block 31, weekday",
        "   prompt-boundary activations show a nontrivial descriptive relationship",
        "   (Spearman 0.675, permutation p=0.0027), while letters are below the",
        "   frozen effect threshold (Spearman 0.357). This is not a causal result.",
        "5. **Direct geometry control:** it shows no corresponding simple behavior-space",
        "   structure in either domain (weekday Spearman -0.001; letters -0.052).",
        "6. **Bench G / Bench E separation:** conceptually plausible, but not",
        "   empirically established here because Bench E did not complete and the",
        "   geometry result is activation-only.",
        "7. **Dense code vectors:** not ready; no verified local official per-test-case",
        "   LiveCodeBench artifact/evaluator was available and no pilot was run.",
        "",
        "8. **Next single experiment:** `STOP_Q1_INSTRUMENT_SEARCH`, pending",
        "   principal review. No steering, vector construction, or new benchmark",
        "   campaign is authorized by this micro-screen.",
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
        "The stop recommendation reflects the incomplete character-count gate, the",
        "semantic saturation of CRUXEval after type-aware postmortem, and the absence",
        "of corresponding direct-behavior structure in the geometry control. If Bench E",
        "is pursued later, authorize a separate bounded technical calibration rather",
        "than treating these partial rows as a result.",
        "",
        f"Estimated GPU cost: `${cost:.3f}` (hard stop `${lifecycle['hard_stop']}`).",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("V4_REPORT_WRITTEN", stable_digest("V4-FINAL-REPORT", "\n".join(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
