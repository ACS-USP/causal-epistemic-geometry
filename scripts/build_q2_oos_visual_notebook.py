#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the executed, deterministic Q2 OOS visual-story notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/q2_oos_visual_story.ipynb"


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(
    source: str,
    execution_count: int,
    *,
    text_markdown: str | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    if text_markdown is not None:
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "text/plain": "<IPython.core.display.Markdown object>",
                    "text/markdown": text_markdown,
                },
            }
        )
    if image_path is not None:
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "text/plain": "<IPython.core.display.Image object>",
                    "image/png": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                },
            }
        )
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def build() -> dict[str, Any]:
    figure_dir = ROOT / "manuscript/figures/paper1_q2_oos"
    analysis = json.loads(
        (
            ROOT / "review/q2_oos_fresh_controller_design/v2_semantic_execution/"
            "Q2_OOS_V2_SEMANTIC_ANALYSIS.json"
        ).read_text()
    )
    primary = analysis["primary"]
    cells = [
        markdown(
            "# Q2 OOS visual story: does geometry generalize to fresh controllers?\n\n"
            "This notebook is a narrative view over the closed, forensic-clean Q2 OOS V2 result. "
            "It is not a scientific source of truth: code regenerates figures from hash-validated, "
            "release-safe derived tables. Raw generations, benchmark text, and private error arrays "
            "are absent from Git. The historical Q2 V4.1 result remains separate.\n"
        ),
        code(
            "import math\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "pipeline = __import__(\n"
            "    'epistemic_geometry.publication.q2_oos.pipeline',\n"
            "    fromlist=['generate_visual_evidence'],\n"
            ")\n"
            "package = pipeline.generate_visual_evidence(ROOT)\n"
            "TABLE_DIR = ROOT / 'manuscript/data/paper1_q2_oos/derived_figure_tables'\n"
            "display(Markdown('**Hash validation and deterministic Q2 OOS regeneration: PASS**'))\n",
            1,
            text_markdown="**Hash validation and deterministic Q2 OOS regeneration: PASS**",
        ),
        markdown(
            "## 1. The prospective generalization unit\n\n"
            "The 31 historical controllers form a fixed reference atlas. Sixteen new controllers were "
            "sampled prospectively, conditioned only on the frozen safety gate, and never redrawn from "
            "semantic outcomes. For each fresh controller $i$, the experiment asks whether historical "
            "controllers that are geometrically closer also tend to have more similar blind-spot profiles.\n\n"
            "The independent unit is **one fresh controller**, not one of the 496 fresh×old dyads.\n"
        ),
        code(
            "display(Image(filename=str(package['figures']['main']['png'])))\n",
            2,
            image_path=figure_dir / "figure_q2_oos_fresh_controller_generalization.png",
        ),
        markdown(
            "## 2. From intervention distance to blind-spot distance\n\n"
            "**A0** is the pre-outcome coefficient-space dissimilarity in the fixed learned rank-8 "
            "intervention subspace. It is fixed before semantic outcomes.\n\n"
            "For each shell, the two-rollout error profiles give an unbiased finite-panel estimate of "
            "centered blind-spot-shape distance $D^{shape}$. Centering removes the component explained "
            "only by average error-rate shift. Negative finite-sample estimates are retained.\n\n"
            "For fresh controller $i$, correlate $A0(i,j)$ with $D^{shape}(i,j)$ over the 31 historical "
            "references $j$, separately in MEDIUM and STRONG shells, then average the two correlations:\n\n"
            "$$r_i=\\tfrac12[\\rho_S(A0_{i,\\cdot}^{M},D_{i,\\cdot}^{shape,M})+"
            "\\rho_S(A0_{i,\\cdot}^{S},D_{i,\\cdot}^{shape,S})].$$\n"
        ),
        code(
            "controllers = pd.read_csv(TABLE_DIR / 'controller_associations.csv')\n"
            "display(controllers)\n",
            3,
            text_markdown=(
                f"**Frozen controller table:** {primary['positive_count']} of "
                f"{len(primary['r_i'])} equal-shell associations are positive."
            ),
        ),
        markdown(
            "## 3. Exact controller-level sign test\n\n"
            "The prospectively frozen null is $P(r_i>0)\\leq 0.5$; zero counts as non-success. "
            "The exact one-sided Binomial upper tail uses the 16 controller signs. Global correlation "
            "is descriptive and cannot replace this controller-level inference.\n"
        ),
        code(
            "positive = int(controllers['primary_positive'].sum())\n"
            "total = len(controllers)\n"
            "exact_p = sum(math.comb(total, k) for k in range(positive, total + 1)) / 2**total\n"
            "display(Markdown(\n"
            "    f'**Primary:** {positive}/{total} positive; exact one-sided $p={exact_p:.12g}$. '\n"
            "    'Classification: `Q2_OOS_V2_A0_PASS`.'\n"
            "))\n",
            4,
            text_markdown=(
                f"**Primary:** {primary['positive_count']}/{len(primary['r_i'])} positive; "
                f"exact one-sided $p={primary['exact_binomial']['p_value']:.12g}$. "
                "Classification: `Q2_OOS_V2_A0_PASS`."
            ),
        ),
        markdown(
            "## 4. Global geometry and secondary analyses\n\n"
            "The full 16×31 association is a useful effect-size summary but its dyads are dependent. "
            "A1, A2, and D2 were frozen secondary geometries. The fresh×fresh result uses a node-level "
            "jackknife and is `SECONDARY_ONLY_CANNOT_RESCUE_PRIMARY`.\n"
        ),
        code(
            "globals_ = pd.read_csv(TABLE_DIR / 'global_associations.csv')\n"
            "fresh_fresh = pd.read_csv(TABLE_DIR / 'fresh_fresh_summary.csv')\n"
            "display(globals_)\n"
            "display(fresh_fresh)\n"
            "display(Image(filename=str(package['figures']['s3']['png'])))\n",
            5,
            text_markdown="**Primary and secondary roles are encoded in the committed tables and figure spec.**",
            image_path=figure_dir / "supplement_q2_oos_fresh_fresh_pairs.png",
        ),
        markdown(
            "## 5. Robustness across shells and controller identities\n\n"
            "The shell plot exposes the two components of every $r_i$ without sorting by outcome. "
            "The LOFO plot removes each fresh controller in the same frozen order; it is a sensitivity "
            "analysis, not a second primary test.\n"
        ),
        code(
            "display(Image(filename=str(package['figures']['s1']['png'])))\n",
            6,
            image_path=figure_dir / "supplement_q2_oos_medium_vs_strong.png",
        ),
        code(
            "display(Image(filename=str(package['figures']['s2']['png'])))\n",
            7,
            image_path=figure_dir / "supplement_q2_oos_distribution_and_lofo.png",
        ),
        markdown(
            "## 6. Why the archived item bootstrap is not shown as a confidence interval\n\n"
            "A post-closeout audit reproduced the historical 50,000 item resamples exactly and found "
            "no defect in the primary objects. An ordinary size-300 resample contains only about 190 "
            "unique items and about 150 multiplicity-effective items. Recomputing the nonlinear "
            "two-rollout $D^{shape}$→Spearman estimator on that perturbed support shifts the distribution "
            "downward. Synthetic calibration did not validate conventional percentile-CI coverage.\n\n"
            "The archived values are therefore displayed only as an **item-panel perturbation "
            "sensitivity distribution**. No post-hoc replacement interval strengthens the primary result.\n"
        ),
        code(
            "display(Image(filename=str(package['figures']['s5']['png'])))\n",
            8,
            image_path=figure_dir / "supplement_q2_oos_item_bootstrap_diagnostic.png",
        ),
        markdown(
            "## 7. Claim boundary\n\n"
            "The result supports controller-identity generalization **within** the same frozen Qwen3-8B, "
            "CRUXEval, layer-27, learned rank-8 intervention laboratory and fixed 31-controller atlas. "
            "It does not establish cross-task or cross-model generalization, random-subspace specificity, "
            "universal geometry, manifold structure, or Q3 utility. A matched random rank-8 subspace "
            "control remains a prospective design question.\n\n"
            "To regenerate the public package from committed tables, run `python "
            "scripts/generate_q2_oos_paper_figures.py`. Regenerating the derived tables requires the "
            "private hash-pinned `D_SHAPE.npz`; the public manifest records its identity.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
