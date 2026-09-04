# Causal Epistemic Geometry

We study whether semantic failure patterns in frozen language models can be
causally controlled through activation interventions, and whether the changes
are organized by the geometry of the intervention space. The central object is
not accuracy alone: it is the itemwise pattern of failures—the model's semantic
blind spots.

## Current headline results

- **Q1 — controllability.** A fixed Qwen3-8B activation controller safely
  reorganized which CRUXEval items the model tended to fail, beyond average
  competence change and matched random-direction controls. The confirmatory
  classification is `Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`.
- **Q2 — geometry.** In a prospectively fixed bank of 31 intervention
  directions, pairwise intervention geometry predicted pairwise semantic
  blind-spot geometry. Q2 V4.1 closed as `Q2_V4_1_G2`; its independent radial
  results are `RS+` and `RT+`. A later prospective validation found positive
  A0 alignment for all 16 safety-conditioned fresh controllers
  (`Q2_OOS_V2_A0_PASS`).
- **Q3 — utility.** Not run. A DEVELOPMENT-only prompt-representation router
  was selectable on closed data, but true controller geometry did not add the
  frozen minimum value over learned policy identity; no fresh utility test has
  been run.

<!-- PROJECT_STATUS:START -->
**Current stage:** `Q3_1_PROMPT_REPRESENTATION_DEVELOPMENT_CLOSED` — **scientific claim:** `Q2_V4_1_G2__Q2_OOS_V2_A0_PASS`.

Q1 confirmatory evidence remains Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL. The separate DEVELOPMENT LiveCodeBench transfer is closed as Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY with a resolved independent audit. Historical Q2 V4.1 remains Q2_V4_1_G2 with RS+ and RT+. The closed fresh-controller validation is Q2_OOS_V2_A0_PASS and Q2_OOS_V2_FORENSIC_CLEAN: all 16 prospectively sampled fresh-controller row associations were positive. Q3.1 used 332 label-free prompt-only Qwen forwards on the closed 300-family development panel. Prompt representations made routing stable (+0.0533 over the cross-fitted champion; 5/5 positive folds), but true controller geometry exceeded a capacity-matched learned policy-ID control by only +0.0033, below the frozen +0.01 criterion. The ruling is Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL. No fresh holdout was allocated or inspected, and Q3 remains NOT_RUN.

**Next authorized action:** PRINCIPAL_REVIEW_OF_Q3_1_ATTRIBUTION_BOUNDARY
<!-- PROJECT_STATUS:END -->

This block is generated from [`project_state.yaml`](project_state.yaml). See the
short [current-status page](docs/CURRENT_STATUS.md) for the active scientific
state.

## Scientific story

```text
Q1: controllability  →  Q2: geometry  →  Q3: utility
    Can blind spots      Is their          Can the structure
    be moved causally?   movement ordered? yield useful systems?
```

Q1 establishes a narrow causal result on Qwen3-8B and CRUXEval. Q2 establishes
a relational result inside a fixed Qwen intervention subspace on a frozen
CRUXEval panel. Neither result establishes universal steering, model- or
task-generality, global smoothness, a Riemannian manifold, or collective
utility.

## Results at a glance

| Question | Canonical result | Evidence boundary |
|---|---|---|
| Q1, Qwen | Confirmatory pass | Safe, null-specific competence-adjusted complementarity on Qwen3-8B + CRUXEval |
| Q1, Ministral | Confirmatory model-level fail | Complementarity components were positive, but frozen validity/evaluability guards failed |
| Q1, character count | Negative boundary | The fixed Qwen controller did not transfer to long character counting |
| Q1, LiveCodeBench | Negative DEVELOPMENT boundary | The fixed Qwen controller did not satisfy the frozen null-specific transfer conjunction; corrected forensic audit matched exactly |
| Q2 V4.1 | `G2`, `RS+`, `RT+` | Relational and radial evidence within the fixed 31-direction Qwen subspace |
| Q2 fresh controllers | `Q2_OOS_V2_A0_PASS` | 16/16 controller-level A0 associations positive against the fixed historical atlas; same model/task/subspace laboratory |
| Q3 | Not run; Q3.1 development closed | Prompt representations enabled stable closed-panel routing, but incremental geometry attribution failed |

Exact numbers, classifications, and links to the evidence are in
[Scientific Results](docs/SCIENTIFIC_RESULTS.md).

## Start here

1. [Start Here](docs/START_HERE.md) — a five-minute scientific orientation.
2. [Scientific Results](docs/SCIENTIFIC_RESULTS.md) — exact current findings.
3. [Current Status](docs/CURRENT_STATUS.md) — what is closed, open, and next.
4. [Q1 Visual Evidence](docs/Q1_VISUAL_EVIDENCE.md) — reproducible figures and notebook.
5. [Q2 V4.1 Visual Evidence](docs/Q2_VISUAL_EVIDENCE_ROADMAP.md) and
   [Q2 OOS Visual Evidence](docs/Q2_OOS_VISUAL_EVIDENCE.md).
6. [Experiment Index](docs/EXPERIMENT_INDEX.md) — methodological genealogy.
7. [Claim–Evidence Matrix](docs/CLAIM_EVIDENCE_MATRIX.md) — allowed wording and limits.
8. [Reproducibility](docs/REPRODUCIBILITY.md) — tracked versus private artifacts.
9. [Document Index](docs/DOCUMENT_INDEX.md) — complete navigation.

## Repository map

```text
docs/          scientific narrative, policies, closeouts, and navigation
review/        frozen protocols, audits, aggregate results, and hash ledgers
experiments/   machine-readable registry, decisions, and prospective specs
src/           reusable experiment and analysis code
scripts/       execution, analysis, audit, and publication entry points
tests/         deterministic scientific and software invariants
notebooks/     narrative notebooks backed by tested loaders
manuscript/    publication figures and figure-data provenance
```

## Quick start

Python 3.11 or newer is supported for local analysis and validation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

make test
make lint
python -m compileall -q src scripts
make scientific-audit
```

Regenerate the publication-oriented Q1 figures when the hash-pinned private Q1
source bundle is available locally:

```bash
python scripts/generate_q1_paper_figures.py
```

The Q2 V4.1 and fresh-controller OOS figures regenerate from committed,
release-safe tables without model inference:

```bash
python scripts/generate_q2_paper_figures.py
python scripts/generate_q2_oos_paper_figures.py
```

No GPU or model download is required for the mock smoke, documentation checks,
or tracked aggregate analyses. Q1 figure regeneration is offline but requires
the private/hash-pinned row-level Q1 source bundle; generated figures and
derived tables are already tracked.

## Reproducibility and data availability

Git contains protocol locks, schedules, aggregate scientific results, audits,
tests, figure data, and SHA-256 identity ledgers. Some raw benchmark-derived
prompts, reference answers, generated text, and row-level scores are not
redistributed because of benchmark licensing, size, and research-data policy.
Their exact identities are recorded by hash. See
[Reproducibility](docs/REPRODUCIBILITY.md) for the three distinct levels:
tracked-artifact reproduction, full raw-data audit, and new model inference.

## Publication status

This is an active research repository, not a final archival data release. Q1,
Q2 V4.1, and the Q2 fresh-controller validation have publication-oriented
visual evidence packages. Use the [claim matrix](docs/CLAIM_EVIDENCE_MATRIX.md)
when citing results.

## License

Package metadata declares MIT licensing. Model, benchmark, and private
scientific-data artifacts retain their own licenses and release constraints.
