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
  results are `RS+` and `RT+`.
- **Q3 — utility.** Not run. No deployable selector, router, or committee has
  yet converted the measured complementarity into realized utility.

<!-- PROJECT_STATUS:START -->
**Current stage:** `Q1_SECOND_TASK_LIVECODEBENCH_STAGE_B_OPEN` — **scientific claim:** `Q2_V4_1_G2`.

Q2 V4.1 is closed and forensic-clean as Q2_V4_1_G2 with independent RS+ and RT+ radial results. A separate Q1 DEVELOPMENT transfer experiment is currently collecting a frozen 5,720-trajectory LiveCodeBench Stage B on Spark 2. Its fixed controller, eight nulls, 130 question families, parser, schedule, endpoints, and decision rules were locked before opening. No partial Stage-B scientific outcome is part of the repository state or may be inspected during collection.

**Next authorized action:** COMPLETE_FROZEN_5720_ROW_STAGE_B_AND_SEAL_BEFORE_ANALYSIS
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
| Q1, LiveCodeBench | Open DEVELOPMENT experiment | Frozen Stage B is collecting; no partial scientific result is available |
| Q2 V4.1 | `G2`, `RS+`, `RT+` | Relational and radial evidence within the fixed 31-direction Qwen subspace |
| Q3 | Not run | Utility remains an open question |

Exact numbers, classifications, and links to the evidence are in
[Scientific Results](docs/SCIENTIFIC_RESULTS.md).

## Start here

1. [Start Here](docs/START_HERE.md) — a five-minute scientific orientation.
2. [Scientific Results](docs/SCIENTIFIC_RESULTS.md) — exact current findings.
3. [Current Status](docs/CURRENT_STATUS.md) — what is closed, open, and next.
4. [Q1 Visual Evidence](docs/Q1_VISUAL_EVIDENCE.md) — reproducible figures and notebook.
5. [Experiment Index](docs/EXPERIMENT_INDEX.md) — methodological genealogy.
6. [Claim–Evidence Matrix](docs/CLAIM_EVIDENCE_MATRIX.md) — allowed wording and limits.
7. [Reproducibility](docs/REPRODUCIBILITY.md) — tracked versus private artifacts.
8. [Document Index](docs/DOCUMENT_INDEX.md) — complete navigation.

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

This is an active research repository, not a final archival data release. Q1
has a publication-oriented visual evidence package; the equivalent Q2 visual
package is a documented [externalization TODO](docs/Q2_VISUAL_EVIDENCE_ROADMAP.md).
Use the [claim matrix](docs/CLAIM_EVIDENCE_MATRIX.md) when citing results.

## License

Package metadata declares MIT licensing. Model, benchmark, and private
scientific-data artifacts retain their own licenses and release constraints.
