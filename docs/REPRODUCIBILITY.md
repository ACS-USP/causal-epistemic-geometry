# Reproducibility and data availability

The repository supports three different reproducibility levels. They should not
be conflated.

## 1. Reproduction from tracked artifacts

This level requires no private benchmark content and no model inference. Git
contains frozen protocols, schedules, aggregate results, audits, deterministic
analysis code, tests, and publication figure data.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

make test
make lint
python -m compileall -q src scripts
make state-check
make registry-check
make docs-check
make scientific-audit
```

In a public clone, tests that require the absent private/hash-pinned Q1 row-level
sources are reported as skips. With those exact sources restored at their
recorded paths and hashes, the same tests become full integration checks.

The generated Q1 figures, derived tables, and provenance manifests are tracked.
Full regeneration additionally requires that private source bundle:

```bash
python scripts/generate_q1_paper_figures.py
pytest -q tests/test_q1_publication_loaders.py \
  tests/test_q1_figure_provenance.py tests/test_q1_figure_tables.py
```

Q2 exact aggregate values are stored in
[`Q2_V4_1_SEMANTIC_CLOSEOUT.json`](../review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.json),
with relational estimands in [`ESTIMANDS.json`](../review/q2_v4_1_semantic_execution/ESTIMANDS.json),
radial results in [`RADIAL_RESULTS.json`](../review/q2_v4_1_semantic_execution/RADIAL_RESULTS.json),
and an independent crosscheck in [`FORENSIC_AUDIT.json`](../review/q2_v4_1_semantic_execution/FORENSIC_AUDIT.json).

## 2. Full raw-data audit

Some scientific artifacts are intentionally absent from Git:

- benchmark text, source code, prompts, and reference answers whose
  redistribution is restricted or unresolved;
- raw generated model text;
- row-level semantic score tables;
- large activation and output-distribution arrays;
- model weights and private GPU execution stores.

These artifacts are identified by SHA-256, row count, schema, and execution
commit. For Q2 V4.1, the canonical ledger is
[`ARTIFACT_HASHES.json`](../review/q2_v4_1_semantic_execution/ARTIFACT_HASHES.json).
The raw journal has 37,800 rows and SHA-256
`d726b473feca8c6922b545bdf8a217e8171c8267697ff2b9714b14e1a0363a99`;
the score ledger has 37,800 rows and SHA-256
`a6a9f4b419d4531716337d2277688063d5655167a5d7b1a9bd85b34217f8a33f`.

A complete third-party raw audit therefore currently requires controlled access
to the private artifact store and any benchmark materials that can legally be
shared. Hashes establish identity and detect mutation; they do not themselves
make absent bytes reproducible.

## 3. New model inference

Re-running generation is a new experiment, not a documentation check. It
requires the exact model/tokenizer revision, qualified numerical environment,
controller vectors, prompts, schedules, seeds, retry semantics, and licensed
benchmark data. Remote GPU infrastructure and credentials are intentionally not
stored in Git. Local mock fixtures validate software mechanics only.

## Environments

- Local analysis and docs: Python 3.11+; dependencies are declared in
  [`pyproject.toml`](../pyproject.toml).
- Historical scientific runs: use the environment fingerprint and exact
  package versions in each frozen lock/closeout.
- GPU endpoints, usernames, SSH keys, tokens, and private topology are never
  committed.

## Artifact integrity

Each canonical experiment should provide, where applicable:

1. a prospective lock and source commit;
2. a schedule and immutable logical-key definition;
3. a raw-data seal;
4. a hash ledger;
5. aggregate results;
6. an independent forensic audit.

The [Experiment Index](EXPERIMENT_INDEX.md) links these artifacts. When a raw
artifact is unavailable, the index and results page must say so explicitly.

## Publication blockers

- A release mechanism and license review are needed for raw CRUXEval and
  LiveCodeBench-derived content.
- Q2 raw journals, row-level scores, and several large label-free arrays are
  hash-pinned but not publicly downloadable.
- A publication-grade Q2 visual evidence package has not yet been generated.
- A durable archival environment/container for all historical inference stacks
  is not yet public.

These are external reproducibility limitations, not hidden scientific results.
