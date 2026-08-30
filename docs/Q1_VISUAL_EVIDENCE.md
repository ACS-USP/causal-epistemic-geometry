# Q1 visual evidence package

The publication-oriented Q1 visual package is generated exclusively from
hash-validated frozen Q1 artifacts. It does not read Q2 semantic outputs and it
does not alter any frozen scientific classification.

## Entry points

- Figure contract: [`Q1_FIGURE_SPEC.md`](Q1_FIGURE_SPEC.md)
- Narrative notebook: [`../notebooks/q1_visual_story.ipynb`](../notebooks/q1_visual_story.ipynb)
- Generated figures: [`../manuscript/figures/paper1/`](../manuscript/figures/paper1/)
- Figure provenance: [`../manuscript/figures/paper1/SOURCE_MANIFEST.json`](../manuscript/figures/paper1/SOURCE_MANIFEST.json)
- Derived tables: [`../manuscript/data/paper1/derived_figure_tables/`](../manuscript/data/paper1/derived_figure_tables/)
- Table provenance: [`../manuscript/data/paper1/FIGURE_DATA_MANIFEST.json`](../manuscript/data/paper1/FIGURE_DATA_MANIFEST.json)

## Regeneration

The rendered figures, derived tables, and provenance manifests are tracked in
Git. Exact end-to-end regeneration additionally requires the private,
hash-pinned Q1 holdout manifest and row-level confirmatory journals identified
by `FIGURE_SPEC.json`; a public clone without those bytes can inspect the
tracked package but cannot reconstruct item-level tables from scratch.

```bash
python scripts/generate_q1_paper_figures.py
```

The generator validates every expected frozen source hash before loading data,
reconciles confirmatory aggregates against the canonical result, materializes
deterministic CSV tables, and emits every implemented figure as SVG, PDF, and
PNG. The notebook calls the same tested pipeline and contains no hand-entered
scientific value.

Historical figures previously present in the directory remain distinguishable
through the `historical_package` entry in `SOURCE_MANIFEST.json`.
