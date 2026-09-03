# Q2 OOS V2 visual evidence

Status: `Q2_OOS_V2_VISUAL_EVIDENCE_READY_FOR_REVIEW`

This package visualizes the closed, independently audited fresh-controller validation:

- primary: `Q2_OOS_V2_A0_PASS`;
- forensic: `Q2_OOS_V2_FORENSIC_CLEAN`;
- independent unit: one prospectively sampled, safety-conditioned fresh controller;
- result: 16/16 controller-level A0 associations were positive under the frozen exact sign test.

The package does not alter any frozen scientific value. It contains no raw generation, prompt, benchmark text, item-level correctness, or private error array.

## Read this first

The [OOS visual-story notebook](../notebooks/q2_oos_visual_story.ipynb) teaches the experiment from the fixed 31-controller atlas to the controller-level statistic, exact sign test, secondary analyses, and claim boundary.

The main figure is [fresh-controller generalization](../manuscript/figures/paper1_q2_oos/figure_q2_oos_fresh_controller_generalization.svg). Its central panel displays MEDIUM, STRONG, and frozen equal-shell associations for all 16 fresh controllers in prospective order. The display is not outcome-sorted, and the 496 fresh×old dyads are not presented as IID observations.

## Figure set

| Figure | Scientific role |
|---|---|
| Main | Prospective design, all 16 primary controller-level associations, global descriptive geometries, and fresh×fresh secondary |
| S1 | MEDIUM versus STRONG association for each fresh controller |
| S2 | Controller-level distribution and all 16 leave-one-controller-out sensitivities |
| S3 | Fresh×fresh pairwise geometry, explicitly secondary only |
| S4 | Efficient-termination runtime audit, operational only |
| S5 | Post-hoc item-bootstrap diagnostic using non-CI terminology |

Every figure is stored as SVG, PDF, and PNG under `manuscript/figures/paper1_q2_oos/`.

## Item-bootstrap language

The post-closeout audit ruling is `Q2_OOS_V2_ITEM_BOOTSTRAP_METHOD_NOT_CALIBRATED`. The archived 50,000-resample values are shown only as an **item-panel perturbation sensitivity distribution**. They are not presented as a conventional 95% confidence interval. No post-hoc replacement uncertainty method is used to strengthen the prospective sign-test result.

The controller-cluster interval remains a separate fresh-controller-population uncertainty summary conditional on the observed item panel.

## Reproduction

Public-clone figure reproduction uses the committed release-safe tables:

```bash
python scripts/generate_q2_oos_paper_figures.py --validate-only
python scripts/generate_q2_oos_paper_figures.py
pytest -q tests/test_q2_oos_publication_visuals.py
```

The private table derivation is deterministic but requires the sealed, hash-pinned `D_SHAPE.npz`:

```bash
python scripts/derive_q2_oos_figure_tables.py \
  --private-dshape /approved/private/path/D_SHAPE.npz
```

The expected private Dshape identity is recorded in `manuscript/data/paper1_q2_oos/FIGURE_DATA_MANIFEST.json`. It is intentionally not tracked in Git.

## Claim boundary

The strongest supported statement is that positive A0 relational alignment generalized across prospectively sampled controller identities within the same frozen Qwen3-8B, CRUXEval, layer-27, learned rank-8 intervention laboratory and fixed historical atlas.

This package does not establish cross-task or cross-model generalization, matched-random-subspace specificity, universal geometry, manifold structure, or Q3 utility.
