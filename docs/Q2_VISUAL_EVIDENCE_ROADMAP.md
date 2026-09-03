# Q2 visual evidence package — implementation status

> The closed fresh-controller OOS validation now has a separate
> [Q2 OOS visual evidence package](Q2_OOS_VISUAL_EVIDENCE.md). The V4.1 figures
> below remain the historical fixed-bank package; the two evidence levels are
> intentionally not pooled.

Q2 V4.1 now has a publication-oriented visual package generated entirely from
frozen tracked aggregates and pre-outcome matrices. The implemented package is
`notebooks/q2_visual_story.ipynb`, with figures under
`manuscript/figures/paper1_q2/`, tables under
`manuscript/data/paper1_q2/`, and deterministic source/data manifests.

Implemented:

1. Explanatory Q1-to-Q2 intervention-laboratory overview.
2. A0/A1/A2 fixed-decile relational panels for MEDIUM and STRONG, calibrated
   to show moderate association together with broad conditional dispersion;
   raw-scatter and shared-scale hexbin alternatives remain available.
3. Full-sample association, separately displayed bootstrap summaries, and G3
   contrasts explaining `G2` rather than `G3`.
4. Per-direction STRONG−MEDIUM radial shape and total displacement for all 31
   directions in frozen order.
5. Secondary accuracy/C/D context.
6. Exact frozen QAP null and delete-one-controller robustness.

Deferred or rejected:

- A standalone 2-D controller map was rejected because projection distance is
  not primary; a small outcome-free PCA inset is sufficient.
- A separate angular/radial “polar system” synthesis was rejected as redundant
  and too easy to overread as a manifold theorem.
- A standalone bootstrap-shift diagnostic was deferred; the registered
  full-sample and bootstrap objects are kept visibly separate in Figure 3 and
  the unresolved mechanical explanation remains explicit.
- The post-hoc capped-output diagnostic remains notebook/table context only; it
  cannot alter retained rows, G2, RS+, or RT+.

The package never describes the 465 dyads per shell as independent, selects no
dramatic controller pair, and does not portray G2 as A2 superiority.
