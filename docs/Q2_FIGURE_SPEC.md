# Q2 V4.1 publication figure specification

This package is a deterministic narrative layer over the frozen Q2 V4.1
aggregates. It does not rerun the scientific experiment or create a new
inferential result. Machine-readable figure metadata and provenance are in
`manuscript/figures/paper1_q2/FIGURE_SPEC.json` and `SOURCE_MANIFEST.json`.

## Figure 1 — From controllability to relational geometry

Classification: `EXPLANATORY_ONLY`.

The intervention laboratory is defined by eight source directions spanning a
fixed rank-8 subspace. Thirty-one prospectively accepted directions are tested
at matched MEDIUM and STRONG amplitudes, producing itemwise blind-spot profiles
on the same frozen model and 300-item panel. The inset is a deterministic 2-D
PCA of the frozen 8-D controller coefficients and is not the primary geometry;
no semantic outcome enters the projection. Q2 compares pre-semantic-outcome
intervention dissimilarities (A0/A1/A2) with the later observed centered
blind-spot distance. The diagram is conceptual and does not itself show an
empirical association or a Q3 utility result.

## Figure 2 — Primary relational geometry

Classification: `MAIN_PAPER_CANDIDATE`.

Intervention geometry predicts blind-spot geometry in both matched shells.
Each panel shows all 465 controller pairs as rank-transformed intervention
dissimilarity versus rank-transformed blind-spot-shape distance, matching the
frozen Spearman estimand; the black trace is a binned rank median, not a linear
fit. A0 (coordinate), A1 (covariance-whitened), and A2 (finite response) are
shown separately for MEDIUM and STRONG. Shell-specific Spearman correlations
are respectively 0.555/0.573, 0.555/0.558, and 0.443/0.440; all maxT-adjusted
QAP p-values are 0.00002. The 465 dyads share 31 controllers and are not
independent observations; inference permutes whole controller identities with
the same map in both shells.

## Figure 3 — Why the result is G2, not G3

Classification: `MAIN_PAPER_CANDIDATE`.

All three frozen geometries qualify, but A2 does not outperform A0 or A1.
Panel A shows shell-specific and shell-aggregated full-sample Spearman
statistics. Panel B separately shows the medians and percentile intervals of
the frozen 10,000-resample item-cluster bootstrap; these are not presented as
conventional intervals centered on Panel A because the frozen bootstrap
distribution is shifted relative to the full-sample statistic. Panel C shows
the two registered superiority contrasts: A2−A0 = −0.123 and A2−A1 = −0.115,
with bootstrap intervals entirely below zero and superiority maxT p=1.0 for
both. A2 therefore carries relational signal (`G2`) without the stronger G3
claim.

## Figure 4 — Radial result in all 31 directions

Classification: `MAIN_PAPER_CANDIDATE`.

Each lollipop is the frozen paired STRONG-minus-MEDIUM difference in
baseline-to-controller displacement for one direction, retained in frozen
manifest order. Blind-spot-shape displacement and total profile displacement
are positive in 31/31 directions. The observed medians are 0.0441 (`RS+`) and
0.0433 (`RT+`), with paired-swap p=0.00002 and p=0.00014 and frozen bootstrap
intervals [0.0300, 0.0511] and [0.0300, 0.0533]. This supports amplitude
ordering across the two tested shells, not global monotonicity, smoothness,
linearity, or a continuous dose-response law.

## Supplement S1 — Movement is not just accuracy

Classification: `SUPPLEMENT_CANDIDATE`.

Baseline and shell means are shown on three separate axes because accuracy,
C, and D have different units. Accuracy is 0.455 at baseline, 0.468 across
MEDIUM controllers, and 0.450 across STRONG controllers. Mean C is 0.011 and
0.034 and mean total profile movement D is 0.026 and 0.067 for MEDIUM and
STRONG. Stronger steering therefore produces much more movement and
complementarity without uniform accuracy improvement. These are secondary
behavioral summaries, not the primary relational endpoint and not evidence of
deployable collective utility.

## Supplement S2 — Dependence-aware robustness

Classification: `SUPPLEMENT_CANDIDATE`.

Panel A compares the observed shell-aggregated A0/A1/A2 associations with the
exact frozen 50,000-map controller-label QAP null distributions; the identity
map is first and the same map is used across shells and metrics. All raw and
maxT-adjusted p-values equal 0.00002. Panel B gives the full range of aggregate
associations after deleting each of the 31 controllers in turn, with the
full-sample value marked by a diamond. Every shell-specific delete-one value
remains positive. These diagnostics respect controller dependence and do not
treat 465 dyads as IID.

## Main-paper ranking

1. Figure 2 — scientifically indispensable and the central Q2 result.
2. Figure 4 — most visually memorable and indispensable for the independent
   radial claim.
3. Figure 3 — highest reviewer value for preventing a G2/G3 overclaim.
4. Figure 1 — strong explanatory option when Q2 cannot be explained adequately
   in the combined Q1/Q2 overview.
5. Supplement S2 — important inferential audit, but redundant with text for a
   space-constrained main paper.
6. Supplement S1 — useful context, but secondary to relational geometry.
