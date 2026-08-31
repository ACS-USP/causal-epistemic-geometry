# Q2 visual evidence — internal design memo

Status: design decision recorded before implementation on
`research/q2-visual-evidence`.

## 1. Single visual idea

Q2 is a correspondence between two relational objects: distances fixed among
activation-steering interventions before semantic outcomes, and distances
later observed among the blind-spot profiles those interventions caused.
Angle asks *which profile pattern* is approached; the matched shell contrast
separately asks *how far* the profile moves.

## 2. Prerequisites for a new reader

The notebook must teach only four prerequisites before showing a result:

1. eight source directions define one fixed 8-D intervention laboratory;
2. 31 prospectively accepted directions sample that laboratory at MEDIUM and
   STRONG amplitudes;
3. `D_shape` compares centered itemwise error-propensity profiles using two
   rollout blocks;
4. A0, A1, and A2 are three pre-semantic-outcome definitions of intervention
   dissimilarity, not three discovered manifolds.

## 3. Minimum figure set

1. **From controllability to relational geometry** — explanatory overview
   combining the intervention laboratory and the pre-outcome/outcome firewall.
2. **Primary relational geometry** — six fixed-decile distribution panels
   (A0/A1/A2 by shell), showing median, IQR, and 10–90% blind-spot-rank
   intervals with the frozen Spearman associations. The full 465-dyad scatter
   remains a supplement and a shared-scale hexbin remains a calibration
   alternate, exposing both moderate tendency and broad conditional spread.
3. **Why the result is G2, not G3** — full-sample associations, separately
   displayed frozen bootstrap summaries, and the two superiority contrasts.
4. **Radial displacement in all 31 directions** — the frozen paired
   STRONG-minus-MEDIUM baseline-to-controller shape and total contrasts.
5. **Dependence-aware robustness and behavioral context** — QAP nulls and
   delete-one-controller ranges, with compact accuracy/C/D context kept
   secondary.

No additional plot is justified solely to increase package size. The post-hoc
generation diagnostic remains notebook text/table material unless a later
paper need makes it necessary.

## 4. Main paper versus supplement

- Main-paper candidates: Figures 1–4.
- Supplement candidate: Figure 5.
- The central empirical figure is Figure 2; Figures 3 and 4 are the two
  essential qualifiers a reviewer needs beside it.

If Q1 and Q2 together are limited to five scientific figures, retain Q2
Figures 2 and 4, then add Figure 3 only if the G2/G3 distinction is not fully
carried by the primary caption and results table.

## 5. Explanatory versus empirical

- Figure 1 is explicitly `EXPLANATORY_ONLY`; its arrows and shells are
  schematics, not measured trajectories.
- Figures 2–4 are empirical and use only frozen tracked matrices/aggregates.
- Figure 5 is empirical robustness/context, classified
  `SUPPLEMENT_CANDIDATE`.

## 6. Visual and inferential safeguards

- The 465 dyads are shown but never described as independent; inference is by
  controller-label QAP over 31 identities.
- Rank-rank displays match the Spearman estimand and avoid fitted linear lines.
- Raw full-sample rhos and frozen bootstrap medians/intervals occupy distinct
  visual lanes; no conventional centered error bar is implied.
- Controller labels carry no semantic interpretation. The intervention-lab
  projection, if shown, is a deterministic PCA of the frozen 8-D coefficients
  only and is labeled non-primary.
- The radial panel preserves the frozen controller order and does not sort by
  observed effect.
- `D_shape` is profile movement after removing mean shift; it is not accuracy,
  damage, utility, a global manifold, or a dose-response law.

## 7. Misleading or redundant candidates rejected

- A separate 2-D “controller map” is rejected as a standalone figure because
  it could make projection distances look primary; a small pre-outcome inset
  in Figure 1 is sufficient.
- A separate angular-plus-radial synthesis figure is redundant with Figures 1,
  2, and 4 and risks overclaiming a polar coordinate system.
- A regression line on raw pairwise points is rejected because the frozen
  claim is rank association, not a linear response law.
- A standalone bootstrap-shift diagnostic is deferred. The distinction is
  made directly in Figure 3 and the unresolved mechanical explanation remains
  explicit in notebook text.
- The post-hoc capped-output diagnostic is not part of the main story and does
  not alter any retained row or frozen classification.
