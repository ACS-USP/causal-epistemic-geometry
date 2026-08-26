# Q2 V3 — four-family statistical redesign

Classification:

`Q2_V3_FOUR_FAMILY_DESIGN_UNDERPOWERED`

This CPU-only sprint reconstructed the exact historical statistics and
redesigned every family-count-dependent gate for the four previously
source-qualified families. No model inference, RunPod, shell calibration,
geometry capture, semantic outcome, or Q3 operation occurred.

## What is now exact

- Families: the exact repository IDs are
  `CONTROL_FLOW_PATH_COVERAGE`, `MUTATION_ALIAS_CAUSALITY`,
  `LOOP_BOUNDARY_ACCOUNTING`, and `HYPOTHESIS_BRANCH_ELIMINATION`.
- Bank: 16 meaningful controllers and four fresh null controls; 21 conditions.
- Dyads: 24 cross-family pairs per shell, 48 primary pairs total, eight radial
  same-direction pairs.
- QAP: block-preserving
  \(S_4\ltimes(\mathbb Z_2)^4\), 384 exhaustive maps, identity included,
  p=`count/384`, maxT across M0/M1/M2, alpha 0.05.
- Family rho: 12 incident dyads per family/shell, equal family weight, equal
  shell weight.
- Family consistency: at least 3/4 summaries strictly positive.
- Identifiability: effective rank >=4.0; family leverage <=0.40; radial nuisance
  R²<=0.10; angular range >=0.20; condition number <=30; remaining amplitude,
  cosine and null-matching gates unchanged.
- LODO: all 8/8 aggregate estimates strictly positive.
- M2 superiority: both M2-M0 and M2-M1 >=0.10, paired bootstrap lower bounds
  above zero, and corrected two-contrast step-down maxT QAP p<=0.05.
- Radial: median >0, at least 7/8 directions positive, all four family medians
  positive, and item-bootstrap lower>0. The impossible four-family exact
  p<=0.05 sign-flip requirement is removed rather than alpha being changed;
  its 16-point p remains descriptive.

## N and trajectories

The selected planning N is 300. The original 200-item order is preserved and
100 items are added by an outcome-independent hash order after inherited
allocation exclusions. The exact candidate order hash is
`d1853656e8320757aa149b8dfe71a661a875fa2127c85b933d367e544bd17392`.

\[
21\times300\times2=12{,}600
\]

semantic trajectories would be required.

## Why the experiment should not be frozen

The panel expansion improves item-level estimation but does not add controller
directions or family units. Across dependent simulations, a large nominal rho
of 0.40 produced only about 0.30 M2 maxT/full-gate probability at N=300 and
about 0.31 at N=400. A true M2 superiority margin of exactly 0.10 satisfied
both effect and precision requirements only about 0.20 of the time at N=300.
The design therefore cannot reliably distinguish G2 from G3 at scientifically
meaningful effects.

The clean action is to stop. An Amendment 2 was not drafted because the
authorization made sufficient informativeness a prerequisite. A later redesign
needs more independent controller/family structure, not simply more Class-C
items or weaker criteria.

## Evidence boundary

- new model inference: NONE;
- RunPod: 0;
- correctness inspected: NO;
- shell calibration: NOT RUN;
- M0/M1/M2: NOT RUN;
- semantic V3 outcomes: 0;
- Q1: unchanged;
- Q3: NOT RUN.

Independent design audit:
`Q2_V3_FOUR_FAMILY_DESIGN_AUDIT_CLEAN`.

