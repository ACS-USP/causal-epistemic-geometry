# Prospective four-family Q2 V3 redesign memo

Status: design sketch only; not frozen; not authorized; not executed.

The scientific question remains unchanged:

> After matching implemented intervention amplitude, does pre-outcome internal
> geometry prospectively predict which semantic blind spots differ across
> genuinely new causal controllers?

## Controller and trajectory structure

Four families, two source locations/directions, and two shells imply:

\[
4\times2\times2=16
\]

meaningful controllers.

Retain four secondary null controllers as two fresh span-orthogonal random base
directions deployed at both shells. They remain controls, not the primary
population.

On the unopened 200-item Class-C panel, baseline + 16 meaningful + 4 null
conditions with two independent rollouts imply:

\[
200\times21\times2=8{,}400
\]

semantic trajectories, 1,600 fewer than the unexecuted five-family design.

## Exact dyad counts

At one shell there are eight meaningful directions. Of the
\(\binom{8}{2}=28\) unordered pairs, four are within-family direction pairs and
24 are cross-family pairs. Across two shells, the primary shell-stratified
cross-family population contains exactly 48 dyads, versus 80 in the old design.
There are eight same-direction cross-shell radial pairs.

The naive dyad-count precision ratio is:

\[
\sqrt{80/48}=1.291,
\]

so an independence heuristic predicts about 29% wider standard errors. Dyads
share controllers and items, so this is a planning heuristic, not an inferential
claim. Family-level replication units also fall from five to four.

## Family-balanced inference and QAP

With two directions in each of four families, exhaustive balanced labeled
family assignments number:

\[
\frac{8!}{(2!)^4}=2{,}520.
\]

If the statistic is invariant to the names of the four families, quotienting
by \(4!\) leaves 105 unique unlabeled partitions. A redesign should enumerate
the exact applicable universe rather than use an asymptotic or arbitrary
10,000-permutation approximation.

The former “4/5 held-out families positive” rule has no meaning-preserving
integer translation. `3/4` weakens 80% to 75%; `4/4` is materially stricter.
A principal-reviewed replacement should therefore be designed anew. A
reasonable candidate is: all four family summaries nonnegative, at least three
strictly positive, positive aggregate effect, and no leave-one-direction-out
sign reversal. This is a proposal, not a frozen threshold.

## Identifiability and leverage

- Maximum direction rank becomes eight.
- The minimum effective-rank requirement should be re-derived around the four
  conceptual family units; `effective rank >=4` is the natural floor, not an
  automatically frozen rule.
- Equal family leverage is 25%. Preserving the old 1.5-times-equal-share
  principle would imply a 37.5% cap, but that tolerance must be checked by new
  synthetic leverage simulations before freeze.
- Leave-one-direction-out sensitivity has eight omissions rather than ten;
  the aggregate sign should not reverse in any omission.

## M0/M1/M2 comparison

Report every metric. Preserve the closed hierarchy in which M0/M1 are simple
geometry controls and M2 is tested only as an incremental finite-response
metric. Because there are only four family summaries and 48 primary dyads, an
M2-superiority threshold and its family-consistency rule require new
outcome-free precision simulation; the old five-family values must not be
copied mechanically.

M2 remains a finite-displacement output-response Jensen-Shannon pseudometric,
not a local JVP/Fisher/pullback metric.

## Assessment

Four families remain algebraically identifiable and permit exact family-QAP,
but they provide weaker family-level replication and roughly 60% of the old
primary dyad count. The redesign is scientifically viable only if the
principal accepts the narrower family basis and prospectively re-freezes the
family-consistency, leverage, rank, precision, and M2-superiority gates.

No semantic panel, shell calibration, geometry, or prediction lock was opened
in producing this memo.
