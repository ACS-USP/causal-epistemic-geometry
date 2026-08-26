# Q2 V3 four-family QAP derivation

Status: prospective CPU-only statistical design; no semantic outcome used.

## Historical five-family implementation

The historical generator in
`src/epistemic_geometry/experiments/q2_v3.py::exact_family_qap_permutations`
enumerates every pair

\[
(\pi,s)\in S_5\times\{0,1\}^5.
\]

For target family slot \(f\) and source-location slot \(\ell\), the induced
controller reindexing is

\[
(f,\ell)\mapsto(\pi(f),\ell\oplus s_f).
\]

Composition lets family permutations move the swap coordinates, so the acting
group is the wreath/semidirect product

\[
S_5\ltimes(\mathbb Z_2)^5,
\qquad |G_5|=5!2^5=3{,}840.
\]

The implementation enumerates `permutations(range(5))` crossed with
`product((0,1), repeat=5)`. It therefore preserves each prompt/execution pair
as one family block, permits its internal orientation to swap, preserves shell
identity, and applies the same block map to both shells. Geometry is reindexed
relative to the fixed semantic-distance/family-incidence structure. The
outcome matrix and shell labels are not permuted.

The scientific null is: after preserving the five two-location controller
blocks and both radial shells, the alignment between conceptual family blocks
in pre-outcome geometry and semantic error relations is arbitrary. The test is
not a null in which arbitrary directions may be regrouped into new families.

## Four-family analogue

The same null for four fixed source-qualified family blocks gives

\[
G_4=S_4\ltimes(\mathbb Z_2)^4,
\qquad |G_4|=4!2^4=384.
\]

The mapping is applied identically to MEDIUM and STRONG. The eight direction
slots, conceptual family incidence, prompt/execution pairing, and shell
identity remain intact. This is the selected exhaustive universe.

The implementation is
`src/epistemic_geometry/experiments/q2_v3_four_family.py::family_qap_mappings`.
Tests establish 384 unique maps and reproduce 3,840 when called with five
families.

## Why 2,520 and 105 are rejected

The count

\[
\frac{8!}{(2!)^4}=2{,}520
\]

assigns eight labeled directions to four labeled pairs of size two. Dividing
by \(4!\) gives

\[
\frac{8!}{(2!)^4 4!}=105
\]

unlabeled pair partitions. Both universes permit a prompt-boundary direction
from one qualified family to be paired with an execution-boundary direction
from another. They thereby redefine which four dyads are “within family,”
which 24 are primary cross-family dyads, and what a family summary means. That
tests whether *some post-hoc pairing* aligns with the outcome, not whether the
four frozen conceptual families carry relational information. Quotienting
family names (105) removes labels but does not repair this scientific change.
Neither universe is used.

## Statistic and shell/family weighting

For each metric and shell, a family-specific Spearman rho is computed over the
12 cross-family dyads incident to that family. The shell statistic is the
arithmetic mean of the four family rhos, and the aggregate statistic is the
arithmetic mean of the two shell statistics. Average ranks resolve ties.

Every family has equal weight. Each of the six unordered family pairs has four
direction-level dyads per shell and appears symmetrically in both endpoint
family summaries. Because correlation is nonlinear, the statistic is not
identical to averaging six separate four-point correlations; that alternative
would be extremely discrete and is not adopted.

## P-value convention and finite grid

The identity map is included. For a one-sided positive statistic \(T\),

\[
p=\frac{\#\{g\in G_4:T_g\ge T_{obs}\}}{384}.
\]

There is no `+1` because the exhaustive universe already includes identity.
The minimum is \(1/384=0.0026041667\). At alpha 0.05, at most 19 null values
may meet or exceed the observed value:

\[
19/384=0.0494792\quad\text{passes},\qquad
20/384=0.0520833\quad\text{fails}.
\]

Alpha remains 0.05.

## Multiplicity hierarchy

The selected procedure is omnibus existence followed by corrected metric
attribution:

1. For each map compute \(T_{M0},T_{M1},T_{M2}\) and
   \(T_{max}=\max_m T_m\).
2. Test the global observed \(\max_m T_m\) against the exhaustive max-statistic
   null.
3. Attribute a metric only when its single-step maxT p-value, computed against
   that same three-metric maximum null, is at most 0.05.
4. Report all raw and corrected p-values, including metrics that fail.

This directly answers “does any frozen geometry carry a relation?” before
“which geometry?”, while preventing post-omnibus uncorrected attribution.

For M2 superiority, retain the two paired contrasts \(M2-M0\) and \(M2-M1\).
Use the same 384 maps and a two-hypothesis maxT step-down procedure, together
with the independently required point margin of at least 0.10 and paired
item-bootstrap lower bounds above zero. No uncorrected superiority claim is
allowed.

