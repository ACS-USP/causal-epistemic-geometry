# Q2 V3 four-family power and precision report

Status: synthetic/resampled DEVELOPMENT planning only.

## Design

The CPU simulation uses four families, two directions per family, two shells,
two independent Bernoulli error rollouts, and the canonical product estimator.
Item, controller, family, shell, and dyad dependence are retained. Dyads are
recomputed from shared controller errors and are never treated as independent.

Grid:

- nominal relational rho: 0, 0.10, 0.20, 0.25, 0.30, 0.40;
- N: 200, 300, 400;
- dependence regimes: balanced, heterogeneous, one weak family;
- 240 repetitions per rho/regime, nested over N;
- exhaustive 384-map QAP in every repetition;
- 200 item-cluster bootstrap replicates per synthetic panel.

No historical V3 outcome or geometry matrix was loaded. Synthetic M0/M1/M2
labels exercise the frozen multiplicity structure only.

## False-positive validation

Averaged over the three dependence regimes, the rho-zero global max-QAP
rejection rates were 0.065, 0.051, and 0.051 for N=200, 300, and 400. The
metric-attributed M2 maxT rates were 0.018, 0.010, and 0.010. The full
relational gate rates were 0.018, 0.010, and 0.008. The global N=200 value is
somewhat high in this finite simulation; the N=300/400 averages are near 0.05,
while metric attribution and the composite rule are conservative.

This validation does not license asymptotic or uncorrected p-values. The exact
384-map maxT remains required.

## Meaningful-effect behavior

Rates below are averages over the three dependence regimes.

| nominal rho | N | M2 maxT reject | 3/4 families + | 4/4 families + | 8/8 LODO + | full gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 200 | 0.089 | 0.793 | 0.422 | 0.572 | 0.083 |
| 0.25 | 300 | 0.085 | 0.817 | 0.446 | 0.606 | 0.082 |
| 0.25 | 400 | 0.085 | 0.863 | 0.486 | 0.654 | 0.082 |
| 0.30 | 200 | 0.115 | 0.843 | 0.521 | 0.663 | 0.113 |
| 0.30 | 300 | 0.136 | 0.872 | 0.533 | 0.708 | 0.131 |
| 0.30 | 400 | 0.147 | 0.881 | 0.558 | 0.724 | 0.146 |
| 0.40 | 200 | 0.274 | 0.897 | 0.656 | 0.772 | 0.271 |
| 0.40 | 300 | 0.299 | 0.913 | 0.686 | 0.815 | 0.294 |
| 0.40 | 400 | 0.313 | 0.940 | 0.717 | 0.847 | 0.306 |

The `full gate` column includes rho >=0.25, metric maxT p<=0.05, bootstrap
lower>0, at least 3/4 positive families, both shells positive, and 8/8 positive
LODO estimates.

Even for a large true rho of 0.40, N=400 reaches only about 31% full-gate
probability. The core limit is the eight directions/four families and the
coarse family-preserving null, not item count.

## N and precision

At rho 0.40, mean estimation RMSE across regimes fell from 0.200 (N=200) to
0.175 (N=300), a 12.3% reduction, then to 0.162 (N=400), only another 7.3%.
Median item-bootstrap width fell from 0.332 to 0.309 (7.2%), then to 0.297
(3.7%). Similar diminishing returns occur at rho 0.25 and 0.30.

N=300 is the precision/cost elbow. It improves estimation of each semantic
distance but cannot create new controller directions, dyads, or family units.
N=400 therefore does not repair the relational-power deficit.

## Family consistency and LODO

The 4/4 family rule is a new uniformity requirement and costs roughly 20–40
percentage points relative to 3/4 at meaningful effects. Since max-QAP,
bootstrap, both-shell and LODO gates already guard concentration, the proposed
family criterion is at least 3/4 strictly positive. It retains the old
one-family tolerance without pretending that 75% equals the historical 80%.

LODO serves a different role: no one base direction may determine the sign.
The proposed rule remains 8/8 strictly positive. At rho 0.40/N=300 it passed
about 82% of repetitions versus about 91% for 7/8; the burden is material but
scientifically aligned with its robustness purpose.

## M2 superiority

The point margin remains 0.10. In the supplementary dependent simulation, a
true M2 advantage of exactly 0.10 satisfied both point margins and paired
precision criteria in 13.1%, 19.9%, and 26.5% of repetitions at N=200, 300,
and 400. Even a true 0.15 advantage reached only 28.2%, 40.3%, and 50.8%.
This is a second independent indication that G3 discrimination is weak with
four families. The margin is not relaxed for convenience.

## Radial inference

The historical four-family sign-flip minimum is 1/16=0.0625, so a p<=0.05
gate is impossible. The proposed radial gate uses median direction effect >0,
at least 7/8 positive directions, all 4 family medians positive, and a 10,000
item-cluster bootstrap lower bound >0. The 16-point family sign-flip p-value is
reported diagnostically only. Synthetic false-positive rate was about 4–5%; at
a true effect of 0.05, power was 0.799/0.842/0.864 for N=200/300/400.

## Conclusion

N=300 is the best item-level design among the candidates, but the exact
four-family experiment is not sufficiently informative for its intended
G2/G3 discrimination. The recommendation is:

`Q2_V3_FOUR_FAMILY_DESIGN_UNDERPOWERED`.

