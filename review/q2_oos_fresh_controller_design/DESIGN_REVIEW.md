# Q2 fresh-controller out-of-bank design

This prospective experiment tests whether the closed Q2 V4.1 A0 relational
association generalizes to entirely fresh controller identities sampled from
the same frozen rank-8 intervention subspace.  It does not test new items,
tasks, models, subspaces, global smoothness, or utility.

The design uses **K=10** fresh safe controllers selected as the first safe
members of one immutable 19-candidate PCG64DXSM stream.  It never
optimizes the bank against old or new semantic outcomes.  Future semantic
inference, which is not authorized here, would add 12,000 trajectories
and reuse the sealed 31-controller reference outcomes and sealed baseline.

The primary metric is A0.  The statistic is the equal-weight mean of MEDIUM
and STRONG Spearman correlations over the complete K×31 cross block.  The
randomization permutes complete fresh-controller rows while keeping frozen
reference columns fixed; this directly tests fresh-identity alignment while
preserving row dependence.  A1 and A2 are predeclared secondary metrics and
do not create an A2-superiority claim.

The primary A0 replication requires positive aggregate rho, one-sided
fresh-row permutation p<=0.05, positive 2.5th-percentile item-bootstrap bound,
and positive aggregate rho in every leave-one-fresh-controller-out fold.
Leave-one-reference-out is descriptive stability, not an extra terminal gate.
