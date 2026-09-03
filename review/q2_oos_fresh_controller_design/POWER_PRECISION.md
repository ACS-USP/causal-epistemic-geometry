# Q2 fresh-controller power and precision

This CPU-only planning analysis used the closed Q2 V4.1 aggregate A0
association (rho=0.5638183484) as a declared planning input.  It did not read
prior item-level semantic outcomes.  Each synthetic cell used 240 independent
300-item panels, two rollout blocks, two shells, 31 fixed reference columns,
and complete fresh-row dependence.  Planning used 999 row permutations (720
for K=6); the future frozen test uses 50,000 maps when K>=10.

| K | Null FPR | Power, 50% prior | Power, 75% prior | Power, full prior | Full-effect 95% sampling width | Full-effect all-LOFO-positive | Future trajectories |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.058 | 0.883 | 0.996 | 1.000 | 0.184 | 1.000 | 7,200 |
| 8 | 0.067 | 0.967 | 1.000 | 1.000 | 0.156 | 1.000 | 9,600 |
| 10 | 0.058 | 0.979 | 1.000 | 1.000 | 0.136 | 1.000 | 12,000 |
| 12 | 0.029 | 0.996 | 1.000 | 1.000 | 0.130 | 1.000 | 14,400 |
| 16 | 0.042 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 19,200 |

The high planning power is not interpreted as 310 independent dyads.  The
randomization unit is one of K complete fresh rows.  K=6 and K=8 are rejected
prospectively because fewer than ten independent fresh-controller vertices
provide a weak identity-generalization sample even when their dyads are many.
K=10 is the smallest design satisfying every frozen utility criterion.  K=12
and K=16 pass but add 2,400 and 7,200 trajectories for modest precision gains.

As a stress test, an independent low-rank controller-by-item profile nuisance
was added on top of the 75%-prior latent relation.  With moderate nuisance
(scale 0.5), K=10 retained 0.946 permutation power and 0.996 all-LOFO-positive
frequency; achieved mean true rho fell to 0.216.  With deliberately severe
nuisance (scale 1.0), achieved mean true rho fell to 0.079 and K=10 power to
0.371.  This is appropriately reported as sensitivity to a much smaller
effective relation, not hidden by the dyad count.

For K=10, a 19-candidate reserve gives P(at least 10 safe)=0.967 at safety
rate 0.70 and 0.996 at the historical 0.775 rate.  Therefore the prospective
recommendation is K=10, 19 candidates, one stream, no redraw.
