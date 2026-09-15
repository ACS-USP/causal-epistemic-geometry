# D75 late-exposure ablation — outcome-free planning review

**Status:** `NO_GO_UNRESOLVED_OPERATING_CHARACTERISTICS`  
**Scope:** synthetic planning only; no fresh scientific latent, Qwen prompt,
forward pass, generation, response, correctness value, or historical journal
was read.

## Proposed estimand

The proposed study would use a new population under the fixed Q1 generator
law, a 4096-token ceiling, and two rollouts per latent. Its three policies were:

1. `BASELINE`: no D75;
2. `D75_FULL`: D75 on every token prediction;
3. `D75_OFF_AFTER_2048`: D75 through the fixed first 2048 token predictions,
   then off.

For every latent, including trajectories that finish before token 2048, the
primary contrast would be

\[
Z_i=\frac{1}{2}\sum_r
(Y_{i,r,\mathrm{OFF\_AFTER\_2048}}-Y_{i,r,\mathrm{FULL}}).
\]

It tests a narrow claim: whether **additional** D75 exposure after the fixed
boundary reduces unconditional delivered correctness. It cannot distinguish
late timing from accumulated dose, establish a general property of D75, or
explain Q1's old cap interaction by itself.

## Fixed sensitivity analysis

`PLANNING_SENSITIVITY.json` evaluates every combination of:

- 96, 144, 192, 240, or 288 latents;
- early-stop correctness 0.4, 0.6, or 0.8;
- late-exposure harm 0, 5, 10, 15, or 20 points;
- a transparent shared-randomness sensitivity parameter of 0, 0.5, or 1;
- 1,000 simulations per cell, seed `20260915`.

The decision threshold is deliberately conservative: separate e-values of 40
for support or exclusion of a 10-point mean benefit from turning D75 off. The
shared-randomness parameter is not a prediction about Qwen. It represents the
unidentified degree to which paired post-boundary correctness remains coupled
after the paths diverge.

At early-stop correctness 0.6, the support probability for a 20-point harm is:

| Latents | Independent | Half shared | Fully shared |
| ---: | ---: | ---: | ---: |
| 96 | 0.22 | 0.34 | 0.56 |
| 144 | 0.38 | 0.58 | 0.93 |
| 192 | 0.48 | 0.78 | 0.99 |
| 288 | 0.68 | 0.91 | 1.00 |

At 288 latents, the probability of excluding a 10-point harm when the true
harm is only five points is 0.09, 0.32, and 0.94 across the same dependence
settings. The planning object therefore does **not** support treating 96
latents as a resolving experiment. Nor does it justify selecting 288 latents
by assuming the favorable dependence regime.

## Decision

No scientific schedule, seeds, or Qwen collection will be created for this
candidate. The proposed shared prefix is mechanically meaningful and the
prefix-limited hook has been implemented as an engineering prerequisite, but
its statistical resolution depends on an unobserved continuation coupling that
this pre-outcome planning exercise cannot establish.

Increasing the sample until the independent setting is adequate would multiply
the 4096-token workload while leaving the causal claim narrow. That is not a
justified next use of the compute budget. The candidate closes as
`NO_GO_UNRESOLVED_OPERATING_CHARACTERISTICS`; it is not evidence that late D75
exposure has no effect.
