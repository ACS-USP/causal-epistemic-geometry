# Q3 Realizable Collective Utility Design Review

## Status

Q3 remains `NOT_RUN`. This review uses closed Q2 rows only as
`DEVELOPMENT_ONLY / POST_CLOSED_RESULT_PLANNING`. It performed no model
inference, generated no semantic trajectory, inspected no future evaluation
correctness, and allocated no future holdout.

The terminal design ruling is:

```text
Q3_FRESH_HOLDOUT_INSUFFICIENT
```

The ruling is not a negative Q3 result. It says that the repository does not
currently contain a sufficiently large, sufficiently fresh CRUXEval evaluation
population for the prespecified 3-point paired utility test. No router is ready
for prelock either: the closed-data tournament found opportunity but no route
met all frozen realization and stability criteria.

## 1. Why Q3 is scientifically unlocked

Q1 established safe, null-specific error-profile complementarity for the fixed
Qwen controller on CRUXEval. Q2 V4.1 then established relational geometry
within the fixed 31-controller atlas (`Q2_V4_1_G2`, `RS+`, `RT+`). Q2 OOS V2
completed the missing controller-held-out step: all 16 prospectively sampled
fresh-controller A0 row associations were positive, with forensic-clean status.

This satisfies the dependency in the original Q3 concept note: geometry can
predict semantic proximity for unseen controller identities inside the same
Qwen3-8B/CRUXEval/learned-rank-8 laboratory. It does not imply item-level
selectability, realized utility, cross-task transfer, or learned-subspace
specificity.

## 2. Exact claim candidates

Q3 must keep four objects separate:

1. **Opportunity:** some policies are correct where others fail.
2. **Selectability:** pre-ground-truth observables predict which policy will be
   correct.
3. **Realization:** a frozen selector beats a frozen development-selected
   single-policy champion on unseen families.
4. **Economics:** the gain survives calls, tokens, latency, invalidity and
   safety costs.

The strongest future one-call claim would be that a frozen, pre-generation
router improves family-level correctness over the development-selected best
single policy on a fresh evaluation population. Pair-oracle accuracy is never
that claim.

## 3. Item/family exposure ledger

The canonical CRUXEval output-prediction universe contains 800 item/families.
The complete release-safe ledger records IDs, revisions, prompt/reference
hashes, roles and exposure booleans without benchmark text.

| Exposure tier | Families | Meaning |
|---|---:|---|
| Closed Q2 development panel | 300 | Outcomes exist for all 47 candidate controllers |
| Tier A: globally untouched | 23 | No prior free generation, correctness scoring or inspected outcome |
| Tier B: no exact candidate-policy outcome | 500 | No outcome from the 47 Q2 candidate controllers, but other project exposure may exist |

There are no exact prompt-hash duplicate groups in the canonical provenance
ledger. CRUXEval uses one output-prediction record per problem, so item and
family are the same scientific unit here. The ledger is
[`ITEM_EXPOSURE_LEDGER.json`](../review/q3_realizable_utility_design/ITEM_EXPOSURE_LEDGER.json).

## 4. Fresh holdout availability

The preferred globally untouched pool has only 23 families. The broader Tier-B
pool has 500 families, but it is not globally untouched and carries
benchmark-level adaptation risk. Neither is large enough for the frozen power
criterion: at two rollouts, 3-point gain, paired discordance 0.20 and alpha
0.05, the first tested N reaching 80% planning power is 800 (estimated power
0.851). At N=500, planning power is 0.682.

No item was permanently assigned to Q3 and no correctness was opened from
either tier. See
[`FRESH_HOLDOUT_FEASIBILITY.json`](../review/q3_realizable_utility_design/FRESH_HOLDOUT_FEASIBILITY.json).

## 5. Development policy population

The compatible closed population comprises:

- 31 historical safe controllers × MEDIUM/STRONG = 62 policies;
- 16 OOS safe controllers × MEDIUM/STRONG = 32 policies;
- one historical baseline;
- 300 common items and two rollouts.

The model/tokenizer revision, panel identity/order, prompt semantics,
external-semantic-v3 parser, 4096-token cap and binary-error convention match.
The OOS campaign alone used the efficient repetition stop; those rows are
already terminal errors under the shared convention. Baseline was not rerun in
OOS, so the one historical baseline is used only under its verified common
panel/protocol provenance.

Private inputs were verified by SHA-256 before use:

- historical scores: `a6a9f4b4…f8a33f` (37,800 rows);
- OOS scores: `9f03d96d…f8a33f` (19,200 rows).

No raw generation text was read.

## 6. Oracle opportunity audit

Nested outer-fold bank selection shows real opportunity. The strongest
geometry-only opportunity bank was A0-maximin K=8 without baseline:

| Quantity | Development estimate |
|---|---:|
| Bank champion accuracy | 0.4633 |
| Committee-oracle accuracy | 0.6233 |
| Oracle headroom | 0.1600 |
| Minimum fold headroom | 0.1417 |
| Mean pair disagreement | 0.1083 |

Smaller banks retained less opportunity: A2-maximin K=2 showed 0.0767 oracle
headroom and A2-maximin K=4 showed 0.1117. These are diagnostic upper bounds,
not deployable gains.

## 7. Geometry-selected policy banks

K ∈ {2,4,8}, baseline included/excluded, and A0/A1/A2 maximin were evaluated
under five outer item folds. Shell choice used outer-training outcomes only;
at most one shell per controller entered a bank. The accuracy-qualified A0 rule
was intentionally fail-closed when fewer policies met the frozen competence
floor; partial fold configurations are ineligible for mechanism selection.

K=8 is the preferred *opportunity* size because it preserved the largest and
most stable oracle headroom. It is not an authorized future bank and does not
make an eight-call committee the preferred deployment mechanism.

## 8. Feature availability and leakage firewall

The Q3.0 one-call models used deterministic prompt structure and frozen
controller coordinates only. Prompt activations, hidden representations,
entropy and confidence would require a separately frozen label-free capture
and were not generated. Item IDs, references, program execution, correctness,
candidate answers and outer-test policy summaries are prohibited. The complete
boundary is in [`Q3_FEATURE_FIREWALL.md`](Q3_FEATURE_FIREWALL.md).

## 9. Route A — one-call pre-generation router

The tournament compared a regularized geometry-aware bilinear model, a
16-feature low-capacity nonlinear geometry-aware model, and a capacity-matched
geometry-blind control. All policy-bank, champion and preprocessing choices
were outer-training-only; hyperparameters were inner-fold-only.

The strongest complete point result used A2-maximin K=8 and the nonlinear
geometry-aware model:

| Quantity | Value |
|---|---:|
| Routed accuracy | 0.4850 |
| Cross-fitted champion accuracy | 0.4533 |
| Gain | +0.0317 |
| Oracle headroom | 0.1533 |
| Fraction realized | 0.2065 |
| Positive outer folds | 3/5 |
| Worst fold gain | −0.0333 |

It crossed the 3-point point-gain criterion but failed the 25% oracle-fraction,
4/5-fold and −0.02 worst-fold criteria. The A0 bilinear router gained 0.0200
and the A0 geometry-blind matched control gained 0.0250; neither was stable or
feasible. Geometry did not clearly outperform the matched blind representation.

Therefore Route A is not ready for prelock.

## 10. Route B — budgeted adaptive policy

After Route A failed, a prospectively staged amendment froze a baseline-first
ridge policy before Route-B analysis. It used prompt structure plus baseline
validity, evaluability and token count to decide whether to invoke one frozen
alternative.

| Quantity | Value |
|---|---:|
| Routed accuracy | 0.4883 |
| Cross-fitted champion accuracy | 0.4533 |
| Gain | +0.0350 |
| Baseline accuracy | 0.4550 |
| Invocation rate | 0.4367 |
| Expected generations | 1.4367 |
| Positive outer folds | 3/5 |
| Worst fold gain | −0.0083 |

The point gain is encouraging but fails the required 4/5 positive-fold rule.
It also uses more compute and does not yet have a complete empirical
self-consistency comparator at every exact budget. Route B is not ready for
prelock.

## 11. Route C — verifier-mediated committee

Mechanical typed plurality used no raw text or reference access.

| K | Accuracy | Champion | Gain | Positive folds | Equal-compute status |
|---:|---:|---:|---:|---:|---|
| 2 | 0.4667 | 0.4533 | +0.0133 | 4/5 | only one closed two-sample baseline realization |
| 4 | 0.4683 | 0.4533 | +0.0150 | 2/5 | unavailable; baseline has only two rollouts |

Neither gain reaches 3 points. K=4 is additionally ineligible without a
credible four-call repeated-baseline comparator. Route C is not ready.

## 12. Cross-fitted development results

The independent unit was the item/family, not policy×item or rollout. Five
balanced hash folds were used outside and four inside. Both rollouts and all
policies stayed together. Development results answer feasibility only; their
bootstrap ranges are descriptive paired-item sensitivities, not Q3 confidence
intervals.

The tournament shows a useful distinction: oracle opportunity is substantial,
but robust selectability is not yet established. Route A's best geometry-aware
configuration, Route B and Route C all fail at least one prespecified
realization/stability condition.

## 13. Equal-compute controls

The frozen future comparator is the best single policy selected on development
data, with ties by evaluability, lower generated-token mean and lexicographic
ID. Route A uses exactly one answer generation plus any explicitly charged
feature prefill. Route B reports `1 + invocation rate` and compares to repeated
baseline at that expected rate. Route C must compare to self-consistency or
repeated baseline at K calls; the closed data support only two baseline
rollouts, which blocks a fair K=4 claim.

## 14. Primary estimand and inference

The proposed future endpoint remains

```text
Delta_route = mean_family[
  correctness(router-selected policy)
  - correctness(frozen development-selected champion)
]
```

Invalid and unevaluable outputs are incorrect. Missing rows block completion.
The preferred inferential form is a one-sided paired family-level randomization
test, with a separately simulation-calibrated paired interval. One primary
contrast would be frozen; secondary controls would use Holm adjustment within
their declared family. The uncalibrated compound Q2 item bootstrap is not
reused.

## 15. Power and compute

Planning used 100,000 replicates per grid cell for gains 1/2/3/5 points,
N=200–2400, discordance 0.05–0.40 and R∈{1,2}. At the conservative 0.20
discordance and R=2:

| N | Power for +3 points | Expected 95% half-width |
|---:|---:|---:|
| 300 | 0.498 | 0.0357 |
| 500 | 0.682 | 0.0277 |
| 800 | 0.851 | 0.0219 |
| 1,200 | 0.950 | 0.0179 |

Using the validated Q2 OOS collection rate and charging 10% for a separate
label-free prefill, a one-call N=800/R=2 campaign is estimated at 0.86/0.98/1.10
Spark-1 hours (P50/P80/P95), 1,600 semantic trajectories. This runtime is
small; item supply, not GPU time, is the blocking resource.

## 16. Recommended prospective Q3 protocol

Do not prelock or execute Q3 on the current CRUXEval inventory. A future review
should first identify a legally usable, family-disjoint evaluation population
of at least 800 units under the frozen power criterion, without opening its
outcomes. Separately, Route A/B selectability needs stronger development
stability; adding a holdout alone does not repair that weakness.

If both blockers are resolved, retain the simplest order:

1. one-call geometry-aware router with a matched geometry-blind control;
2. only then baseline-first adaptation at equal expected compute;
3. committee only with a complete equal-call baseline.

No current controller bank, mechanism, feature capture or holdout is frozen for
execution.

## 17. Reviewer and fragility audit

- **Main strength:** the opportunity audit uses 47 closed controller identities
  and strict item-level nested cross-fitting.
- **Main weakness:** 300 development items are modest for policy/bank/model
  selection; fold champions and gains are unstable.
- **Freshness:** only 23 items are globally untouched; the 500-item broader
  pool has prior project exposure and still misses the frozen power target.
- **Geometry attribution:** the best A0 geometry-blind control exceeded the A0
  bilinear router; current data do not show that geometry improves routing.
- **Compute fairness:** Route B/C require stronger equal-compute baselines.
- **Specificity:** matched-random-rank-8 subspace specificity remains unknown.
- **Scope:** all planning is Qwen3-8B + CRUXEval + learned-rank-8; no task,
  model or universal utility claim is available.

## 18. Repository and resource state

- Q3 semantic trajectories: **0**
- new Qwen inference: **0**
- prompt-activation capture: **0**
- fresh evaluation correctness inspected: **NO**
- fresh holdout permanently allocated: **NO**
- development-only outcomes: Q2 V4.1 historical scores and Q2 OOS V2 scored
  rows, exact hashes listed above
- preferred policy-bank size: **K=8 for opportunity only**
- preferred mechanism: **none ready for prelock**
- primary endpoint: family-level routed correctness minus frozen
  development-selected champion correctness
- fresh families available: **23 Tier A / 500 Tier B / 800 required**
- projected N=800 one-call runtime: **0.86/0.98/1.10 h P50/P80/P95**
- Q1/Q2 classifications changed: **NO**
- personal handbook changed: **NO**
- paper workspace changed: **NO**
- Spark 1 GPU used: **NO**
- Spark 2 used: **NO**
- RunPod used: **NO**

`Q3_FRESH_HOLDOUT_INSUFFICIENT`
