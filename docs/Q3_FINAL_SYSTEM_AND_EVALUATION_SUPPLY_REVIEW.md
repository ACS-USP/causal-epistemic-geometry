# Q3.3 Final-System Freeze and Evaluation-Supply Review

## 1. Immutable Q3.2 result

`Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING` remains unchanged. Part A was `GEOMETRY_BANK_SELECTION_SUPPORTED`: the A0-maximin routed gain was +0.0400, at percentile 0.986328 among 512 competence-matched banks. Part B was `CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED`: true-coordinate gain was +0.011875, with 3/5 positive folds. These are closed-data DEVELOPMENT results; Q3 remains `NOT_RUN`.

The Q3.2 “outcome-optimized upper bound” bounds bank opportunity/construction only. It is not an upper bound on routed accuracy, and no historical result is changed by this editorial clarification.

## 2. Prospective development closure

The base precheck and additive steer were frozen and pushed before Q3.3 analyses. `DEVELOPMENT_PHASE_CLOSED = YES`: no further architecture, hyperparameter, or open-ended bank tournament is permitted on the 300-family panel.

## 3. Final candidate deployment system

Status: `DEVELOPMENT_SELECTED_NOT_EVALUATED`.

1. `V4_DIRECTION_31_MEDIUM` — vector `19ca2a98ff342a4565f6ea84b9c1b48fd44d1b8459001228df333403cfe5ca48`
2. `V4_DIRECTION_10_MEDIUM` — vector `edd580f19b8b3cdd1c742c0baf992fd4bec066bb37a78e06fd2dba32a117855a`
3. `V4_DIRECTION_32_MEDIUM` — vector `ae04e9c3d46915436de7ebc8298f4a8635e4ea78111d6e308ebef324d8bfea38`
4. `Q2_OOS_V2_DIRECTION_13_MEDIUM` — vector `88bcb1b94d50caec05ee96165072a29712d972ca1523f039c74f576db2845a87`
5. `V4_DIRECTION_19_MEDIUM` — vector `8baa1867a7be252705f3751df19c9996f9104bd3607fe815e86c392141631e2d`
6. `Q2_OOS_V2_DIRECTION_03_MEDIUM` — vector `d9e224d04b414926116a5d1a6c95ff2c482dcbd2ce3fec0dde7d3f821836ba32`
7. `Q2_OOS_V2_DIRECTION_16_MEDIUM` — vector `f991620c3ee6a0bec9d18e8053321133bb51418b7f7e50c31f0bd2f1ea826cce`
8. `V4_DIRECTION_35_STRONG` — vector `e4909caac7fc944c94e0e77cdd0670f649188e55467a38a74a9632d0c1675934`

The frozen champion is `V4_DIRECTION_02_MEDIUM`. The router uses the ordinary unsteered layer-27 block-input representation at the final non-padding prompt token, PCA dimension 8, a rank-2 learned-policy-identity interaction, L2=1, 400 deterministic full-batch Adam steps at learning rate 0.03, and seed 2026090511. Those values are the componentwise fold consensus (8/2/1), not the best apparent full-panel fit. Controller coordinates constructed the portfolio but are not router inputs.

Deployment is one-call: the unsteered prefill reaches L27, the frozen router chooses one policy, and sustained-current-token steering continues from the same prefill into one answer decode. The private fitted parameters are hash-pinned as `269dc116c70b64dd47cf59340b07dbe558ec8c0f13be8410ed97017310ebad3d` and are not stored in Git.

## 4. Tier-B exposure-severity audit

Of 500 candidate families: A=0, B=11, C=0, D=177, E=0, F=312. Thus zero are confirmatory-eligible and only 11 are eligible for bounded internal validation. No Tier-B correctness was opened, no IDs were allocated, and the public audit contains only counts and set hashes. Route I is not scientifically or numerically viable.

## 5. Power and precision

The model-free tournament used 20,000 null panels and 10,000 alternative panels per cell over N={23,100,250,400,500,800,1000,1200}, R={1,2,4,6,8}, gains +1/+2/+3/+4/+5 pp, variable discordance, family difficulty/effect heterogeneity, rare harm, and independent versus common-uniform seeds.

For the selected independent-seed N≥800 regime, the paired family-level studentized t test had maximum FPR 0.05445 and minimum 95% interval coverage 0.946. Common seeds are rejected for distinct autoregressive policies. At +3 pp under the conservative combined scenario: N=800/R=2 power=0.742; N=800/R=4 power=0.942; N=1000/R=2 power=0.823. N=1000/R=2 is preferred because it clears 0.80 while using more independent families and only 4,000 maximum trajectories.

## 6. Frozen primary inference

The future primary is Claim U: `Delta_utility`, the family-weighted mean of rollout-mean correctness(router-selected policy) minus correctness(frozen champion), with invalid/unevaluable outputs incorrect and missing rows blocking completion. The test is one-sided paired family-level studentized t; the interval is the two-sided 95% family-level t interval. Family is the independent unit and rollout is nested replication.

If router and champion select the exact same policy, one frozen-seed generation may back both experimental roles and contributes paired difference zero. Distinct policies use independent seeds. This preserves the estimand without pretending divergent autoregressive paths are common-random-number coupled.

## 7. Utility, full-bank, and portfolio-attribution designs

- Design U (primary): N×R×2 maximum trajectories; one deployed answer plus an experimental champion comparator.
- Design F (diagnostic): N×R×8 trajectories; needed only for oracle/headroom and full counterfactual policy matrices.
- Design P (optional): compare the learned bank with S random-bank systems. Banks share families and may share policies, so they are not IID controller/dyad observations.

For N=1000/R=2, U is 4,000 trajectories (P50/P80/P95 2.03/2.30/2.59 Spark-1 hours, about 79506 generated tokens and 26.7 MB). F is 16,000 trajectories, not a requirement for Claim U. P with S=20 is at most 44,000 trajectories and remains a separate research cost, never a 20-call deployment claim.

## 8. Claim precedence and paper strategy

- U: `DEVELOPMENT_SUPPORTED_NOT_CONFIRMATORY` — future primary.
- P: `DEVELOPMENT_SUPPORTED_NOT_CONFIRMATORY` — retain as development evidence.
- G: `NOT_SUPPORTED`.
- C-OOS: `NOT_SUPPORTED`.

Recommended paper strategy A: confirm realized one-call utility and keep portfolio geometry as development evidence. Strategy B would require a separately calibrated and much larger multi-bank campaign; it is not needed to establish U and should not be co-primary by default.

## 9. Fresh-instrument source audit

The official [CRUXEval repository](https://github.com/facebookresearch/cruxeval) documents 800 short Python function/input/output samples, execution-derived references, output prediction, and an MIT license. It is the task anchor but its current 800 families have been exhausted by prior project exposure. [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) officially supports test-output prediction, but the project’s pinned instrument has only 182 independent families and is already a closed negative Q1 development path.

[BigCodeBench](https://github.com/bigcode-project/bigcodebench) supplies 1,140 Apache-2.0 executable code-generation tasks; [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) has a 500-problem official test split; and [HumanEval](https://github.com/openai/human-eval) has an MIT executable harness. All change the response contract to code generation and carry substantial public-training exposure. [Project CodeNet](https://github.com/IBM/Project_CodeNet) and [CodeContests](https://github.com/google-deepmind/code_contests) offer thousands of natural programs/problems, but input-domain reconstruction, mixed third-party provenance, source exposure, and family definition make the hybrid route less clean.

## 10. Recommended supply route

Route II is selected: a genuinely fresh, separately generated deterministic program-execution instrument. Route III is rejected because 11 eligible Tier-B families cannot de-risk the system and opening them would add adaptation pressure. Proposed supply is 1,600 independent families: 300 qualification, 1,000 confirmation, 300 untouched reserve. No IDs, seeds, items, or permanent split allocation were generated in Q3.3.

## 11. Generator and evaluator design

Use a model-free typed restricted-Python AST grammar. Every family has an independently generated program skeleton and one allocated input; another input to the same program is nested, not a new family. Allow bounded mutation/aliasing, branching, loops, nested control flow, containers, pure helpers, and depth-bounded recursion. Forbid filesystem, network, imports, reflection, dynamic code, randomness, ambient state, and unbounded computation.

Reference outputs require exact agreement between a restricted-AST interpreter and sandboxed pinned CPython, repeated twice under deterministic locale/hash settings, 2-second timeout, 256 MB memory, bounded recursion/iterations/container size/integer magnitude. Deduplication combines canonical AST, token MinHash, and private multi-input behavioral signatures. Generator namespaces and grammar productions are split before generation. Nonscientific fixtures are permanently excluded.

## 12. Qualification and training contract

The 300 qualification families may test evaluator determinism, parser roundtrip, validity/evaluability, champion difficulty, frozen-bank opportunity, independence, near duplicates, runtime, and repetition. They cannot be reused for confirmation, and routed gain is not a qualification gate. The final router remains trained only on the original 300 CRUXEval development families. Refitting on a new-development split would create a new candidate system and require a new untouched confirmation split; reopening architecture selection would be a new development phase.

## 13. Safety, economics, and controls

A future positive claim requires positive routed utility, commitment validity and semantic evaluability no worse than champion by 3 pp, one-call accounting, token/latency and routing-concentration reporting, and no frozen pathological failure mode. Primary comparator is the frozen champion. Baseline, random routing in the same bank, prompt-only control, and optional matched-random bank are secondary; oracle/headroom is diagnostic only.

## 14. Reviewer/fragility audit

The release summary records fourteen attacks with the fragility, cheapest discriminative check, and stop condition. The central ones are development overfit, synthetic shortcuts, family leakage, comparator leakage, invalid matched-seed coupling, and confusing one-call deployment with evaluation cost. The cheapest decisive check is the untouched, independently generated family-level confirmation. Q3 should remain an extension to the Q1/Q2 paper until that check exists.

## 15. Repository and resource state

- New semantic trajectories: 0
- New Qwen forwards: 0
- Fresh evaluation outcomes inspected: NO
- Q3.2 classification changed: NO
- Q3: `NOT_RUN`
- Spark 1 GPU used: NO
- Spark 2 used: NO
- RunPod used: NO
- Final items/seeds generated: 0
- Holdout permanently allocated: NO

`Q3_FRESH_INSTRUMENT_DESIGN_READY_FOR_PRELOCK`
