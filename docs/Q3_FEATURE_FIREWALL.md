# Q3 feature firewall

## Status

This is a prospective design boundary for Q3. Q3 remains `NOT_RUN`. Closed
Q1/Q2 outcomes used in Q3.0 are `DEVELOPMENT_ONLY` and
`POST_CLOSED_RESULT_PLANNING`; they are not Q3 evidence.

The deployment-time rule is simple: a feature is eligible only if the future
system can obtain it before ground-truth correctness is known, under the same
frozen compute accounting used for the comparator.

## Route A: one-call pre-generation router

| Feature family | Available before answer? | New model inference? | Answer needed? | Cost | Leakage ruling |
|---|---:|---:|---:|---|---|
| Character, line, token-free lexical, delimiter and keyword counts | Yes | No | No | Negligible CPU | Allowed |
| Python AST node and control-flow proxy counts | Yes | No | No | Small CPU; no execution | Allowed if parsing fails closed |
| Frozen controller coefficients | Yes | No | No | Static | Allowed |
| Frozen A0/A1/A2 controller descriptors | Yes | No | No | Static | Allowed; A2 is secondary provenance, not a correctness feature |
| Policy priors from the outer training partition | Yes | No | No | Static | Allowed only within the matching fold |
| Prompt hidden representation | Yes | Yes, label-free prefill | No | Must be counted | Future-capture candidate; not captured in Q3.0 |
| Prompt confidence/entropy | Potentially | Yes | No | Must be counted | Future-capture candidate; exact definition must be prelocked |
| Item ID | Yes | No | No | Negligible | Forbidden as a predictive feature; memorization shortcut |
| Reference answer or derived type/value | No | No | No | — | Forbidden |
| Program execution result | No | External execution | No | — | Forbidden |
| Any candidate answer | No | Yes | Yes | At least one generation | Forbidden for Route A |

The Q3.0 Route-A tournament used only deterministic prompt structure plus
frozen controller descriptors. It performed no prompt-activation capture and
no model inference.

## Route B: baseline-first adaptive policy

After one baseline answer exists, a frozen policy may additionally use:

- baseline commitment-valid and evaluable flags;
- baseline generated-token count;
- prospectively frozen confidence, entropy, or hidden-state summaries, if a
  later protocol explicitly captures and charges for them.

Baseline correctness and the reference remain unavailable. If an alternative
policy is invoked, the final answer-selection rule and the expected generation
budget must be fixed. Route B must beat repeated-baseline/self-consistency at
equal expected compute as well as the development-selected one-call champion.

## Route C: verifier-mediated committee

Once a frozen committee has generated answers, an eligible verifier may use
typed parsed answers, validity, agreement, likelihood/confidence and prompt
features. It may not execute the benchmark, see the reference, or use test
correctness. Q3.0 mechanically evaluated typed plurality without reading raw
text. K=4 lacks a closed four-call repeated-baseline comparator and therefore
cannot be promoted from the present development evidence.

## Cross-fitting rules

- The independent split unit is the semantic problem/family.
- Every policy, shell, rollout, baseline row and label for one item stays in
  one outer fold.
- Preprocessing, policy priors, shell choice, bank selection and champion
  selection are fit on the outer-training partition only.
- Hyperparameters use inner folds only.
- Item IDs and fold membership never enter the model as features.
- A future evaluation holdout cannot be used for calibration, bank reselection,
  stopping, or tie-breaking.

## Absolute prohibitions

The following are never router inputs:

- reference answers, correctness or any derivative;
- execution of benchmark programs;
- test-item policy outcome summaries;
- outer-test preprocessing or policy priors;
- item-specific exceptions;
- an LLM judge with reference access;
- policy-bank reselection after evaluation;
- raw controller outcome matrices used without fold isolation.

Invalid and unevaluable outputs count as incorrect. Missing scheduled outcomes
block a future analysis rather than being silently imputed.
