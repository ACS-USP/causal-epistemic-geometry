# Gate 3 — substrate race closeout

Status: `COMPLETE_EXPLORATION`. This was a baseline-only substrate selection
screen. No activation collection, steering, PCA, geometry, Q2, or holdout
access occurred.

## Frozen comparison

- Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`, full
  non-thinking generation.
- meta-llama/Llama-3.1-8B-Instruct, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- Fresh common instruments: 20 long character-count items and 20 CRUXEval
  semantic items.
- BF16, SDPA, `max_new_tokens=4096`, sampled generation with the frozen
  `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `min_p=0` policy.

## Outcome

The only eligible model-policy arm was Qwen full non-thinking:

| Cell | Valid | Correct | Wrong | Conditional accuracy | Eligible |
|---|---:|---:|---:|---:|---|
| Qwen × fresh long character count | 20/20 | 17 | 3 | 85.0% | Yes |
| Qwen × CRUXEval semantic | 19/20 | 9 | 10 | 47.4% | Yes |
| Llama-Instruct × CRUXEval semantic | 14/20 | 6 | 8 | 42.9% | No |
| Llama-Instruct × character count | 3/5 technical gate | 1 | 2 | 33.3% | No |

The two Qwen cells received the frozen second-seed resampling. CRUXEval had
19/20 valid outcomes per seed, 10 errors in each, zero hard-error disagreement,
and pair-oracle accuracy 47.4%. Character count had 20/20 valid outcomes,
25% hard-error disagreement, pair-oracle accuracy 95%, and 12.5 percentage
points of ordinary-resampling gain over mean single-rollout accuracy.

## Development recommendation

Primary substrate:

`QWEN_NONTHINKING × CRUXEVAL_SEMANTIC`

It provides the largest clean mass of completed genuine errors while retaining
correct outcomes and objective deterministic grading. Its two-rollout pattern
is compatible with stable item difficulty, but `R=2` is explicitly low
resolution and is not a claim about latent error propensities.

Backup substrate:

`QWEN_NONTHINKING × FRESH_PSEUDOWORD_LONG`

It has perfect completion and a useful correct/wrong split, but its 85%
conditional accuracy and high pair-oracle gain mean ordinary resampling already
provides substantial complementarity; it is less attractive as the first
substrate for an intervention pilot.

Llama-Instruct was not promoted because completion/mechanical failures failed
the frozen eligibility rule. No prompt, cap, parser, or model revision was
changed after outcomes.

## Cost and artifacts

The run produced 105 trajectory rows. Generation-time attribution was about
US$0.131 at US$0.44/A40-hour. Including the remote Pod wall-clock interval,
the conservative estimate was about US$0.277. The Pod was stopped immediately
after recovery. The detailed journal, manifests, tables, and hashes are in
`review/substrate_race/` locally and remain separate from the Git-tracked
protocol history.

## Boundary

This result selects a development substrate only. It does not show that
steering changes semantic errors, that errors are causally controllable, that
geometry predicts error covariance, or that any committee utility is
realizable. The next step requires principal review and a separate explicit
authorization for a tiny original micro-Q1.
