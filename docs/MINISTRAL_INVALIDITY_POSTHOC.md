# Ministral confirmatory invalidity: aggregate post-hoc report

Status: `POST_HOC_DESCRIPTIVE_ONLY`.

The immutable confirmatory classification remains
`Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL`. This document contains only
remote-safe aggregates. It excludes raw model text, benchmark content, item
identities, and row-level recovered answers.

## Provenance

- Canonical confirmatory closeout: `0a6127eb6bff68b27f64bbbbdf8d813561e00e76`.
- Preserved local forensic commit: `8682607dde267827c9fa5367cbc63d92510f3da4`.
- Immutable local Ministral journal size: 32,170,782 bytes.
- Immutable local journal SHA-256:
  `c571078c48a06759478968e9b92509c09ac53dd4ea83de03f20694ff7867415d`.
- No model inference, parser change, row regeneration, or complete-case
  confirmatory analysis occurred.

The local forensic commit intentionally remains outside this branch because it
contains raw outputs. The branch is preserved locally and was not rewritten.

## Aggregate findings

There were 13 invalid or non-evaluable `MEANINGFUL_FIXED` rows among 114,
distributed over 11 items. Nine items had one of two meaningful rollouts
invalid; two had both rollouts invalid.

| Exclusive primary category | Rows |
|---|---:|
| Token-cap truncation | 3 |
| Malformed final commitment | 4 |
| Multiple final commitments | 3 |
| Clear semantic answer in rejected commitment structure | 3 |
| Other primary categories | 0 |

All three token-cap cases also had runaway or unresolved reasoning as a
secondary feature. A condition-blind second pass by the same analyst agreed on
all 13 primary categories; it was not an independent reviewer.

Descriptive human recovery found nine correct candidates, three wrong
candidates, and one ambiguous output. Recovered candidates did not enter
accuracy, G, C, D, validity, evaluability, or the terminal decision.

Invalid meaningful rows were longer than valid meaningful rows: mean/median
token counts were 1500.38/840 versus 339.86/222. Three invalid rows reached
exactly 4096 tokens. Invalidity occurred in 13 meaningful rows, compared with
4 baseline, 5 textual-careful, and a mean of 3.75 rows across four random
conditions.

Across the frozen 228 baseline-by-meaningful rollout pairs, invalid meaningful
outputs produced zero rescues, 11 of the 16 damage pairs, and 15 double-fault
pairs. They therefore attenuated rather than generated the positive
complementarity result.

## Interpretation boundary

The best-supported description is mixed commitment-format instability and
generation instability, not predominantly semantic degeneration and not a
frozen-parser implementation bug. This is a mechanism hypothesis generated
after the experiment. It does not establish a safer Ministral controller and
does not soften the safety-guard failure.

The generic offline summarizer is
[`scripts/summarize_posthoc_ministral_invalidity.py`](../scripts/summarize_posthoc_ministral_invalidity.py).
It reads a local journal and emits aggregates only; it never exports raw text.
