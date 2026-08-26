# Q2 V4.1 normative execution and analysis lock addendum

Status: `Q2_V4_1_NORMATIVE_LOCK_COMPLETE`

This is a presemantic normative addendum to the frozen Q2 V4.1 prediction lock.
It makes existing execution and analysis intent explicit. It adds no model
output, semantic outcome, correctness inspection, controller, panel, schedule,
metric, threshold, or scientific authorization.

## Scope and lineage

- Parent lock: `review/q2_v4_1_prediction_lock/PROTOCOL_LOCK.json`
- Parent-lock SHA-256: `0adc2d04e314bca4bf488595cdbd171da1a47f439b90170cb8125c9def35d278`
- Starting/source commit: `1a88b869540c58823f31e903ff37fea7c8de0d6c`
- Exact bank: 31 controllers, in immutable original candidate order.
- Semantic execution remains unauthorized; Q2 remains `UNTESTED`; Q3 remains
  `NOT_RUN`.
- V4/V4.1 semantic trajectories remain `0`; correctness remains uninspected.

The machine-readable normative source is
`Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json`; its SHA-256 is recorded
in the final commit and in the handoff.

## A. Generation parameters

The future executor must consume the hash-pinned 37,800-row schedule and reject
any row whose frozen provenance differs. The reconstructed values are:

- Model: `Qwen/Qwen3-8B`, revision and tokenizer revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- `BF16`, no quantization, `AutoModelForCausalLM`, `trust_remote_code=false`.
- `enable_thinking=false`; chat rendering uses the canonical
  `apply_chat_template(..., tokenize=false, add_generation_prompt=true,
  enable_thinking=false)` path.
- The V4.1 panel has one user message and no system message in item metadata.
  The frozen user prompt is the canonical Q2 V3 code-output prompt, with the
  exact prompt and prompt hash supplied by the panel/schedule.
- Sampling: `do_sample=true`, temperature `0.6`, `top_p=0.95`, `top_k=20`,
  `min_p=0.0`, `max_new_tokens=4096`.
- No project-supplied repetition, frequency, or presence penalty; no stop
  strings; EOS follows the model GenerationConfig at the frozen revision;
  `pad_token_id` is supplied from the tokenizer.
- Attention: SDPA; serial reference execution; batch size, item batch size,
  and condition chunk size are all `1`; no cross-condition batching.
- Layer: `model.model.layers[27]`; PROMPT_BOUNDARY source; sustained current
  token; frozen row alpha times the immutable vector; final non-padding prompt
  token during prefill and current token once per decode forward; cached
  historical positions are never modified.
- Baseline has no intervention. All 31 non-baseline controllers are the frozen
  meaningful bank; V4.1 has no random controller conditions.
- Seeds are the exact row seeds in
  `FUTURE_SEMANTIC_SCHEDULE.json`; the schedule was generated with the frozen
  Q2-V4.1 namespaces and PCG64DXSM condition-order permutations.

Sources of truth and hashes are listed under
`generation_specification.source_references` in the JSON addendum, including
the backend, prompt renderer, canonical task prompt, V4.1 schedule builder,
and the frozen parent lock. The historical helper
`src/epistemic_geometry/experiments/q2_v4_presemantic.py::semantic_schedule`
still contains a V4-era `SELECTED_COUNT=32`; it is explicitly excluded from
V4.1 and must not regenerate or replace the hash-pinned 31-controller schedule.

## B. Retry and resume

The logical identity is exactly:

`(item_id, condition, rollout_index)`

A completed row must contain the raw generated output or a recorded runtime
outcome, parser/scoring fields, token metadata, the exact seed, and matching
frozen provenance, persisted by append, flush, and fsync.

Only failures before a scientific row is persisted are operationally
retryable: transport/network failure, process or scheduler interruption,
recoverable incomplete artifact write/final journal tail, model-load failure
before a trajectory exists, or timeout before a valid row exists. A retry keeps
the same logical key, seed, scientific inputs, and provenance, and appends retry
provenance. The frozen operational limit is three infrastructure attempts; after
that the run terminates as `ENGINE_FAILURE` without replacement.

Valid wrong answers, invalid/unevaluable commitments, truncation recorded as an
outcome, missing final commitments, recorded model-runtime outcomes, and any
undesirable result are scientific outcomes and are not redrawn. No item,
condition, rollout, seed, controller, prompt, threshold, or model-level outcome
may be replaced.

`CrashSafeJournal` ignores an identical duplicate, raises on a conflicting
duplicate, and only quarantines a malformed final tail; a malformed non-final
row blocks. Resume validates the existing manifest/journal and processes only
pending schedule keys with matching provenance. A third scientific draw is
forbidden.

## C. Semantic estimands

For item `t`, condition `i`, and independent rollout `r`,

`e[i,t,r] = 0` iff the frozen external-semantic-v3 result is correct; otherwise
`e[i,t,r] = 1`, including valid wrong, invalid/unevaluable, no/ambiguous
commitment, truncation, and a recorded model-runtime outcome. Missing logical
rows are not converted into errors: an incomplete schedule blocks analysis.

For a condition pair `(i,j)`:

```text
d[i,j,t,r] = e[i,t,r] - e[j,t,r]
Dtotal[i,j] = mean_t(d[i,j,t,0] * d[i,j,t,1])
m[r]        = mean_t(d[i,j,t,r])
Dshape_panel[i,j] = Dtotal[i,j] - m[0] * m[1]
Dshape_superpopulation[i,j] = N/(N-1) * Dshape_panel[i,j]
```

Here `N=300`, there are two independent `INDEPENDENT_PRIMARY` rollouts, item
weights are uniform `1/N`, and negative finite-sample estimates are retained
without clipping. The panel quantity is named `D_shape_panel`; the primary
superpopulation quantity is named `D_shape_superpopulation`. Baseline is
condition zero and is paired separately with each controller in each shell.
MEDIUM and STRONG are not pooled before shell-specific quantities are formed.
Invalid scientific rows remain `e=1`; missing rows block.

The existing radial secondary estimands are inherited exactly:

```text
R_shape = D_shape(BASELINE, STRONG) - D_shape(BASELINE, MEDIUM)
R_total = D_total(BASELINE, STRONG) - D_total(BASELINE, MEDIUM)
```

They use the frozen paired shell-swap procedure and remain independent of the
G0/G1/G2/G3 classification. No second radial test is introduced.

The primary geometry matrices are unchanged: A0 is coordinate-space angular
dissimilarity `1-cosine`; A1 is regularized covariance-whitened angular
dissimilarity with lambda `0.1`; A2 is baseline-centered natural-log full-
vocabulary JS response angle with equal `0.5/0.5` mixture and uniform mean over
48 raw probe/checkpoint rows; D2 is the finite-response total-distance
secondary. Their sealed hashes are in both the JSON addendum and the frozen
matrix metadata.

Bootstrap is 10,000 percentile item-cluster resamples, seed
`1885846737463784981`; all 63 conditions and both rollouts move together for a
sampled item. Pairwise upper-triangle weights are uniform for association.

## D. G0-G3 mechanical classification

For each A0, A1, and A2 metric, compute average-tie Spearman correlation
(Pearson correlation of average ranks) between the upper-triangle metric and
`D_shape_superpopulation`, separately for MEDIUM and STRONG. The primary
association is the equal-weight arithmetic mean of the two shell-specific
correlations. A primary metric qualifies only if all frozen metric-gate
requirements pass: both shell correlations strictly positive, shell-mean rho
at least `0.2`, delete-one-controller sign stability, a strictly positive
item-bootstrap lower bound, and single-step maxT-adjusted QAP p at most `0.05`.

QAP is controller-label permutation only, `E -> P_pi E P_pi^T`, using the
hash-pinned 50,000-map schedule with identity first, the same permutation across
shells and A0/A1/A2, and p-value
`count(T_perm >= T_observed) / 50000`. The maxT correction is single-step over
A0/A1/A2. No pairwise independence assumption is used.

The mechanical precedence table is:

| State | Rule |
|---|---|
| `V4-G3` | A2 qualifies, and both A2-minus-A0 and A2-minus-A1 meet the frozen `>= 0.10` margin, with positive paired-bootstrap lower bounds and both single-step maxT superiority p-values `<= 0.05`. |
| `V4-G2` | G3 does not pass, and A2 qualifies. |
| `V4-G1` | G3 and G2 do not pass, and A0 or A1 qualifies. |
| `V4-G0` | None of A0, A1, or A2 qualifies. |

All listed conditions are AND conditions within a rule; precedence is G3,
then G2, then G1, then G0. Radial suffixes `RT+`, `RT-`, `RS+`, and `RS-` are
reported only under the frozen independent radial procedure. D2 remains
secondary unless the frozen V4 plan explicitly uses it; it does not replace
A0/A1/A2.

`G3_POWER_CHARACTERIZATION_IS_PLANNING_ONLY`: the existing planning artifact
uses delta grid `0, 0.05, 0.10, 0.15, 0.20`, with 0 representing the true
no-superiority null. It does not modify the G3 rule.

## Sealed objects and consistency

The JSON addendum pins the exact 31-bank manifest, panel manifest, 37,800-row
schedule, QAP schedule and permutation array, A0/A1/A2/D2 matrices and archive,
24-file A2 archive, environment fingerprint, and final presemantic bundle. It
also records source file hashes for every reconstructed normative section.

The consistency audit passed against the parent lock, frozen panel and schedule,
QAP schedule, matrix metadata, V4 statistical plan, V4 geometry implementation,
V4.1 schedule builder, semantic-v3 implementation, and reliability policy. The
legacy 32-controller helper is excluded by source-of-truth boundary. No
hash-pinned scientific object changed and no scientific choice was added.

## Absolute firewall

This addendum does not authorize execution. Before a separate principal
authorization, the required state remains:

- model/GPU/Spark inference: none;
- semantic trajectories: `0`;
- correctness inspected: `NO`;
- D-shape, radial semantic statistic, and empirical G0-G3: absent;
- Spark 1, Spark 2, and RunPod in this task: not used;
- Q3: `NOT_RUN`.

The next action is only a separate principal-researcher authorization for
`Q2_V4_1_SEMANTIC_EXECUTION`.
