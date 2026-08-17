# Q1 DEVELOPMENT PROTOCOL V1.2 — LABEL / POSITION BIAS DECONFOUNDING

**DEVELOPMENT FOLLOW-UP — NOT CONFIRMATORY**

Frozen before new V1.2 real-model outcomes: 2026-08-16. The repository baseline
used to freeze the choices was `c863c4f`; this protocol file and its runner are
committed as the next engineering change before any real V1.2 execution. This
protocol is motivated by the Q1 V1.1 principal-review artifact and does not
authorize V1.3, Q2, or access to `CONFIRMATORY_HOLDOUT`.

## Question

After mathematically balancing away displayed answer-slot identity, does the
frozen PC1 intervention still change which semantic option the model prefers?
This distinguishes a displayed answer-slot/letter effect from a semantic
error-profile change beyond that first-order confound.

## Frozen scientific objects

| Object | Frozen value |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Tokenizer revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Dataset | `TIGER-Lab/MMLU-Pro` |
| Dataset revision | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| Split | exact 512-item `DEV_EVALUATION` |
| Split manifest hash | `84982e4c72e230ffff78363f085d4d5c53447fd1e248e5e170ed5e8c508d343e` |
| PC1 vector hash | `abca43ae3b9621614562798dbfbd8c3ad9932fc9fcb0cfd2c58d28adc48897c5` |
| Layer | 17, zero-based |
| Token scope | `last_token` |
| PC1 alpha− | `-4.8855751862975145` |
| PC1 alpha+ | `4.8855751862975145` |
| PC1 calibration SD | `9.771150372595029` |
| Probe beta | `0.05` |
| Scorer | approved V1.1 candidate-only single-token scorer |
| Dtype | BF16 model; approved scoring arithmetic |
| Sampling/thinking | `do_sample=false`; `enable_thinking=false` |

The approved inference engine is the V1.1 exact-equivalence profile:
`full_prompt_batched`, `candidate_only`, `serial_shape_reference=true`, SDPA,
with no cached-decode, suffix-replay, compile, or CUDA-graph substitution.

## Balanced cyclic design

For an item with `K` semantic options, ordering `r` assigns semantic option
`j` to displayed slot `(j + r) mod K`, for `r = 0, ..., K-1`. Equivalently,
the semantic ID at displayed slot `s` is `(s - r) mod K`. Ordering `r=0` is
the original ordering. No random or outcome-selected permutations are added.

The implementation must assert for every item that every semantic option visits
every displayed slot exactly once, the target semantic identity is invariant,
and every target label remapping is consistent.

## Conditions

For every cyclic ordering, run only `baseline`, `PC1+`, and `PC1−`. In the same
item/order execution family, run the mechanistic probe at `+epsilon` and
`-epsilon`, where `epsilon = 0.05 * 9.771150372595029`. The probe is diagnostic
and does not replace the frozen main alpha conditions.

The original ordering's main V1.1 rows may be reused only when the complete
scientific cache key matches: model/tokenizer revisions, dataset and split
hashes, item ID, rendered prompt hash, scorer and candidate-head semantics,
vector hash, alpha, layer, token scope, and inference provenance. Each reused
row is marked `CACHE_REUSED_EXACT`; otherwise it is recomputed.

## Primary symmetrized score

Let `z[i,r,c,k]` be the candidate-only logit for displayed candidate `k`.
Center within each ordering:

```text
z_tilde[i,r,c,k] = z[i,r,c,k] - mean_k z[i,r,c,k]
```

For semantic option `j`, let `slot(i,r,j) = (j+r) mod K`. The primary score
is:

```text
S[i,c,j] = mean_r z_tilde[i,r,c,slot(i,r,j)]
```

The primary prediction is `argmax_j S[i,c,j]`. Centering removes arbitrary
ordering-level common score offsets. Cyclic balance cancels a first-order
additive displayed-slot contribution, but does not claim to remove every
possible position interaction.

As a secondary robustness diagnostic, compute candidate-set softmax values
within each ordering and average them after semantic remapping:

```text
Q[i,c,j] = mean_r softmax_k(z[i,r,c,*])[slot(i,r,j)]
```

Do not choose between `S` and `Q` after seeing outcomes. Report their agreement.

## Analyses frozen in advance

Report paired metrics for `baseline_sym` versus `PC1+_sym` and `PC1-_sym`, plus
the original-order V1.1 reference: accuracy, delta accuracy, 2×2 counts,
rescues, damages, disagreement, rescue/damage rates, error Jaccard, phi with
explicit undefined handling, pair oracle accuracy, complementarity headroom,
and deterministic item-level bootstrap intervals.

Compute directional response using the displayed candidate logits:

```text
D[i,r,k] = (z_plus[i,r,k] - z_minus[i,r,k]) / (2*epsilon)
```

Store raw and within-order centered `D`. Report mean/median/bootstrap intervals
for displayed labels A–J, pre-specified `A_vs_rest`, balanced slot versus
semantic-content tracking, and a descriptive displayed-slot variance fraction.
Also report main-alpha displayed-label changes, raw versus symmetrized margins,
category rescue/damage counts, and the six planned questions Q1.2-A through
Q1.2-F separately.

## Interpretation discipline

If the symmetrized effect disappears while slot response remains strong, that
is development evidence that the earlier signal was substantially
slot/position-mediated. If semantic rescues/damages and headroom remain, that
is development evidence for a residual effect beyond a simple fixed slot bias.
A mixed result remains mixed. No outcome is a confirmatory claim, and a
negative deconfounding result is not to be "salvaged" by changing layers,
alphas, vectors, prompts, or token scope.

## Provenance and firewall

All V1.2 artifacts live under `runs/q1_v1_2/`; V1 and V1.1 artifacts are never
overwritten. The manifest records the protocol commit, model/dataset revisions,
split hash, PC1 hash, prompt hash, approved engine, reuse decisions, probe
epsilon, grouped-bootstrap unit, and remote environment. The confirmatory
holdout is forbidden and must remain untouched.
