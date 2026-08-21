# Gate 7 adversarial premortem

Classification: `PREMORTEM_PASS`

This premortem was completed before any Gate 7 model output. It treats the
historical Gate 6.3 V2 result as immutable and the V3 audit as an additive
development diagnostic only.

## A. Parser invariance

`external-semantic-v3` is reused byte-for-byte from the accepted audit branch.
Its module, specification, blinded corpus, and condition-invariance test hashes
are frozen in the protocol lock. Commitment validity, semantic evaluability,
and correctness remain separate. Markdown, multiline, bytes, and wrong-type
commitments are handled by global rules, never condition or item rules. Any
post-output concern triggers a preserved-result Class C audit, not a parser
change or replacement output.

## B. Data freshness

The exclusion scan covers every preserved JSON, JSONL, CSV, and Markdown review
artifact, including reserve pools and allocations. Gate 7's own output directory
is excluded from recursive re-scans. The scan found 473 consumed or reserved
CRUXEval IDs and 327 eligible IDs in the pinned 800-item test set, so the frozen
N=120 branch is feasible. The pinned dataset is resolved remotely without a
model; no benchmark cache is created on the Mac.

## C. Controller identity

The meaningful vector is loaded from the historical L27 `.npy` file without
reconstruction, sign change, or renormalization. Both file SHA-256 and canonical
float64 vector SHA-256 are checked against Gate 6.3. Eta, reference scale,
layer, constructor, source location, duration, and per-forward delta are frozen.

## D. Random-null matching

Four new deterministic Gaussian directions are orthogonalized against the
meaningful vector and each earlier Gate 7 random. Unit norm and pairwise cosine
checks are fail-closed. Every controller uses L27, equal standardized energy,
the same sustained hook, duration, scope, engine, and generation policy.

## E. Causal timing

The existing sustained `Gate6HookTrace` applies the delta to the final prompt
token on prefill and the sole current token on cached decode. It never rewrites
cached historical states. Engineering checks cover alpha-zero identity, exact
shift, current-token scope, forward count, cache safety, and hook cleanup.

## F. Attrition and missingness

Wrong, invalid, ambiguous, semantically unevaluable, truncated, and model
runtime outputs are scientific rows and primary errors. They never abort or
trigger behavioral retries. Only an explicitly classified infrastructure
failure may retry the same logical key and seed. One model output cannot erase
or replace another.

## G. Resume

The complete schedule is frozen before collection. The append-only journal is
flushed and fsynced after each row. Resume keys are exactly item, condition, and
rollout. Completed keys with matching provenance are skipped; duplicates,
third draws, seed changes, and overwrites are rejected.

## H. Schedule

For each item and rollout, all seven conditions are ordered by a stable hash.
The complete order is frozen and hashed before outputs. This interleaves
baseline, textual, meaningful, and random conditions without using response
length, parser status, model output, or later analysis.

## I. Estimands

G, C, and D use exactly two independent rollouts per condition. Invalid model
outcomes remain errors. Item IDs—not rows or tokens—are the bootstrap unit, and
all conditions/rollouts for an item move together. Synthetic estimator and
algebra tests run before collection.

## J. Auxiliary arm

Textual CAREFUL is a fixed behavioral reference. Its source-policy
classification and gain-recovery diagnostic are reported separately and cannot
change the primary controller classification or random-null comparison.

## K. Interpretation

The exhaustive frozen classification is ordered by validity/competence,
strong/minimum/qualitative replication, style-only control, then no replication.
The final report must describe which mechanism and guard produced the class,
not merely quote the first failed threshold. No threshold is modified after
outcomes.

## L. Environment

`CORE_QWEN` is mandatory. Remote preflight must verify Python, Torch/CUDA,
Transformers, Accelerate, Hugging Face Hub, SDPA, A40, model revision, source
commit, cache, and disk. A restart cannot silently change versions because the
runner records environment and source provenance per row and fails closed on a
lock mismatch.

## Decision

No unresolved scientific-design ambiguity remains. Dataset resolution on the
remote host, dependency restoration, SSH/Pod migration, cache repair, and
journal recovery are Class A operations. One Class-B clarification is frozen
before outputs: “clearly reproduces the textual CAREFUL token/style regime”
means recovering at least 50% of both the mean and median textual-CAREFUL token
increase over baseline, conditional on the CAREFUL source-policy gate passing.
This affects only the exhaustive style-only classification; it changes no
controller, outcome, primary estimand, replication threshold, or sample.
Collection remains forbidden until manifests, engineering tests, protocol lock,
source commit, remote preflight, and cost gate all pass.
