# Engineering policy

## Canonical locations

The local Mac repository is canonical for source, configuration, tests,
documentation, and Git history. Real model/data operations run only on an
explicitly authorized remote host with fail-closed cache/location checks. Model
weights, benchmark caches, real activations, and raw scientific predictions are
not downloaded to the Mac unless a principal explicitly authorizes a small
review artifact.

## Reproducibility

- Never use Python's randomized `hash()` for scientific identity or seeds.
- Record immutable model and dataset revisions, source commit, dirty status,
  exact prompts/hashes, parser/evaluator version, seed regime, and engine.
- Scientific identity is item × condition × rollout, never batch position.
- Resume skips only validated complete keys with matching provenance.
- Behavioral failures are measured outcomes and are never retried; only genuine
  infrastructure failures may repeat the same key/seed with retry provenance.

## Execution-engine changes

A reference engine remains available. Optimizations that claim equivalence must
pass the protocol's exact discrete or token-level gate. Batch-shape, cache,
precision, kernel, and sampling changes are scientific when they alter
trajectories.

A prospectively frozen efficient engine is valid without matching an older
engine only if it is chosen before outcomes, applied symmetrically to every
condition, preserves per-row RNG semantics, uses outcome-independent batching,
and remains unchanged throughout calibration and evaluation.

## Exploration versus infrastructure

Exploration should be cheap: a 5–20 item signal screen does not require a giant
manifest bureaucracy. Once a signal earns `DEVELOPMENT_LOCK`, package code,
schemas, tests, cost simulation, source freeze, and independent validators
become mandatory. Engineering work must not grow merely because an instrument
has already consumed effort.

## Generated-code security

Generated code is untrusted. `reliability_guard`, subprocess limits, and a
Python timeout are not security boundaries. Production evaluation requires a
disposable, credential-free Linux environment with:

- no network;
- no host home, repository, SSH agent, cloud credentials, or container socket;
- non-root UID, dropped capabilities, no-new-privileges;
- read-only root and minimal purpose-built mounts;
- CPU, memory, PID, output, and wall-time limits;
- immutable evaluator/image identity;
- destruction after artifact recovery.

The current `infra/evalplus_sandbox/` is a fail-closed template, not a ready
executor. Its unresolved base-image digest and non-portable host `timeout`
dependency are deliberate blockers. See
[the sandbox decision note](Q1_V4_DENSE_CODE_PILOT.md).

## Infrastructure options for dense code

| Option | Strength | Limitation | Current decision |
|---|---|---|---|
| Docker Desktop | Familiar isolation controls | Local daemon/license/desktop dependency | Not configured; no current priority |
| Disposable Linux CPU VM | Clean credential-free boundary, easy destruction | Additional provisioning and artifact plumbing | Preferred future evaluator |
| Isolated cloud sandbox | Elastic and disposable | Provider trust, egress, and cost controls required | Acceptable after explicit audit |
| Run untrusted code beside GPU/model | Low transfer friction | Catastrophic credential/model/workspace exposure | Forbidden |

The preferred architecture generates model outputs on the GPU host, transfers
only candidate code plus immutable task identity to a disposable evaluator,
returns signed/hashed test outcomes, then destroys the evaluator.

## Artifact and documentation policy

Historical scientific artifacts are immutable. Corrected analyses write a new
artifact with source hashes and a tracked interpretation note. Large raw data
remain ignored; small manifests, schemas, protocols, and hashes may be tracked.

`project_state.yaml` and `experiments/registry.yaml` are the live sources of
truth. Generated status must pass `make state-check`; canonical links and
historical classifications must pass `make docs-check` and
`make scientific-audit`.

