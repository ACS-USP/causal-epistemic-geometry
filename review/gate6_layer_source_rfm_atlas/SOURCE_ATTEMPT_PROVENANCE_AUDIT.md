# Gate 6 source-attempt provenance audit

The first source attempt remains immutable and is classified as `GATE6_SOURCE_PHASE_INCOMPLETE`. It preserved two item-level rows and stopped at `sample_169` CAREFUL after the frozen 4096-token cap. No controller, manipulation, or evaluation outcome was collected.

Protocol-lock source commit: `e4223f6a8910464f9df479f6cb270c673ad84f20`

Effective execution checkout: `a3233771332687acfd3a30ac86011cdfed5c23bf`

The exact diff between those commits is recorded below. It contains only state/lock/provenance metadata; it does not change the runner, prompts, manifests, direction, alpha, seed schedule, evaluator, or intervention implementation.

```text
experiments/registry.yaml                              | 2 +-
 experiments/specs/gate6_layer_source_rfm_atlas.yaml    | 2 +-
 project_state.yaml                                     | 2 +-
 review/gate6_layer_source_rfm_atlas/PROTOCOL_LOCK.json | 2 +-
 review/gate6_layer_source_rfm_atlas/PROTOCOL_LOCK.md   | 2 +-
 5 files changed, 5 insertions(+), 5 deletions(-)
```

The repair is prospective: it adds deterministic mechanical marker localization, condition-level journaling, and preallocated reserves. It never regenerates the failed item and never uses correctness for attrition decisions.
