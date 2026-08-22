GATE 12 — UTILITY-ALIGNED DIRECTIONAL PULLBACK GEOMETRY
======================================================================

PREMORTEM / ENGINEERING
----------------------------------------------------------------------

premortem:
    PREMORTEM_PASS

full-sequence / KV-cache equivalence:
    FAIL under frozen BF16-aware tolerances
    max absolute logit difference = 0.8125
    max absolute KL difference = 0.5572845

exact JVP:
    IMPLEMENTED, BUT ENGINEERING GATE FAIL

JVP / finite-difference agreement:
    frozen aggregate median cosine = 0.6175741
    best prescribed single-scale diagnostic cosine = 0.9416130
    required = 0.995

local KL quadratic validation:
    FAIL

free-generation outputs:
    0

ITEMS
----------------------------------------------------------------------

control-validation:
    24 CRUX + 24 CHARCOUNT frozen, NOT COLLECTED

held-out utility prediction:
    32 CRUX + 32 CHARCOUNT frozen, NOT COLLECTED

outcome-based selection:
    NO

CONTROL GEOMETRY
----------------------------------------------------------------------

Not collected. No Q, Fisher/Hellinger energy, or log-Q/log-KL association was
computed.

UTILITY ALIGNMENT
----------------------------------------------------------------------

Not collected. Historical Gate-9/Gate-10 outcomes remained sealed and no
U_mean, utility slope, or domain-level alignment was computed.

POLICY ALIGNMENT
----------------------------------------------------------------------

Not collected.

PRIMARY CLASSIFICATION
----------------------------------------------------------------------

GATE12_JVP_ENGINE_FAILURE

INTERPRETATION
----------------------------------------------------------------------

Forward-mode autograd and an independent exact autograd JVP implementation
agreed closely (cosine 0.9999834). The failure is specifically that the exact
derivative could not satisfy the prospectively frozen finite-difference,
local-KL, and full-sequence/KV engineering checks in the frozen BF16 execution
regime. The protocol forbids substituting finite differences, loosening the
thresholds, or collecting geometry after this failure.

This result says nothing about whether directional pullback energy or local
utility alignment predicts control. Those quantities were not scientifically
collected. It does not establish a full pullback matrix, Fisher geometry, Q2,
or a null scientific geometry result.

FORENSIC
----------------------------------------------------------------------

classification:
    GATE12_FORENSIC_CLEAN

scientific metric comparison:
    NOT APPLICABLE — zero scientific geometry shards

RAW ARTIFACTS
----------------------------------------------------------------------

full baseline logits:
    NOT COLLECTED

full JVP vectors:
    NOT COLLECTED

raw archive:
    MANIFEST-ONLY ENGINEERING ARCHIVE
    1,517 bytes
    SHA-256 c615a6b2f09788e64884ca92a642a89894a02cb412034bc29271f1af472a008c

COST / INFRASTRUCTURE
----------------------------------------------------------------------

A40 runtime:
    approximately 0.203 hours

incremental cost:
    approximately US$0.0893

RunPod:
    TERMINATED

retained volumes:
    0

SCIENTIFIC FIREWALL
----------------------------------------------------------------------

new controller:
    NO

new dose:
    NO

new free generation:
    NO

historical outcome reveal:
    NO

Q2:
    NOT RUN

confirmatory holdout:
    UNTOUCHED
