GATE 12.1 — CONTINUOUS GEOMETRY ENGINE QUALIFICATION
======================================================================

FIREWALL
----------------------------------------------------------------------

historical Gate-12 result:
    GATE12_JVP_ENGINE_FAILURE preserved

scientific items processed:
    0

historical outcomes revealed:
    NO

free-generation outputs:
    0

SEQUENCE LOCALIZATION
----------------------------------------------------------------------

source:
    mixed BF16 kernel, cache/reduction-order, and dtype effects;
    no sequence-semantic bug

first BF16 exceedance:
    layer 0, token 0, MLP output; max difference 0.015625

first FP32 diagnostic exceedance:
    layer 0, token 0, MLP output; max difference 2.0981e-05

FP32 full-sequence/KV qualification:
    PASS
    top-1 agreement 1.000000
    median JS 5.0033e-12
    p99 JS 2.5062e-11
    median target-logp difference 1.5718e-06
    maximum target-logp difference 6.1905e-05
    median logit cosine 0.9999999999995

HISTORICAL BF16 BRIDGE
----------------------------------------------------------------------

classification:
    FAIL

top-1 agreement:
    0.977444 (required >= 0.99)

median JS:
    4.2599e-05

median target-logp difference:
    0.008524

EXACT DERIVATIVES
----------------------------------------------------------------------

forward/independent JVP:
    PASS
    minimum cosine 0.9999999999990
    maximum relative norm difference 2.2787e-07

JVP/VJP duality:
    PASS
    maximum relative error 5.2961e-06

Fisher JVP/second derivative:
    PASS
    maximum relative error 5.1578e-06

utility derivative:
    PASS
    maximum relative error 3.7935e-05

FINITE DIFFERENCES / LOCAL KL
----------------------------------------------------------------------

stable window:
    FAIL — only epsilon 0.03 and 0.1 passed consecutively; three required

at epsilon 0.03, pooled medians:
    JVP cosine 0.999683
    Fisher relative error 0.006040
    utility relative error 0.047398
    local-KL relative error 0.016286

at epsilon 0.1, pooled medians:
    JVP cosine 0.999965
    Fisher relative error 0.001161
    utility relative error 0.017930
    local-KL relative error 0.002549

QUALIFICATION
----------------------------------------------------------------------

qualified geometry object:
    NONE

classification:
    GATE12_1_DERIVATIVE_ENGINE_NOT_QUALIFIED

The FP32 lift is mathematically coherent under exact-AD identities, but the
complete prospectively frozen qualification rule did not pass. This result is
not evidence for or against scientific pullback/Fisher prediction.

FORENSIC
----------------------------------------------------------------------

classification:
    GATE12_1_FORENSIC_CLEAN

maximum primary/audit metric difference:
    4.3632e-14

COST / INFRASTRUCTURE
----------------------------------------------------------------------

GPU:
    NVIDIA A40 48GB

numerical runner:
    480.4 seconds

total Pod runtime:
    approximately 0.25 hours

incremental cost:
    approximately US$0.11

RunPod:
    TERMINATED

retained volumes:
    0
