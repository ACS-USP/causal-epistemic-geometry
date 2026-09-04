# Q3 Single-Forward Routing Feasibility

## Status

`SINGLE_FORWARD_DEPLOYMENT_FEASIBLE` for the proposed Route-A mechanism.
This is an engineering result, not a Q3 utility result and not evidence that a
geometry-aware router is scientifically adequate.

## Mechanism

The frozen capture point is the layer-27 block input at the final non-padding
prompt token. In Qwen3-8B this tensor was mechanically identical to the
layer-26 block output at the same token in every audited capture. A pre-registered
layer-27 pre-hook can therefore read the representation and select one already
frozen policy before the layer-27 block executes. A layer-27 output hook can
then apply that policy's frozen current-token intervention during the same
module invocation. Decode-time intervention proceeds under the unchanged
sustained-current-token semantics.

No earlier token, historical activation, or past KV-cache entry is modified.
The mechanism does not require a second full prompt prefill.

## Evidence

- A synthetic hook-order test showed that a pre-hook selection is visible to
  an output hook in the same module invocation and that only the current final
  position changes.
- The authorized prompt-only Qwen capture found maximum absolute difference
  `0.0` between the layer-26 output and layer-27 input at the capture site.
- The deterministic 16-family repeat subset had maximum absolute capture
  difference `0.0`.
- The capture path calls the model forward directly and contains no generation,
  parser, scorer, reference-answer, or correctness path.

The capture and analysis did not run a steered semantic generation. Thus the
result establishes mechanical feasibility of the one-forward handoff, not its
realized accuracy, latency, or scientific utility.

## Compute interpretation

The development audit used 332 prompt-only forwards: 300 primary captures and
two captures of each of 16 forensic-repeat prompts. The capture runner reported
141.951 seconds from model construction through artifact persistence. A future
deployed Route-A call would use the ordinary answer-generation prefill itself;
there is no additional full-prefill charge in the proposed mechanism. Router
CPU latency and hook overhead would still need to be measured prospectively
before any confirmatory execution.

## Provenance

The release-safe capture audit is
[`Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSICS.json`](../review/q3_route_a_prompt_representation/Q3_ROUTE_A_PROMPT_REPRESENTATION_CAPTURE_FORENSICS.json).
The hidden-state matrix and benchmark prompt manifest remain private and are
identified by SHA-256 in the release-safety manifest.
