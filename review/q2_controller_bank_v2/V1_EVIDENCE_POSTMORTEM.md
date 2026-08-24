# Q2 V1 offline evidence postmortem

This is a descriptive DEVELOPMENT postmortem of the immutable Q2-V1
qualification artifacts. It used 144 source rows and
204 matched manipulation rows. No correctness, accuracy,
G, C, D, rescue, damage, or common-panel outcome was read or computed.

## Main findings

- The old common displacement norm was exactly
  97.8516893058 for every controller.
- 2/12
  meaningful controllers reached the historical 0.25 raw-sequence movement
  threshold.
- The 10 failing meaningful controllers were below that threshold
  by a mean of 0.141667; movement values were discrete
  multiples of 1/12.
- 7
  meaningful controllers were at 1/12, and
  3
  were at 2/12, immediately below the old cutoff.
- Sign and source-location effects are summarized below; they are descriptive,
  not selection criteria.
- Sensitivity was sign-asymmetric but not in one universal direction: the
  prompt-boundary minus sign was strongest for independent verification and
  type discipline, while explicit state tracking was stronger at the execution
  boundary with the plus sign.
- Axis sensitivity was modest but visible: explicit state tracking averaged
  0.1042 raw movement, versus 0.1458 for verification and type discipline.
- Prompt-boundary interventions averaged 0.1528 raw movement versus 0.1111 at
  the execution boundary. This is a descriptive location pattern, not a
  causal source-location claim.

## Axis aggregates

{
  "EXPLICIT_STATE_TRACKING": {
    "mean_raw_sequence_movement": 0.10416666666666666,
    "mean_semantic_movement": 0.08333333333333333,
    "mean_token_delta": 0.18750000000000003
  },
  "INDEPENDENT_VERIFICATION": {
    "mean_raw_sequence_movement": 0.14583333333333334,
    "mean_semantic_movement": 0.12499999999999999,
    "mean_token_delta": 0.12499999999999999
  },
  "TYPE_REPRESENTATION_DISCIPLINE": {
    "mean_raw_sequence_movement": 0.14583333333333331,
    "mean_semantic_movement": 0.125,
    "mean_token_delta": 0.22916666666666669
  }
}

## Source-location aggregates

{
  "EXECUTION_BOUNDARY": {
    "mean_raw_sequence_movement": 0.1111111111111111,
    "mean_semantic_movement": 0.1111111111111111,
    "mean_token_delta": 0.16666666666666666
  },
  "PROMPT_BOUNDARY": {
    "mean_raw_sequence_movement": 0.15277777777777776,
    "mean_semantic_movement": 0.1111111111111111,
    "mean_token_delta": 0.19444444444444445
  }
}

## Sign pairs

{
  "EXPLICIT_STATE_TRACKING:EXECUTION_BOUNDARY": {
    "minus_raw_sequence_movement": 0.08333333333333333,
    "plus_minus_difference": 0.0,
    "plus_raw_sequence_movement": 0.08333333333333333
  },
  "EXPLICIT_STATE_TRACKING:PROMPT_BOUNDARY": {
    "minus_raw_sequence_movement": 0.16666666666666666,
    "plus_minus_difference": -0.08333333333333333,
    "plus_raw_sequence_movement": 0.08333333333333333
  },
  "INDEPENDENT_VERIFICATION:EXECUTION_BOUNDARY": {
    "minus_raw_sequence_movement": 0.08333333333333333,
    "plus_minus_difference": 0.08333333333333333,
    "plus_raw_sequence_movement": 0.16666666666666666
  },
  "INDEPENDENT_VERIFICATION:PROMPT_BOUNDARY": {
    "minus_raw_sequence_movement": 0.25,
    "plus_minus_difference": -0.16666666666666669,
    "plus_raw_sequence_movement": 0.08333333333333333
  },
  "TYPE_REPRESENTATION_DISCIPLINE:EXECUTION_BOUNDARY": {
    "minus_raw_sequence_movement": 0.16666666666666666,
    "plus_minus_difference": -0.08333333333333333,
    "plus_raw_sequence_movement": 0.08333333333333333
  },
  "TYPE_REPRESENTATION_DISCIPLINE:PROMPT_BOUNDARY": {
    "minus_raw_sequence_movement": 0.25,
    "plus_minus_difference": -0.16666666666666669,
    "plus_raw_sequence_movement": 0.08333333333333333
  }
}

## Interpretation boundary

The observed bank had a narrow discrete first-stage range at the single frozen
norm. This supports per-direction dose calibration and a continuous bank-level
dynamic-range rule in V2. It does not establish that any direction is useful for
semantic error control, because downstream correctness was intentionally absent.
