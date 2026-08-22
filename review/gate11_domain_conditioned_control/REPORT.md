GATE 11 — DOMAIN-CONDITIONED CONTROL POSTMORTEM
======================================================================

No new free generation or semantic evaluation was performed. Prompt-only
activations and teacher-forced historical baseline sequences were used.

COMPONENT DIAGNOSTICS
----------------------------------------------------------------------

source-axis transfer: SOURCE_AXIS_TRANSFER_SUPPORTED
downstream control-gain shift: CONTROL_GAIN_SHIFT_NOT_ESTABLISHED
policy-realization shift: POLICY_REALIZATION_SHIFT_NOT_ESTABLISHED
policy-utility shift: POLICY_UTILITY_DOMAIN_SHIFT_SUPPORTED

Relative-dose geometry:

[
  {
    "domain": "CRUXEval",
    "natural_gap": 124.4308745568854,
    "ordinary_projection_scale": 8.342790438533301,
    "delta_d75": 97.85168930581241,
    "delta_over_gap": 0.7863939689749434,
    "delta_over_scale": 11.728892152661475,
    "ordinary_to_careful_distance_before": 120.94128344266153,
    "ordinary_to_careful_distance_after": 23.089594136849126,
    "fraction_moved_toward_careful_centroid": 1.0
  },
  {
    "domain": "CHARCOUNT",
    "natural_gap": 112.17015036771677,
    "ordinary_projection_scale": 3.0036840308695436,
    "delta_d75": 97.85168930581241,
    "delta_over_gap": 0.8723505227106721,
    "delta_over_scale": 32.57722460157871,
    "ordinary_to_careful_distance_before": 56.44290208513587,
    "ordinary_to_careful_distance_after": 41.40878722067654,
    "fraction_moved_toward_careful_centroid": 1.0
  }
]

Control-gain summary:

[
  {
    "domain": "CRUXEval",
    "metric": "next_token_kl",
    "meaningful": 0.21232722433117276,
    "random_mean": 0.029154383086021668,
    "random_max": 0.04655474769631727,
    "meaningful_minus_random_mean": 0.18317284124515107,
    "meaningful_minus_random_max": 0.1657724766348555
  },
  {
    "domain": "CRUXEval",
    "metric": "A35",
    "meaningful": 3.0734786825772153,
    "random_mean": 2.0314159381358343,
    "random_max": 2.0796414248636395,
    "meaningful_minus_random_mean": 1.0420627444413808,
    "meaningful_minus_random_max": 0.9938372577135759
  },
  {
    "domain": "CRUXEval",
    "metric": "top1_flip",
    "meaningful": 0.11884920634920633,
    "random_mean": 0.03072916666666667,
    "random_max": 0.04861111111111111,
    "meaningful_minus_random_mean": 0.08812003968253967,
    "meaningful_minus_random_max": 0.07023809523809521
  },
  {
    "domain": "CHARCOUNT",
    "metric": "next_token_kl",
    "meaningful": 0.06766350901994611,
    "random_mean": 0.014348872864711244,
    "random_max": 0.020219815506287486,
    "meaningful_minus_random_mean": 0.05331463615523487,
    "meaningful_minus_random_max": 0.047443693513658625
  },
  {
    "domain": "CHARCOUNT",
    "metric": "A35",
    "meaningful": 3.635641142843133,
    "random_mean": 1.8866983563515525,
    "random_max": 1.9231283003423236,
    "meaningful_minus_random_mean": 1.74894278649158,
    "meaningful_minus_random_max": 1.7125128425008094
  },
  {
    "domain": "CHARCOUNT",
    "metric": "top1_flip",
    "meaningful": 0.06527777777777778,
    "random_mean": 0.030324074074074073,
    "random_max": 0.04814814814814814,
    "meaningful_minus_random_mean": 0.03495370370370371,
    "meaningful_minus_random_max": 0.01712962962962964
  }
]

Careful-alignment summary:

[
  {
    "domain": "CRUXEval",
    "mean_careful_logit_alignment": 0.24498206914734846
  },
  {
    "domain": "CHARCOUNT",
    "mean_careful_logit_alignment": 0.5555074607884846
  }
]

Historical utility reanalysis:

{
  "domains": {
    "CRUXEval": {
      "meaningful_accuracy_change": 0.13,
      "textual_accuracy_change": 0.33000000000000007,
      "meaningful_estimands": {
        "B00": 0.51,
        "B0j": 0.3775,
        "O00": 0.49,
        "O0j": 0.6225,
        "G": 0.1325,
        "U00": 0.2784848484848485,
        "U0j": 0.21032828282828284,
        "C": 0.06434343434343434,
        "D": 0.12,
        "rescue": 0.1525,
        "damage": 0.0225,
        "accuracy_baseline": 0.47,
        "accuracy_condition": 0.6
      },
      "textual_estimands": {
        "B00": 0.51,
        "B0j": 0.19,
        "O00": 0.49,
        "O0j": 0.81,
        "G": 0.32,
        "U00": 0.2784848484848485,
        "U0j": 0.10515151515151515,
        "C": 0.14666666666666667,
        "D": 0.3,
        "rescue": 0.34,
        "damage": 0.01,
        "accuracy_baseline": 0.47,
        "accuracy_condition": 0.8
      }
    },
    "CHARCOUNT": {
      "meaningful_accuracy_change": -0.02750000000000008,
      "textual_accuracy_change": -0.05750000000000011,
      "meaningful_estimands": {
        "B00": 0.045,
        "B0j": 0.06125,
        "O00": 0.955,
        "O0j": 0.93875,
        "G": -0.01625,
        "U00": 0.018542713567839195,
        "U0j": 0.022493718592964822,
        "C": -0.012298994974874373,
        "D": -0.025,
        "rescue": 0.07625,
        "damage": 0.10375,
        "accuracy_baseline": 0.8625,
        "accuracy_condition": 0.835
      },
      "textual_estimands": {
        "B00": 0.045,
        "B0j": 0.0325,
        "O00": 0.955,
        "O0j": 0.9675,
        "G": 0.012499999999999997,
        "U00": 0.018542713567839195,
        "U0j": 0.02678391959798995,
        "C": 0.02074120603015075,
        "D": 0.035,
        "rescue": 0.105,
        "damage": 0.1625,
        "accuracy_baseline": 0.8625,
        "accuracy_condition": 0.8049999999999999
      }
    }
  },
  "domain_contrasts": {
    "meaningful_accuracy": {
      "estimate": 0.1573675,
      "q025": 0.0775,
      "q975": 0.23500000000000001,
      "point": 0.1575
    },
    "textual_accuracy": {
      "estimate": 0.38712250000000004,
      "q025": 0.28250000000000003,
      "q975": 0.49250000000000005,
      "point": 0.3875
    },
    "G": {
      "estimate": 0.14904900000000001,
      "q025": 0.08500000000000006,
      "q975": 0.21874999999999994
    },
    "C": {
      "estimate": 0.07633893447033147,
      "q025": 0.03787169591645095,
      "q975": 0.11621528253641944
    },
    "D": {
      "estimate": 0.145795,
      "q025": 0.08,
      "q975": 0.22000000000000003
    }
  },
  "policy_utility_shift_supported": true
}

PRIMARY SYNTHESIS
----------------------------------------------------------------------

GATE11_POLICY_UTILITY_DOMAIN_MISMATCH

MEASUREMENT DISTINCTIONS
----------------------------------------------------------------------

1. Source-axis gaps, AUROC, and direction cosines measure representation
   transfer.
2. D75 next-token KL/JS and downstream hidden displacement are finite-
   displacement control-gain diagnostics.
3. Gate 11 did not measure an exact local pullback metric and did not establish
   Fisher geometry.
4. Historical accuracy and G/C/D measure task utility, not control energy.

RAW-PERSISTENCE BOUNDARY
----------------------------------------------------------------------

Prompt-boundary activations were preserved in float32. The fixed-sequence
journal preserved per-item/per-condition/per-checkpoint scalar logit metrics,
hidden displacement norms, token checkpoints, target-token indexing, D75
normalization, and provenance. It did not preserve complete per-checkpoint
vocabulary-logit arrays or hidden-state difference vectors. Consequently the
primitive KL/JS/vector calculations cannot be independently recomputed from
the recovered artifact alone; no replacement diagnostic collection was run.

INTERPRETATION BOUNDARY
----------------------------------------------------------------------

This DEVELOPMENT postmortem localizes candidate domain conditioning. It does
not establish Q2, optimize a controller, score new semantic responses, or touch
the confirmatory holdout.
