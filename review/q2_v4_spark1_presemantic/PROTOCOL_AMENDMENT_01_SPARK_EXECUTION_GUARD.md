# Q2 V4 protocol amendment 01 — Spark execution guard

Classification: `CLASS_B_OPERATIONAL_PRE_OUTCOME_AMENDMENT`.

Chronology:

1. The prospective qualification lock was committed as `a512c4e9557e2233aacf1e564134f751b6198251`.
2. The exact model revision was downloaded to Spark-1 shared storage without GPU use.
3. The first engine invocation stopped before model loading because the repository's
   HuggingFace safety guard recognized only the historical RunPod `/workspace` profile.
4. No model output, source output, activation, candidate direction, shell result, A1/A2
   value, or semantic outcome existed.
5. This amendment adds an explicit opt-in `SPARK1` profile requiring hostname `spark1`,
   `HF_HOME=/srv/shared/hf-cache`, and a checkout below the laboratory project root.

The model, revision, layer, source definitions, thresholds, subspace rule, candidate RNG,
bank size, safety gate, A1/A2 definitions, panel, estimands, and classifications are
unchanged. Spark 2 remains forbidden. The semantic panel remains forbidden.
