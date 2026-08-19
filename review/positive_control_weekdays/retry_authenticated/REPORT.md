# Authenticated weekday positive-control retry

## Classification

`POSITIVE_CONTROL_PASS`

The exact frozen CausaLab weekday control completed on one NVIDIA A40. The
upstream mean cumulative behavior distance was:

- manifold/geometric: `0.32336652278900146` (SE `0.019892308861017227`);
- linear: `1.3987454175949097` (SE `0.1289888620376587`).

The manifold path reduced the frozen primary metric by
`0.7688167419736623` (76.9%), exceeding the prospectively frozen 30% threshold.
Endpoint top-1 weekday sanity was 672/672 (100%) for both geometric and linear
paths, exceeding the 90% threshold. Standard and extra path tensors contained
only finite, nonnegative weekday-candidate scores and showed no gross
intervention corruption.

## Exact reproduction

- Upstream: `goodfire-ai/causalab` at
  `7dcc8ec4ffd11efec8b3cf9febd6b523df7637b6`.
- Paper: `arXiv:2605.05115v1`.
- Model: base `meta-llama/Llama-3.1-8B` at
  `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- BF16, eager attention, layer 28, last prompt token.
- Frozen geometric and linear paths, 50 path steps, 16 prompts, 21 standard
  weekday pairs plus 29 extra pairs, seed 42.

Local and remote metadata-only gated-file probes passed under the Hugging Face
account `costaleirbag`; no credentials were logged. Model weights were
downloaded only to `/workspace/hf-cache`. No model weights were downloaded to
the Mac.

The first retry process exposed an upstream compatibility issue before path
steering: NumPy newer than the pinned lock rejected conversion of a singleton
one-dimensional array to `float`. The isolated CausaLab environment was restored
to the exact upstream locked NumPy/SciPy/scikit-learn versions and the same
frozen pipeline then completed. No scientific code, path, prompt, layer, model,
metric, threshold, or intervention setting changed.

## Cost and recovery

The retry Pod was billed for 4,106 seconds (1.1406 A40-hours), approximately
US$0.5018 at US$0.44/hour. The successful pipeline itself ran for about 44.4
minutes. The compact 17.5 MB result archive was copied to the Mac and verified
against the remote SHA-256
`2c420243c105226d1821d74c0f46697eb4a613a99dcc2b556a8e492b69386cdc`.
The Pod was then stopped and is `EXITED`.

## Interpretation boundary

This reproduces one published activation-to-behavior phenomenon and validates
that the compatibility intervention stack can cut. It does not establish
original Q1, semantic error complementarity, representation-to-error geometry,
Q2, or collective utility. The additive operator check remains the previously
recorded software-only PASS. Original Q1 steering, the substrate race, Q2, and
the confirmatory holdout were not run.

Next action: `PRINCIPAL_RESEARCHER_REVIEW`.
