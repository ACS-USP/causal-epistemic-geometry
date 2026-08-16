# Maximum inference optimization report

Date: 2026-08-16

This is an engineering report for Q1 DEVELOPMENT V1.1. It is not a scientific
result and does not alter the frozen protocol.

## Reference and approval

- Serial reference: preserved as `review/serial_reference_v1_1_partial/`.
- Clean optimized RunPod artifact: remote run
  `20260816T231331Z_q1-v1-1-development_b87abab787`.
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Dataset revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
- Split: frozen `DEV_EVALUATION`, 512 items.
- Conditions: 31; rows: 15,872.
- Full predictions remain on RunPod. Local review contains only small metadata,
  metrics, audit, summary, and permutation-manifest files.

## Approved execution profile

```text
engine                         full_prompt_batched
serial_shape_reference         true
candidate_head_mode            candidate_only
item_batch_size                1
condition_chunk_size           1
padding                        left
attention                      SDPA (requested auto)
torch.compile                  false
CUDA graphs                    false
```

The A–J candidate continuation audit passed: all allowed labels were
single-token and context-compatible under the frozen Qwen3 chat template.
Serial reference versus the approved profile passed on all 512 DEV items for
baseline, PC1−, and PC1+, with zero discrete prediction, correctness, and
ranking differences. Candidate-only scores are unnormalized logits; margins
and rankings are comparable, absolute values are not full-vocabulary
log-probabilities.

## Rejected alternatives

Ordinary BF16 cached decode and shape-changing batching were exercised and
rejected for exact Q1 use after prediction flips against the serial oracle.
Native Qwen3 suffix replay was implemented with strict version/model guards and
matched cached decoding on technical smoke, but was slower and is not
canonical. `torch.compile`, CUDA graphs, and formal A40 autotuning were not
approved or enabled.

## Runtime

The prediction journal reached its complete 15,872 rows at approximately
17.48 minutes after the run timestamp. The old serial estimate was 183.80
minutes, giving approximately 10.52× observed speedup. At the requested
$0.44/A40-hour accounting rate, the observed compute estimate is approximately
$0.13 total, or $0.081 per 10,000 item-condition rows. The run manifest's
conservative $0.40 planning rate gives approximately $0.12. GPU utilization and
peak VRAM were not formally captured for this run.

## Scientific firewall

- Stage: DEVELOPMENT.
- Confirmatory holdout: not accessed.
- Scientific Q1 result: none frozen.
- V1.2: not run.
- Q2 geometry: not run.

The clean run is eligible for principal-researcher review as a reproducible
development artifact. It is not evidence that steering improves accuracy,
complementarity, or collective utility.
