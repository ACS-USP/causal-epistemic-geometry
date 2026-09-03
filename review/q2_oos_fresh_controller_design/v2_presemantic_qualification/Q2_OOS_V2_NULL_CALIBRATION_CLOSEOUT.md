# Q2 OOS V2 null-calibration closeout

## Immutable scientific state

The accepted V2 design remains `K=16`, candidate reserve `n=34`, Route C, with A0 as the primary fresh-by-old geometry and fresh-by-fresh geometry secondary. The historical V1 stream and its blocked classification are unchanged.

This V2 attempt stopped before PRELOCK, before derivation of an actual V2 seed, before generation of a candidate stream, and before any model or semantic inference.

## Prospective calibration

The calibration procedure and ruling thresholds were committed and pushed before the full simulation:

- precheck commit: `630b0854b191d1da87800396987e1e6d85c2bdc6`;
- map-count erratum commit: `fe255060f4ec9ee2143a62b810a9d91aecd9167b`;
- implementation commit: `48cba4e740fc65940d11b980d58780e381a6be7e`;
- nominal alpha: 0.05;
- strict exchangeable null: 10,000 panels, 1,000 total maps;
- reviewer-hardening stress null: 10,000 panels, 999 total maps;
- future implementation check: 500 panels at 1,000 versus 50,000 maps;
- small-K audit: 2,000 K=6 panels, exact 720 maps versus 500 sampled maps.

The local vectorized benchmark projected more than the frozen 30-minute local threshold. The full model-free calibration therefore ran CPU-only on Spark 1 with CUDA disabled. It completed in 60.13 seconds and did not load a model.

## Strict exchangeable null

| Quantity | Result |
|---|---:|
| panels | 10,000 |
| p <= 0.01 | 0.0111 |
| p <= 0.05 | 0.0512 |
| Wilson 95% interval at 0.05 | [0.04705, 0.05569] |
| p <= 0.10 | 0.0983 |
| materially anti-conservative | false |

The strict exchangeable-null calibration did not trigger the prospectively frozen anti-conservatism gate.

## Reviewer-hardening stress null

| Quantity | Result |
|---|---:|
| panels | 10,000 |
| p <= 0.01 | 0.0054 |
| p <= 0.05 | 0.0758 |
| Wilson 95% interval at 0.05 | [0.07077, 0.08115] |
| p <= 0.10 | 0.1552 |
| materially anti-conservative | true |

The earlier 0.0733 estimate was not explained by Monte Carlo noise. Under the unchanged stress model, the row-permutation test remains materially anti-conservative. The appropriate scientific diagnosis is `NONEXCHANGEABLE_STRESS_NULL_STRUCTURE`.

## Implementation audit

The statistical implementation itself uses controller-row permutations, holds reference columns fixed, applies the same map across shells, includes one identity map, samples unique nonidentity maps, uses the right tail without sign inversion, handles tied ranks, and fails closed on degenerate Spearman inputs. The exact K=6 and future 50,000-map checks passed:

- K=6 exact rejection rate: 0.0565;
- K=6 sampled rejection rate: 0.0565;
- absolute rejection-rate difference: 0.0;
- p95 absolute p-value difference: 0.01834;
- 1,000-map versus 50,000-map rejection rates: 0.034 versus 0.034;
- p95 absolute p-value difference: 0.024455.

However, the frozen audit aggregator contains a fail-closed bookkeeping defect. It records the desired fact `dyad_level_permutation: false` and then applies `all(checks.values())`, incorrectly converting that desired false value into an audit failure. Because this was discovered after the calibration results were visible, no code, check, threshold, or output was patched or rerun. The machine result is therefore preserved exactly as `PRIMARY_ROW_QAP_IMPLEMENTATION_NOT_CALIBRATED`.

This bookkeeping defect is not needed to justify the scientific stop: the independently frozen stress-null warning also mandates principal review.

## Mechanical ruling

The V2 attempt is blocked at Phase 1:

`Q2_OOS_V2_NULL_CALIBRATION_BLOCKED`

Per the prospective protocol, no runtime autopsy, V2 PRELOCK, seed derivation, 34-controller stream, Spark-1 safety qualification, A1/A2 capture, future semantic schedule, or 19,200-row semantic execution was performed.

## Firewall and resource audit

- V1 modified: NO
- V2 streams generated: 0
- redraws: 0
- actual V2 seed derived: NO
- new semantic trajectories: 0
- correctness inspected: NO
- historical raw text inspected: NO
- historical runtime metadata inspected: NO
- Spark-1 CPU-only model-free use: YES
- Spark-1 safety trajectory count: 0
- Spark-1 A2 use: NO
- Spark 2 used: NO
- RunPod used: NO
- Q3 run: NO

The historical Q2 V4.1 result remains `Q2_V4_1_G2` with `RS+` and `RT+` and is not modified by this presemantic stop.
