# Q2 OOS V2 Post-Maintenance Presemantic Closeout

## 1. Post-maintenance environment qualification

Spark 1 requalified before Qwen was loaded. The frozen scientific profile was
matched on machine class, aarch64, one NVIDIA GB10, Python 3.12.3, PyTorch
2.13.0+cu130, CUDA 13.0, Transformers 4.57.6, BF16, SDPA, exact model and
tokenizer revision `b968826d9c46dd6066d109eabc6255188de91218`, all 15 pinned
model files, code/lock hashes, clean frozen worktree, available disk, and absence
of stale A2 or unrelated GPU processes.

Result: `Q2_OOS_V2_POST_MAINTENANCE_ENVIRONMENT_QUALIFIED`.

The qualified environment profile remains
`8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386`.

## 2. Exact environment diff versus frozen qualified profile

There was no scientifically relevant drift. Two reboot/maintenance observations
changed: kernel build `6.17.0-1026-nvidia` to `6.17.0-1032-nvidia`, and NVIDIA
driver patch `580.159.03` to `580.173.02`. The frozen qualification rules do not
treat either observation as a static scientific gate. CUDA, Torch, Transformers,
BF16, SDPA, model/tokenizer bytes, architecture, GPU class, and all scientific
hashes remained exact. No package, driver, CUDA, model, or cache repair was made.

## 3. A2 capture execution

The exact label-free teacher-forced schedule ran from frozen source commit
`f2ec44c2da9f00688a918f55b0f5d70b198f4744`. It produced 12 primary and 12
repeat archives (24/24 total) over the exact 12 probes, four checkpoints per
probe, 16 selected fresh controllers, two shells, and frozen baseline recapture.
Wall time was 1,418.586 seconds. The private raw-archive aggregate SHA-256 is
`a479f6c35fb6e9da106eef989ae980b230ecc4d88abce29a295f20e085e2dd66`.

Raw captures remain outside Git and are identified individually and jointly by
`A2_FRESH_RAW_ARCHIVE_HASHES.json`. This phase used Qwen inference but no free
generation, no correctness, and no semantic outcome.

## 4. Repeat / baseline / forensic checks

Both shells passed every frozen A2 gate. Repeat radius and distance errors were
zero and repeat angular rank correlation was 1.0. Fresh-versus-historical
baseline recapture differed by 0.0. Historical A2/D2 reproduction differed by
at most `1.11e-15`. PSD, symmetry, diagonal, radius floor, and cosine-bound gates
all passed.

The independent implementation fully recomputed A0 and A1 and compared A2/D2
against a scalar natural-log JS reference on frozen raw-array subsets. Its
maximum primary/audit difference was `1.233e-11`, within the frozen tolerance.

Result: `Q2_OOS_V2_LABEL_FREE_FORENSIC_CLEAN`.

## 5. A0/A1/A2/D2 consolidation

The frozen consolidator used natural-log Jensen-Shannon divergence, mixture
weights 0.5/0.5, uniform mean over 48 probe/checkpoint rows, and controller order
fresh-16 then historical-reference-31. It sealed MEDIUM and STRONG fresh×fresh
and fresh×reference blocks for A0, A1, A2, and D2.

- A0: `QUALIFIED_BY_SELECTED_BANK_GATE`
- A1: `Q2_OOS_V2_A1_INSTRUMENT_QUALIFIED`
- A2: `Q2_OOS_V2_A2_INSTRUMENT_QUALIFIED`
- D2: `SECONDARY_SEALED`
- Aggregate matrix archive SHA-256:
  `b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703`

Overall result: `Q2_OOS_V2_LABEL_FREE_INSTRUMENT_QUALIFIED`.

## 6. Final selected-bank geometry

The immutable first-16-safe bank is directions 01–09, 11, 13–18 in original
stream order. It has rank 8, effective rank 6.8420, stable rank 3.5962,
descriptive condition number 2.6681, maximum absolute pair cosine 0.8148,
fresh×reference A0 q90−q10 of 0.9591, and shell-amplitude CV `9.96e-7`.
Every prospectively frozen selected-bank gate passed. No controller changed,
was redrawn, or was replaced.

## 7. Inference lock

The primary external-validity unit is one prospectively sampled,
safety-conditioned fresh controller. For each fresh controller, the frozen row
statistic averages its MEDIUM and STRONG Spearman associations against the 31
fixed historical references. All 16 statistics must be finite; at least 12/16
must be positive; inference is the exact one-sided binomial sign test for
`P(r_i > 0) > 0.5`.

Global fresh×reference rho is descriptive, the historical row-QAP is diagnostic,
and the studentized controller mean is sensitivity-only. Fresh×fresh uses the
node-jackknife pseudovalue t procedure as a secondary that cannot rescue the
primary. Item bootstrap uses 50,000 resamples; LOFO is sensitivity-only.

## 8. Efficient semantic schedule/runtime forecast

The future schedule contains exactly 16 controllers × 2 shells × 300 items × 2
rollouts = 19,200 unique logical rows and 19,200 unique seeds. Each item-rollout
block has a deterministic PCG64DXSM permutation of all 32 conditions. The
generation contract retains `max_new_tokens=4096` and the prospectively frozen
`EXTREME_MECHANICAL_REPETITION_V1` terminal policy.

The normal runtime forecast is mean 9.90 h, P50 9.76 h, P80 11.05 h, P90 11.81 h,
P95 12.45 h, and P99 13.75 h. Stress P95 is 13.63 h at 1.5× and 14.84 h at 2×.
These forecasts are monitoring-only and cannot change the science.

The semantic schedule is frozen but explicitly **not authorized and not run**.

## 9. Prediction-lock hashes

- Selected bank: `9a544b4ec6d43ec1c3530feb963cd0340db516e82f91a40c2624300483e2e0fd`
- Future semantic schedule: `dac5c284b90c726016968f31d25200a362c42d96f63b63d730665f3f47e85ec5`
- Inference lock: `a8d9ead49d9265211906a0f367ac3062d03b32c924e8606a2f8c12caaf3fbea1`
- Runtime monitor lock: `fadafb50b9c26c42bfb2abd4dcdeb7a93870ac46c3a1e0925b4ba8fc3707ea8d`
- Prediction matrices: `b4ec00985e750c5bb8fd7fd49228267ec576bf6c2ad2ac3984f6f2390d927703`
- Prediction lock: `825d6e3536b51a31956cbd5c9e75bedfed38f9e3df5da05a4452a5681f65f9bb`
- Prediction-lock hash manifest: `2b02c2a6e0fa14a1d6760e384d726787d54c3e8c66b1d81585be914e500e9f68`

## 10. Repository/resource state

- Spark 1 post-maintenance qualified: **YES**
- Qwen A2 model inference performed: **YES**
- Exact A2 capture count: **24 archives (12 primary + 12 repeat)**
- Correctness inspected: **NO**
- N=300 semantic trajectories: **0**
- Selected 16 changed: **NO**
- New controller stream generated: **NO**
- Spark 2 used: **NO**
- RunPod used: **NO**
- Q3 run: **NO**

Historical V1/V2 blocked states and closed Q2 V4.1 G2/RS+/RT+ remain immutable.
The only permissible next step is principal review of this prediction-lock
package. Semantic execution requires a separate explicit authorization.

`Q2_OOS_V2_READY_FOR_PREDICTION_LOCK`
