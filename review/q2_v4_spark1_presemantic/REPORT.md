# Q2 V4 — Spark-1 presemantic qualification closeout

Classification: `Q2_V4_SAFE_BANK_INSUFFICIENT`
Forensic classification: `Q2_V4_PRESEMANTIC_FORENSIC_CLEAN`

## Scientific boundary

This sprint stopped at the prospectively frozen bank-level safety gate. It is an
instrument non-qualification, not a predictive Q2 result. The 300-item semantic
panel was never executed, semantic outcomes remain zero, and Q3 remains not run.

## Engine and source basis

The Spark-1 engine qualified. All eight native source directions qualified. The
native source matrix retained rank 8, condition number 2.735664, entropy effective
rank 6.587926, and equal 0.25 leverage for each of the four concepts.

## Unique PRELOCK and candidate stream

- PRELOCK: `99782d6f4f3ce1ca52d2cf6caeacafd4d0de9081`
- Candidate-bank commit: `c82c1cb79392f9a5d9bd9e8d258a1d1b54e8fd41`
- RNG: NumPy `PCG64DXSM`
- Seed (128-bit, big-endian): `164758257368056574741665736289526988272` (`7bf34940ccce6c38442f1cdc728bbdf0`)
- Candidates generated exactly once: 40
- Redraw permitted: NO
- Algebraic gate: PASS (rank 8; condition 1.906976; effective rank 7.291781)

## Label-free shell safety

The complete 1,944-row matched schedule ran with 1,944 unique logical keys, zero
duplicates, and no correctness access. Baseline validity/evaluability were both
1.000000. Exactly 31 of 40 candidates passed both the
medium (implemented radius 0.25) and strong (0.50) frozen gates. The required count
was 32, so the first-32-safe bank could not be formed.

Safe candidates in frozen generation order:

- `V4_DIRECTION_00`
- `V4_DIRECTION_01`
- `V4_DIRECTION_02`
- `V4_DIRECTION_03`
- `V4_DIRECTION_04`
- `V4_DIRECTION_06`
- `V4_DIRECTION_07`
- `V4_DIRECTION_08`
- `V4_DIRECTION_09`
- `V4_DIRECTION_10`
- `V4_DIRECTION_11`
- `V4_DIRECTION_13`
- `V4_DIRECTION_15`
- `V4_DIRECTION_17`
- `V4_DIRECTION_18`
- `V4_DIRECTION_19`
- `V4_DIRECTION_20`
- `V4_DIRECTION_22`
- `V4_DIRECTION_23`
- `V4_DIRECTION_24`
- `V4_DIRECTION_26`
- `V4_DIRECTION_28`
- `V4_DIRECTION_29`
- `V4_DIRECTION_30`
- `V4_DIRECTION_31`
- `V4_DIRECTION_32`
- `V4_DIRECTION_33`
- `V4_DIRECTION_34`
- `V4_DIRECTION_35`
- `V4_DIRECTION_37`
- `V4_DIRECTION_39`

No candidate 41+ was generated. Thresholds were not altered. The bank was not
regenerated or optimized.

## Downstream pre-outcome geometry

A1 covariance capture: NOT RUN.
A2 fingerprint capture: NOT RUN.
A0/A1/A2/D2 matrices: NOT CREATED.
QAP schedule: NOT CREATED.
Future 39,000-row semantic schedule: NOT CREATED.
Prediction lock: NOT CREATED by the frozen stop rule.

The future endpoint definitions (D-total, N/(N-1)-corrected D-shape, R-total, and
R-shape) remain protocol definitions only; no error outcome exists for V4.

## Throughput and resources

- Source: 384 rows, 62435 tokens,
  1.503 measured GPU-hours.
- Safety: 1944 rows, 103755 tokens,
  2.545 measured GPU-hours.
- Total measured phase time excluding model loads: 4.050
  GPU-hours; approximate occupancy including loads/setup: 4.2 GPU-hours.
- Spark 1 used: YES. Spark 2 used: NO. RunPod resources: ZERO.

Had the bank qualified, the measured safety rate would project the 39,000-row future
run at 51.0 hours, or
76.6 hours with the frozen
50% tail margin. That execution is not ready or authorized.

## Forensic audit

An independent raw-journal recomputation reproduced every shell metric and the
31-safe classification exactly (maximum absolute metric difference 0). It verified
the immutable schedule, seed bindings, no correctness access, zero semantic outcomes,
and absence of A1/A2/prediction artifacts.

## Next action

`Q2_V4_SAFE_BANK_INSUFFICIENT` — principal-researcher review. No scientific rescue,
semantic execution, or Q3 transition is authorized.
