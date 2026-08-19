# Q1 V4 dense-code micro-pilot

Status: **BLOCKED BEFORE GPU — `DENSE_CODE_PILOT_BLOCKED_BY_EVALUATOR`**

An authorized follow-up attempted to provision the required disposable Docker
sandbox on macOS. Homebrew installed Colima, Docker CLI, and the QEMU fallback,
but neither the `vz` nor `qemu` Colima profile remained alive long enough to
expose a Docker daemon. The follow-up therefore stopped with
`DENSE_CODE_PILOT_BLOCKED_BY_SANDBOX` and zero GPU spend. See the provisioning
and validation reports under `review/q1_dense_code_pilot/`.

This branch records the model-free gate for the proposed five-problem, two-seed
baseline-only code-generation pilot. No Qwen inference, activation extraction,
steering, geometry, holdout access, or RunPod start is part of this result.

## What was audited

The historical V4 audit at
`review/q1_v4_microbench/DENSE_CODE_FAILURE_VECTOR_AUDIT.md` remains unchanged.
It correctly records that the normalized LiveCodeBench fixture has one final
reference answer per row, not a verified official per-test-case execution
artifact. The existing `ProgramOutcome` type is only a schema fixture; it is not
an evaluator and must not be treated as one.

The official EvalPlus `v0.3.1` implementation was inspected as the next
prospective fallback. It is attractive instrument-wise: with `test_details`,
its evaluator returns a deterministic boolean detail vector for the base and
plus test inputs. That is sufficient to reconstruct a stable nested failure
vector without treating test cases as independent statistical observations.

References: [EvalPlus execution guidance](https://github.com/evalplus/evalplus/blob/v0.3.1/docs/execution.md),
[EvalPlus evaluator source](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/evaluate.py).

However, EvalPlus also explicitly warns that its `reliability_guard` is **not a
security sandbox**. Its normal local path runs generated code in a child
process with time and resource limits, but does not by itself establish the
required no-network, no-secrets, no-arbitrary-filesystem boundary. The official
documentation recommends Docker for safe execution.

The current Mac has no Docker, Podman, or Firejail executable. A local
`sandbox-exec` profile has not been validated as an equivalent evaluator for the
Linux RunPod execution environment and is therefore not substituted silently.

## Decision

The pilot cannot safely select five items or execute generated programs yet.
This is a pre-GPU evaluator gate, not a benchmark failure and not a scientific
result. No items were selected, no model outputs exist, and no cost was spent.

The next authorized implementation step is to provide one of:

1. the official EvalPlus Docker image and a verified remote execution wrapper
   that emits the full per-test detail vector; or
2. another already audited evaluator with equivalent isolation and deterministic
   per-test outcomes.

After that gate passes, the exact five-item selection rule, frozen manifest,
Qwen generation policy, and US$0.50 hard cost ceiling from the principal review
remain unchanged. No outcome-dependent benchmark selection is permitted.

## Scientific firewall

- Steering: not run.
- PCA or activation directions: not constructed.
- Geometry: not run.
- LiveBench, CRUXEval, and character-count instruments: not reopened.
- Confirmatory data: untouched.
- Q1 V3 and all historical V4 artifacts: preserved.
