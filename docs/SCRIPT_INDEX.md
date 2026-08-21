# Script classification

This inventory prevents old experiment entry points from masquerading as the
current action. No historical reproduction script is deleted by this reset.

## CURRENT CANONICAL

- `scripts/render_project_state.py` — generated status source.
- `scripts/check_docs.py` — offline document/state audit.
- `scripts/check_experiment_registry.py` — experiment-registry audit.
- `scripts/validate_scientific_metrics.py` — synthetic estimand validation.
- `scripts/validate_research_os.py` — policy, lifecycle, contract, and named-environment audit.
- `scripts/prepare_gate6_3_semantic_validity_audit.py` — freezes the condition-masked V3 audit corpus.
- `scripts/analyze_gate6_3_semantic_validity_audit.py` — local-only condition-symmetric V3 reanalysis.
- `scripts/validate_gate6_3_semantic_validity_audit.py` — fail-closed V3 artifact validator.
- `scripts/reanalyze_q1_v4_geometry.py` — corrected, artifact-only V4 analysis.
- `scripts/run_mock_smoke.sh` — network-free package smoke.
- `scripts/bootstrap_runpod.sh` — conservative remote environment bootstrap;
  not execution authorization.
- `scripts/predeploy_gate.sh` — generic predeployment gate.
- `scripts/sync_to_runpod.sh`, `scripts/sync_from_runpod.sh` — scoped artifact/source
  synchronization; use only under a separately authorized protocol.

## HISTORICAL REPRODUCTION

- `scripts/recompute_v1_v2_from_raw.py`
- `scripts/build_q1_v2_calibration_manifest.py`
- `scripts/build_q1_v2_design_artifact.py`
- `scripts/build_q1_v2_instrument_review.py`
- `scripts/recompute_q1_v2_instrument.py`
- `scripts/run_q1_v2_instrument_calibration.py`
- `scripts/run_e3_structural_gate.py`
- `scripts/audit_e3_tokenization.py`
- `scripts/build_q1_v3_calibration_manifests.py`
- `scripts/build_q1_v3_design_artifact.py`
- `scripts/run_q1_v3_calibration.py`
- `scripts/run_q1_v3_structural_gate.py`
- `scripts/analyze_q1_v3_stage_a.py`
- `scripts/validate_q1_v3_stage_a.py`
- `scripts/benchmark_q1_v3_reasoning_engines.py`
- `scripts/benchmark_q1_v3_reasoning_qwen.py`
- `scripts/prepare_external_benchmark.py`
- `scripts/validate_external_benchmarks.py`
- `scripts/run_external_qualification.py`
- `scripts/analyze_external_qualification.py`
- `scripts/run_completion_diagnostics.py`
- `scripts/analyze_completion_diagnostics.py`
- `scripts/report_completion_diagnostics.py`
- `scripts/finalize_completion_diagnostic.py`
- `scripts/reclassify_low_cap_runs.py`
- `scripts/prepare_q1_v4_microbench.py`
- `scripts/run_q1_v4_character.py`
- `scripts/run_q1_v4_geometry.py`
- `scripts/analyze_q1_v4_microbench.py` — corrected for future reruns; original
  outputs remain preserved.
- `scripts/run_q1_v4_charcount_postmortem.py`
- `scripts/analyze_q1_v4_charcount_followup.py`
- `scripts/finalize_q1_v4_report.py`
- `scripts/prepare_q1_dense_code_pilot.py`

## THIN WRAPPER

- `scripts/run_q1_smoke.sh` — invokes a named Q1 configuration.
- `scripts/runpod_preflight.sh` — invokes doctor/preflight checks.
- `scripts/runpod_environment.sh` — prints safe environment metadata.
- `scripts/before_pod_stop.sh` — invokes artifact-recovery checks.
- `scripts/check_runpod_connection.sh` — SSH connectivity only.
- `scripts/configure_runpod_ssh.sh` — scoped SSH alias helper.
- `scripts/qwen3_technical_smoke.py` — technical model-path smoke, not science.
- `scripts/profile_tiny_engines.py` — tiny software profile.

## DUPLICATED / MERGEABLE

The completion-diagnostic sequence—`run_completion_diagnostics.py`,
`analyze_completion_diagnostics.py`, `report_completion_diagnostics.py`, and
`finalize_completion_diagnostic.py`—is intentionally preserved but should be
replaced by one package workflow if that instrument family is ever reopened.

The V2/V3 build/analyze/validate scripts duplicate manifest and report plumbing.
New protocols should use package-level registry/state/artifact helpers rather
than copy another script family. They are not merged now because doing so would
risk historical reproduction for no current scientific benefit.

## DEAD / UNSAFE TO DELETE

No script is declared safely dead. The historical entry points above encode
frozen artifact schemas and source identities. Deleting them would weaken audit
and reanalysis capability; invoking them as current protocol would also be
unsafe. Their classification—not deletion—is the conservative resolution.
