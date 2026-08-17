PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)

.PHONY: test lint smoke doctor preflight preflight-v1-1 preflight-v1-2 tiny-smoke predeploy storage-check q1-v3-gate q1-v3-design q1-v3-engine-benchmark

test:
	$(PYTHON) -m pytest -q

lint:
	$(RUFF) check .

smoke:
	$(PYTHON) -m epistemic_geometry.cli run configs/mock_smoke.yaml

doctor:
	$(PYTHON) -m epistemic_geometry.cli doctor

preflight:
	$(PYTHON) -m epistemic_geometry.cli preflight configs/mock_smoke.yaml

preflight-v1-1:
	$(PYTHON) -m epistemic_geometry.cli preflight-q1-v1-1 configs/q1_v1_1_qwen3_8b.yaml

preflight-v1-2:
	$(PYTHON) -m epistemic_geometry.cli preflight-q1-v1-2 configs/q1_v1_2_qwen3_8b.yaml

tiny-smoke:
	$(PYTHON) -m epistemic_geometry.cli run configs/tiny_transformer_smoke.yaml

predeploy:
	bash scripts/predeploy_gate.sh

storage-check:
	$(PYTHON) -m epistemic_geometry.cli storage-check

q1-v3-gate:
	$(PYTHON) scripts/run_q1_v3_structural_gate.py --n-per-cell 5000

q1-v3-design:
	$(PYTHON) scripts/build_q1_v3_design_artifact.py

q1-v3-engine-benchmark:
	$(PYTHON) scripts/benchmark_q1_v3_reasoning_engines.py
