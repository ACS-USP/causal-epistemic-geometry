PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)

.PHONY: test lint smoke doctor preflight preflight-v1-1 tiny-smoke predeploy storage-check

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

tiny-smoke:
	$(PYTHON) -m epistemic_geometry.cli run configs/tiny_transformer_smoke.yaml

predeploy:
	bash scripts/predeploy_gate.sh

storage-check:
	$(PYTHON) -m epistemic_geometry.cli storage-check
