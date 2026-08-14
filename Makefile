PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)

.PHONY: test lint smoke doctor

test:
	$(PYTHON) -m pytest -q

lint:
	$(RUFF) check .

smoke:
	$(PYTHON) -m epistemic_geometry.cli run configs/mock_smoke.yaml

doctor:
	$(PYTHON) -m epistemic_geometry.cli doctor
