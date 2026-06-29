PYTHON ?= $(shell if [ -x ./.venv/bin/python ]; then echo ./.venv/bin/python; elif [ -x ./.conda/bin/python ]; then echo ./.conda/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)

.PHONY: install check run

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) src/healthcheck.py

run:
	$(PYTHON) src/train.py --data data/raw/noaaSSTcoralData.nc
