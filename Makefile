PYTHON ?= python

.PHONY: install check run

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) src/healthcheck.py

run:
	$(PYTHON) src/train.py --data data/raw/noaaSSTcoralData.nc
