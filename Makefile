PYTHON ?= python3

.PHONY: help gate test check sanity

help:
	@echo "make gate    - design pins + smoke gate (python3 ci_gate.py)"
	@echo "make test    - standard-library test suite (no extra dependencies)"
	@echo "make check   - gate + tests"
	@echo "make sanity  - quick reproducibility/split/overfit pre-flight"

gate:
	$(PYTHON) ci_gate.py

test:
	$(PYTHON) -m unittest discover -s tests -v

check: gate test

sanity:
	$(PYTHON) -m cage.experiments --only sanity --quick --out results_quick
