PYTHON ?= python3
OUT ?= results

.PHONY: help gate test check sanity verify verify-env verify-artifacts verify-provenance

help:
	@echo "make gate             - design pins + smoke gate (python3 ci_gate.py)"
	@echo "make test             - standard-library test suite (no extra dependencies)"
	@echo "make check            - gate + tests"
	@echo "make sanity           - quick reproducibility/split/overfit pre-flight"
	@echo "make verify           - env + artifact + provenance checks on $(OUT)"
	@echo "make verify-env       - compare $(OUT)/environment.json with this process"
	@echo "make verify-artifacts - verify $(OUT) against artifact_index.json"
	@echo "make verify-provenance- recompute each record's setup/environment ids"

gate:
	$(PYTHON) ci_gate.py

test:
	$(PYTHON) -m unittest discover -s tests -v

check: gate test

sanity:
	$(PYTHON) -m cage.experiments --only sanity --quick --out results_quick

verify: verify-env verify-artifacts verify-provenance

verify-env:
	$(PYTHON) -m cage.experiments --check-environment --out $(OUT)

verify-artifacts:
	$(PYTHON) -m cage.experiments --verify-artifacts --out $(OUT)

verify-provenance:
	$(PYTHON) -m cage.experiments --verify-provenance --out $(OUT)
