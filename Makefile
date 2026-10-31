PYTHON ?= python3
OUT ?= results
MANIFEST ?=
REPLAY_OUT ?=

.PHONY: help gate test check sanity replay verify verify-env verify-artifacts verify-provenance verify-source

help:
	@echo "make gate             - design pins + smoke gate (python3 ci_gate.py)"
	@echo "make test             - standard-library test suite (no extra dependencies)"
	@echo "make check            - gate + tests"
	@echo "make sanity           - quick reproducibility/split/overfit pre-flight"
	@echo "make replay MANIFEST=results/e1_manifest.json [REPLAY_OUT=results_replay]"
	@echo "                      - re-run a frozen manifest as a new evidence directory"
	@echo "make verify           - env + artifact + provenance + source checks on $(OUT)"
	@echo "make verify-env       - compare $(OUT)/environment.json with this process"
	@echo "make verify-artifacts - verify $(OUT) against artifact_index.json"
	@echo "make verify-provenance - recompute each record's setup/environment ids"
	@echo "make verify-source    - compare archived code hashes with this checkout"

gate:
	$(PYTHON) ci_gate.py

test:
	$(PYTHON) -m unittest discover -s tests -v

check: gate test

sanity:
	$(PYTHON) -m cage.experiments --only sanity --quick --out results_quick

replay:
	@if [ -z "$(MANIFEST)" ]; then \
		echo "usage: make replay MANIFEST=results/e1_manifest.json [REPLAY_OUT=results_replay]"; \
		exit 2; \
	fi
	$(PYTHON) -m cage.experiments --replay-manifest "$(MANIFEST)" $(if $(REPLAY_OUT),--out "$(REPLAY_OUT)",)

verify: verify-env verify-artifacts verify-provenance verify-source

verify-env:
	$(PYTHON) -m cage.experiments --check-environment --out $(OUT)

verify-artifacts:
	$(PYTHON) -m cage.experiments --verify-artifacts --out $(OUT)

verify-provenance:
	$(PYTHON) -m cage.experiments --verify-provenance --out $(OUT)

verify-source:
	$(PYTHON) -m cage.experiments --check-source --out $(OUT)
