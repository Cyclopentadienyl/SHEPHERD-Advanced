.PHONY: help validate test test-unit test-integration lint lint-imports format typecheck check index deploy-linux deploy-win clean
PY?=python

help:
	@echo "SHEPHERD-Advanced Makefile targets:"
	@echo ""
	@echo "  Testing"
	@echo "    make test             - Run the full test suite"
	@echo "    make test-unit        - Run unit tests only (part of 'check')"
	@echo "    make test-integration - Run integration tests (1 known failure, see docs)"
	@echo ""
	@echo "  Gates (baseline-green — these must pass)"
	@echo "    make check            - lint-imports + test-unit"
	@echo "    make lint-imports     - Enforce the layered architecture (import-linter)"
	@echo ""
	@echo "  Debt reports (NOT baseline-green — they report existing issues)"
	@echo "    make lint             - Ruff findings across src/tests/scripts"
	@echo "    make typecheck        - mypy --strict findings in src/"
	@echo "    make format           - Format with black and apply ruff --fix"
	@echo ""
	@echo "  Environment & deployment"
	@echo "    make validate         - Validate the Python/PyTorch installation"
	@echo "    make index CFG=..     - Build a vector index from a config file"
	@echo "    make deploy-linux     - Deploy on Linux x86/ARM (calls deploy.sh)"
	@echo "    make deploy-win       - Deploy on Windows x86 (calls deploy.cmd)"
	@echo "    make clean            - Remove caches and build artifacts"

# --- testing -----------------------------------------------------------------
# Note: pytest is invoked as `$(PY) -m pytest` so the repo root stays on sys.path
# and `import src...` resolves without an editable install.
test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest tests/unit

test-integration:
	$(PY) -m pytest tests/integration

# --- gates: expected to pass on the current tree ------------------------------
# `check` deliberately contains only checks that are baseline-green. Adding a
# known-failing check here would make the gate meaningless — contributors would
# learn to ignore a red `make check`, which is worse than not having one.
#
# `test-integration` is excluded: TestVectorIndexE2E::test_pipeline_with_vector_index
# fails on hosts where the cuVS backend cannot initialise, and the persisted index
# format does not match the backend chosen at load time. That failure is a real
# signal about the vector-index subsystem, which is under review — see
# docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md. It is left red on purpose
# rather than skipped, so `check` routes around it instead of hiding it.
check: lint-imports test-unit

# import-linter ships a console script rather than a runnable module, and
# .import-linter.ini is not one of its auto-discovered filenames
# (.importlinter / setup.cfg / pyproject.toml), so the config path is explicit.
lint-imports:
	lint-imports --config .import-linter.ini

# --- debt reports: these currently FAIL, by design ----------------------------
# The repository predates any lint/type gate, so `lint` and `typecheck` report a
# large backlog rather than passing. They are kept because measuring the debt is
# useful and they are the commands that will eventually become gates — but they
# are deliberately NOT in `check`, and `make help` says so. Silencing them with
# blanket ignores to force a green result was rejected: a check that passes
# without checking anything is the failure mode this repository has been
# removing elsewhere.
#
# Clearing the backlog is tracked as separate work; most Ruff findings are
# auto-fixable via `make format`, but that rewrites annotations across src/ and
# needs its own reviewed change rather than riding along with repo hygiene.
lint:
	$(PY) -m ruff check src tests scripts

typecheck:
	$(PY) -m mypy src

format:
	$(PY) -m black src tests scripts
	$(PY) -m ruff check --fix src tests scripts

# --- environment & deployment ------------------------------------------------
validate:
	$(PY) scripts/validate_installation.py

index:
	$(PY) scripts/build_index.py --config $(CFG)

deploy-linux:
	bash deploy.sh

deploy-win:
	cmd /c deploy.cmd

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
