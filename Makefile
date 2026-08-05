.PHONY: help validate test test-unit test-integration lint lint-imports format typecheck check index deploy-linux deploy-win clean
PY?=python

help:
	@echo "SHEPHERD-Advanced Makefile targets:"
	@echo ""
	@echo "  Testing"
	@echo "    make test             - Run the full test suite"
	@echo "    make test-unit        - Run unit tests only"
	@echo "    make test-integration - Run integration tests only"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint             - Lint with ruff (no changes written)"
	@echo "    make lint-imports     - Enforce the layered architecture (import-linter)"
	@echo "    make format           - Format with black and apply ruff fixes"
	@echo "    make typecheck        - Type-check src/ with mypy"
	@echo "    make check            - lint + lint-imports + typecheck + test"
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

# --- code quality ------------------------------------------------------------
lint:
	$(PY) -m ruff check src tests scripts

# import-linter ships a console script rather than a runnable module, and
# .import-linter.ini is not one of its auto-discovered filenames
# (.importlinter / setup.cfg / pyproject.toml), so the config path is explicit.
lint-imports:
	lint-imports --config .import-linter.ini

format:
	$(PY) -m black src tests scripts
	$(PY) -m ruff check --fix src tests scripts

typecheck:
	$(PY) -m mypy src

check: lint lint-imports typecheck test

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
