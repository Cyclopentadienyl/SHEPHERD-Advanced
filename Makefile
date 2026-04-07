.PHONY: validate index test deploy-linux deploy-win clean help
PY?=python

help:
	@echo "SHEPHERD-Advanced Makefile targets:"
	@echo "  make validate     - Validate Python/PyTorch installation"
	@echo "  make test         - Run pytest integration tests"
	@echo "  make index CFG=.. - Build vector index from config"
	@echo "  make deploy-linux - Deploy on Linux x86/ARM (calls deploy.sh)"
	@echo "  make deploy-win   - Deploy on Windows x86 (calls deploy.cmd)"

validate:
	$(PY) scripts/validate_installation.py

test:
	$(PY) -m pytest tests/integration/test_pipeline.py -v

index:
	$(PY) scripts/build_index.py --config $(CFG)

deploy-linux:
	bash deploy.sh

deploy-win:
	cmd /c deploy.cmd
