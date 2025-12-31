.PHONY: validate index win linux linux-arm
PY?=python

validate:
	$(PY) scripts/validate_installation.py

index:
	$(PY) scripts/build_index.py --config $(CFG)

win:
	powershell -ExecutionPolicy Bypass -File scripts/deploy/windows_x86.ps1 -Config configs/deployment/windows.yaml

linux:
	bash scripts/deploy/linux_x86.sh configs/deployment/linux_x86.yaml

linux-arm:
	bash scripts/deploy/dgx_spark_arm.sh configs/deployment/linux_arm_dgx.yaml
