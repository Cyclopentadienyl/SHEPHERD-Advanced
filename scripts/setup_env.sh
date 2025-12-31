\
    #!/usr/bin/env bash
    set -euo pipefail
    CONFIG=${1:-configs/deployment/linux_x86.yaml}
    echo "Using config: $CONFIG"
    python -m pip install -r requirements_arm.txt || true
    python scripts/validate_installation.py
