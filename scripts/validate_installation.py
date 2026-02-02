"""
SHEPHERD-Advanced Installation Validator
========================================
Functionality:
  - Validate PyTorch and CUDA installation
  - Check vector index backends (Voyager, cuVS)
  - Verify optional accelerators (xformers, flash-attn)
  - Generate JSON report of installation status

Path:
  - Relative: scripts/validate_installation.py
  - Absolute: SHEPHERD-Advanced/scripts/validate_installation.py

Input:
  - None (reads environment and installed packages)

Output:
  - JSON report to stdout with errors, warnings, and info
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

REPORT: dict = {"errors": [], "warnings": [], "info": []}


def add(level: str, msg: str) -> None:
    REPORT[level].append(msg)


def check_torch(min_ver: str = "2.9.0") -> None:
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        from packaging import version

        add("info", f"Torch version: {torch.__version__}")
        if version.parse(torch.__version__) < version.parse(min_ver):
            add("errors", f"Torch >= {min_ver} required, found {torch.__version__}")

        if torch.cuda.is_available():
            add("info", f"CUDA available: {torch.version.cuda}")
            try:
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    add("info", "SDPA (Scaled Dot Product Attention): Available")
                else:
                    add("warnings", "SDPA: Not available in this Torch version")
            except Exception as e:
                add("warnings", f"SDPA probe failed: {e}")
        else:
            add("warnings", "CUDA not available (running on CPU)")
    except Exception as e:
        add("errors", f"import torch failed: {e}")


def check_pkg(name: str, required: bool = False) -> bool:
    """Check if a package is importable."""
    try:
        importlib.import_module(name)
        add("info", f"{name}: OK")
        return True
    except Exception as e:
        level = "errors" if required else "warnings"
        add(level, f"{name}: not available ({e})")
        return False


def check_vector_backends() -> None:
    """Check vector index backends (Voyager, cuVS)."""
    # Voyager is the cross-platform fallback (required)
    voyager_ok = check_pkg("voyager", required=True)
    if not voyager_ok:
        add("errors", "Voyager not installed - required for vector search")
        add("info", "Install with: pip install voyager>=2.0")

    # cuVS is optional (Linux GPU only)
    if sys.platform != "win32":
        cuvs_ok = check_pkg("cuvs", required=False)
        if cuvs_ok:
            add("info", "cuVS: GPU-accelerated vector search available")
        else:
            add("info", "cuVS: not installed (optional, Linux GPU only)")
            add("info", "For GPU acceleration: pip install --extra-index-url https://pypi.nvidia.com cuvs-cu12")
    else:
        add("info", "cuVS: skipped (not supported on Windows)")


def check_flash_attn() -> None:
    """Check FlashAttention installation."""
    if os.environ.get("FLASHATTN_FORCE_DISABLE"):
        add("info", "FLASHATTN_FORCE_DISABLE set; skipping flash-attn check")
        return

    # Skip on ARM (not supported)
    if platform.machine().lower() in {"aarch64", "arm64"}:
        add("info", "flash-attn: skipped (not supported on ARM)")
        return

    try:
        flash_attn = importlib.import_module("flash_attn")
        add("info", f"flash-attn: OK (v{flash_attn.__version__})")
    except Exception as e:
        add("warnings", f"flash-attn: not available ({e})")


def main() -> int:
    add("info", f"Python: {sys.version.split()[0]}")
    add("info", f"OS: {platform.system().lower()}  Arch: {platform.machine()}")

    # Core checks
    check_torch()

    # Vector index backends
    check_vector_backends()

    # Optional accelerators
    check_pkg("xformers", required=False)
    check_flash_attn()

    # Print report
    print(json.dumps(REPORT, indent=2))

    # Return non-zero if there are errors
    return 1 if REPORT["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
