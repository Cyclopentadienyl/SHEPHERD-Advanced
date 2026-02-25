#!/usr/bin/env python3
"""
SHEPHERD-Advanced PyTorch & PyG Installation Script
====================================================
Platform-specific installation for PyTorch and PyTorch Geometric.

Script: scripts/install_pytorch_pyg.py
Absolute Path: /home/user/SHEPHERD-Advanced/scripts/install_pytorch_pyg.py

Usage:
    python scripts/install_pytorch_pyg.py              # Auto-detect platform
    python scripts/install_pytorch_pyg.py --cuda 12.4  # Specify CUDA version
    python scripts/install_pytorch_pyg.py --cpu        # CPU-only mode
    python scripts/install_pytorch_pyg.py --check      # Check current installation

Supported Platforms (all using PyTorch 2.9.0 + cu130):
    - Windows x86_64 (CUDA 12.8 → cu130, CUDA 13.0)
    - Linux x86_64 (CUDA 12.8 → cu130, CUDA 13.0)
    - Linux ARM64 / DGX Spark (CUDA 13.0 native)

Based on: https://github.com/pyg-team/pyg-lib (pyg-lib compatibility matrix)
Version: 2.1.0
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PlatformInfo:
    """Platform detection results"""
    os_name: str  # windows, linux, darwin
    arch: str  # x86_64, arm64, aarch64
    python_version: str
    cuda_version: Optional[str]
    is_conda: bool


@dataclass
class InstallConfig:
    """Installation configuration"""
    torch_version: str
    cuda_suffix: str  # cu121, cu124, cu128, cpu
    pyg_version: str
    extra_index_url: str
    pyg_wheel_url: str


# Known compatible versions (as of pyg-lib compatibility matrix 2026-01)
# Reference: https://github.com/pyg-team/pyg-lib
#
# Strategy: Use PyTorch 2.9.0 + cu130 for ALL platforms
# - Unified version across Windows x86, Linux x86, DGX Spark ARM
# - Better ecosystem compatibility than bleeding-edge 2.10
# - CUDA 13.0 native support for latest hardware features
#
# PyTorch 2.9 Support Matrix:
#   Linux:   cpu ✓, cu126 ✓, cu128 ✓, cu129 ✓, cu130 ✓
#   Windows: cpu ✓, cu126 ✓, cu128 ✓
#
TORCH_CUDA_MAP = {
    # CUDA version -> (torch_version, torch_cuda_suffix, pyg_wheel_suffix)
    # Primary target: PyTorch 2.9.0 + cu130 (unified across all platforms)
    "13.0": ("2.9.0", "cu130", "cu130"),  # Primary: All platforms use cu130
    "12.8": ("2.9.0", "cu130", "cu130"),  # Map 12.8 to cu130 for forward compat
    "12.6": ("2.9.0", "cu126", "cu126"),  # Fallback for older CUDA
    "cpu": ("2.9.0", "cpu", "cpu"),
}

# PyG wheel base URL
PYG_WHEEL_BASE = "https://data.pyg.org/whl"


def detect_platform() -> PlatformInfo:
    """Detect current platform and CUDA version"""
    os_name = platform.system().lower()
    if os_name == "darwin":
        os_name = "macos"

    # Architecture detection
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Check if conda
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))

    # CUDA detection
    cuda_version = detect_cuda_version()

    return PlatformInfo(
        os_name=os_name,
        arch=arch,
        python_version=py_version,
        cuda_version=cuda_version,
        is_conda=is_conda,
    )


def detect_cuda_version() -> Optional[str]:
    """Detect installed CUDA version"""
    # Try nvcc first
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # nvidia-smi shows CUDA version in the full output
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check environment variable
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        version_file = os.path.join(cuda_home, "version.txt")
        if os.path.exists(version_file):
            with open(version_file) as f:
                match = re.search(r"(\d+\.\d+)", f.read())
                if match:
                    return match.group(1)

    return None


def get_install_config(cuda_version: Optional[str], force_cpu: bool = False) -> InstallConfig:
    """Get installation configuration for the given CUDA version"""
    if force_cpu or cuda_version is None:
        key = "cpu"
    else:
        # Find closest supported version
        major_minor = cuda_version
        if major_minor not in TORCH_CUDA_MAP:
            # Try major version only
            major = cuda_version.split(".")[0]
            for k in TORCH_CUDA_MAP:
                if k.startswith(major):
                    major_minor = k
                    break
            else:
                # Fall back to latest supported
                major_minor = "12.8"
                print(f"Warning: CUDA {cuda_version} not explicitly supported, using {major_minor} wheels")
        key = major_minor

    torch_ver, cuda_suffix, pyg_suffix = TORCH_CUDA_MAP[key]

    if cuda_suffix == "cpu":
        extra_index = "https://download.pytorch.org/whl/cpu"
        pyg_wheel = f"{PYG_WHEEL_BASE}/torch-{torch_ver}+cpu.html"
    else:
        extra_index = f"https://download.pytorch.org/whl/{cuda_suffix}"
        pyg_wheel = f"{PYG_WHEEL_BASE}/torch-{torch_ver}+{pyg_suffix}.html"

    return InstallConfig(
        torch_version=torch_ver,
        cuda_suffix=cuda_suffix,
        pyg_version="2.6.1",  # Latest stable
        extra_index_url=extra_index,
        pyg_wheel_url=pyg_wheel,
    )


def check_installation() -> Dict[str, Optional[str]]:
    """Check current PyTorch and PyG installation"""
    result = {}

    # Check PyTorch
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
    except ImportError:
        result["torch"] = None

    # Check PyG
    try:
        import torch_geometric
        result["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        result["torch_geometric"] = None

    # Check PyG extensions
    for ext in ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster"]:
        try:
            mod = __import__(ext)
            result[ext] = getattr(mod, "__version__", "installed")
        except ImportError:
            result[ext] = None

    return result


@dataclass
class InstallStep:
    """A single installation step with criticality level."""
    label: str
    cmd: List[str]
    critical: bool  # If True, failure aborts the entire install


def build_install_steps(config: InstallConfig, upgrade: bool = True) -> List[InstallStep]:
    """Build installation steps with criticality levels.

    Installation strategy (smart adaptation):
      1. PyTorch stack          - torch, torchvision, torchaudio    [CRITICAL]
      2. torch_geometric        - main PyG package (pure Python)    [CRITICAL]
      3. pyg-lib               - pyg-team native ops (has wheels)   [OPTIONAL]
      4. torch_scatter/sparse  - third-party extensions (may lag)   [OPTIONAL]

    Steps 3-4 are optional: PyG auto-falls back to torch.scatter_reduce
    (PyTorch 2.0+ native) for all operations our model uses.
    """
    steps = []
    upgrade_flag = ["--upgrade"] if upgrade else []

    # [CRITICAL] PyTorch stack
    steps.append(InstallStep(
        label="PyTorch stack",
        cmd=[
            sys.executable, "-m", "pip", "install",
            *upgrade_flag,
            f"torch=={config.torch_version}",
            "torchvision==0.24.0",
            "torchaudio==2.9.0",
            "--index-url", config.extra_index_url,
        ],
        critical=True,
    ))

    # [CRITICAL] PyG main package (pure Python, always installs)
    steps.append(InstallStep(
        label="PyTorch Geometric",
        cmd=[
            sys.executable, "-m", "pip", "install",
            *upgrade_flag,
            f"torch_geometric>={config.pyg_version}",
        ],
        critical=True,
    ))

    # [OPTIONAL] pyg-lib (pyg-team maintained, has torch 2.9+cu130 wheels)
    # --only-binary :all: prevents pip from downloading source tarballs and
    # attempting to compile (which fails with long tracebacks when no wheel exists)
    steps.append(InstallStep(
        label="pyg-lib (native GNN kernels)",
        cmd=[
            sys.executable, "-m", "pip", "install",
            *upgrade_flag,
            "--only-binary", ":all:",
            "pyg_lib",
            "-f", config.pyg_wheel_url,
        ],
        critical=False,
    ))

    # [OPTIONAL] Third-party scatter/sparse extensions (wheels often lag)
    steps.append(InstallStep(
        label="PyG native extensions (torch_scatter, torch_sparse, torch_cluster)",
        cmd=[
            sys.executable, "-m", "pip", "install",
            *upgrade_flag,
            "--only-binary", ":all:",
            "torch_scatter", "torch_sparse", "torch_cluster",
            "-f", config.pyg_wheel_url,
        ],
        critical=False,
    ))

    return steps


def build_install_commands(config: InstallConfig, upgrade: bool = True) -> List[List[str]]:
    """Build installation commands (legacy interface for dry-run display)."""
    return [step.cmd for step in build_install_steps(config, upgrade)]


def run_install(commands: List[List[str]], dry_run: bool = False) -> bool:
    """Run installation using the new step-based system."""
    # Legacy interface: convert to steps for backward compat with dry_run
    if dry_run:
        for cmd in commands:
            print(f"\n[DRY RUN] Running: {' '.join(cmd)}")
        return True

    # Build proper steps to get criticality info
    # (caller should use run_install_steps directly for production use)
    for cmd in commands:
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed with code {e.returncode}")
            return False
    return True


def run_install_steps(steps: List[InstallStep]) -> bool:
    """Run installation steps with smart fallback for optional components."""
    optional_skipped = []

    for step in steps:
        print(f"\n{'=' * 60}")
        print(f"  {'[REQUIRED]' if step.critical else '[OPTIONAL]'} {step.label}")
        print(f"{'=' * 60}")
        print(f"  Command: {' '.join(step.cmd)}")

        try:
            subprocess.run(step.cmd, check=True, capture_output=False)
            print(f"\033[92m  [OK] {step.label} installed successfully\033[0m")
        except subprocess.CalledProcessError as e:
            if step.critical:
                print(f"\033[91m  [ERROR] {step.label} failed (exit code {e.returncode})\033[0m")
                return False
            else:
                print(f"\033[93m  [SKIP] {step.label} not available for this platform/torch version\033[0m")
                print(f"         This is expected and non-critical.")
                print(f"         PyG will use PyTorch-native fallbacks (torch.scatter_reduce).")
                optional_skipped.append(step.label)

    if optional_skipped:
        print(f"\n\033[93mNote: {len(optional_skipped)} optional component(s) skipped:\033[0m")
        for name in optional_skipped:
            print(f"  - {name}")
        print("  These do not affect model correctness. See PyG docs for details.")

    return True


def print_platform_info(info: PlatformInfo):
    """Print platform information"""
    print("\n" + "=" * 60)
    print("Platform Information")
    print("=" * 60)
    print(f"  OS:           {info.os_name}")
    print(f"  Architecture: {info.arch}")
    print(f"  Python:       {info.python_version}")
    print(f"  CUDA:         {info.cuda_version or 'Not detected'}")
    print(f"  Conda:        {'Yes' if info.is_conda else 'No'}")


_OPTIONAL_PKGS = {"pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster"}


def print_install_status(status: Dict[str, Optional[str]]):
    """Print installation status"""
    print("\n" + "=" * 60)
    print("Current Installation Status")
    print("=" * 60)

    for pkg, version in status.items():
        if version:
            status_str = f"\033[92m{version}\033[0m"
        elif pkg in _OPTIONAL_PKGS:
            status_str = "\033[93mNot installed (optional)\033[0m"
        else:
            status_str = "\033[91mNot installed\033[0m"
        print(f"  {pkg:20s}: {status_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Install PyTorch and PyTorch Geometric with correct CUDA support"
    )
    parser.add_argument(
        "--cuda",
        type=str,
        help="CUDA version to target (e.g., 12.4, 12.8, 12.9)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install CPU-only version",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current installation only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--no-upgrade",
        action="store_true",
        help="Don't upgrade existing packages",
    )
    args = parser.parse_args()

    # Detect platform
    platform_info = detect_platform()
    print_platform_info(platform_info)

    # Check current installation
    status = check_installation()
    print_install_status(status)

    if args.check:
        # Just checking, exit
        all_installed = all(v is not None for v in status.values())
        return 0 if all_installed else 1

    # Determine CUDA version to use
    if args.cpu:
        cuda_version = None
    elif args.cuda:
        cuda_version = args.cuda
    else:
        cuda_version = platform_info.cuda_version

    # Get install configuration
    config = get_install_config(cuda_version, force_cpu=args.cpu)

    print("\n" + "=" * 60)
    print("Installation Plan")
    print("=" * 60)
    print(f"  PyTorch:      {config.torch_version}+{config.cuda_suffix}")
    print(f"  PyG:          {config.pyg_version}")
    print(f"  Index URL:    {config.extra_index_url}")
    print(f"  PyG Wheels:   {config.pyg_wheel_url}")

    # Build installation steps
    steps = build_install_steps(config, upgrade=not args.no_upgrade)

    print("\n" + "=" * 60)
    print("Installation Plan")
    print("=" * 60)
    for step in steps:
        tag = "[REQUIRED]" if step.critical else "[OPTIONAL]"
        print(f"  {tag:11s} {step.label}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("Commands (dry run)")
        print("=" * 60)
        for step in steps:
            print(f"\n  # {step.label}")
            print(f"  {' '.join(step.cmd)}")
        return 0

    # Confirm
    print("\nThis will install/upgrade the packages listed above.")
    print("Optional components will be skipped gracefully if unavailable.")
    response = input("Continue? [y/N]: ")
    if response.lower() not in ("y", "yes"):
        print("Aborted.")
        return 1

    # Run installation with smart fallback
    success = run_install_steps(steps)

    if success:
        print("\n" + "=" * 60)
        print("\033[92mInstallation completed successfully!\033[0m")
        print("=" * 60)

        # Verify
        print("\nVerifying installation...")
        new_status = check_installation()
        print_install_status(new_status)

        return 0
    else:
        print("\n" + "=" * 60)
        print("\033[91mInstallation failed!\033[0m")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
