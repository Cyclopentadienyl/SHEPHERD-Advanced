#!/bin/bash
set -u

# ============================================================================
# SHEPHERD-Advanced | Linux Deployment Script (Unified x86/ARM)
# ============================================================================
#
# Supported Platforms:
#   1. Linux x86_64 (Standard Servers/Workstations)
#   2. Linux aarch64 (NVIDIA DGX Spark / Grace Hopper / Jetson Orin)
#
# Phase C onwards: this script uses uv (https://docs.astral.sh/uv/) as the
# primary deployment tool. PyTorch (torch/torchvision/torchaudio) is now
# managed by uv via [tool.uv.sources] in pyproject.toml and recorded in
# uv.lock for cross-platform reproducibility. PyG native extensions
# (pyg-lib, torch-scatter/sparse/cluster) remain out of the lock and are
# installed via 'uv pip install' with graceful fallback on missing wheels.
#
# Usage:
#   ./deploy.sh
#
# Environment Variables:
#   PYTHON_EXE   - Python executable used to bootstrap uv venv (default: python3)
#
# Note on Optional Accelerators (xFormers, FlashAttention, SageAttention):
#   These are NOT installed during deployment. They are auto-installed
#   at launch time via command-line arguments. See launch_shepherd.sh.
#
# ============================================================================

# === Colors for Output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "\n${CYAN}============================================================================${NC}"
echo -e "${CYAN}   SHEPHERD-Advanced Deployment Script (Linux)${NC}"
echo -e "${CYAN}============================================================================${NC}\n"

# === Configuration ===
PYTHON_EXE="${PYTHON_EXE:-python3}"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.10.0+cu130.html"

# Prebuilt PyG ARM wheels (Phase D). data.pyg.org publishes NO aarch64 wheels,
# so for ARM+CUDA hosts we host self-compiled wheels on a GitHub Release built
# for exactly this torch+CUDA combo. deploy compares the host's actual torch
# against these to decide whether the prebuilt wheels are usable (Case A) or a
# source build is required (Case B). When the prebuilt target changes, update
# all three together with the Release (see medical-kg-todo.md Phase 3 note).
PYG_PREBUILT_TORCH="2.10.0"
PYG_PREBUILT_CUDA="130"
PYG_RELEASE_REPO="cyclopentadienyl/shepherd-advanced"
PYG_RELEASE_TAG="pyg-arm-cu130-torch2.10.0"

# Detect Architecture
ARCH=$(uname -m)
echo -e "${CYAN}[INFO] Detected Architecture: ${ARCH}${NC}"
if [ "$ARCH" == "aarch64" ]; then
    echo -e "${YELLOW}[INFO] Running on ARM64 (Likely DGX Spark / Grace Hopper)${NC}"
fi

# ============================================================================
# STAGE 1/4: HARDWARE DETECTION & uv SETUP
# ============================================================================
echo -e "\n${CYAN}[STAGE 1/4] Hardware Detection & uv Setup${NC}"
echo "----------------------------------------------------------------------------"

# Check Python (used only to bootstrap uv venv selection; uv manages versions itself)
echo -e "[INFO] Bootstrap Python: $PYTHON_EXE"
if ! $PYTHON_EXE --version > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] Python not found. Please install Python 3.12+${NC}"
    exit 1
fi

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[INFO] NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader | while read gpu; do
        echo -e "[INFO] GPU: $gpu"
    done
else
    echo -e "${YELLOW}[WARN] nvidia-smi not found. GPU may not be available.${NC}"
fi

# uv installation check (with y/N confirmation if missing)
if ! command -v uv >/dev/null 2>&1; then
    echo ""
    echo -e "${YELLOW}[INFO] uv is not installed. uv is required for SHEPHERD-Advanced deployment.${NC}"
    echo -e "[INFO] Official installer: https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    read -r -p "Install uv now via official installer? (y/N): " ans
    case "$ans" in
        [yY]|[yY][eE][sS])
            echo -e "[INFO] Installing uv..."
            if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
                echo -e "${RED}[ERROR] uv installer failed (network error?). Install manually and re-run.${NC}"
                exit 1
            fi
            # uv installs to ~/.local/bin by default; ensure it's on PATH for this session
            export PATH="$HOME/.local/bin:$PATH"
            if ! command -v uv >/dev/null 2>&1; then
                echo -e "${RED}[ERROR] uv installed but not found on PATH.${NC}"
                echo -e "${YELLOW}[HINT] Add ~/.local/bin to your PATH and re-run this script.${NC}"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}[ERROR] Deployment aborted: uv is required but not installed.${NC}"
            exit 1
            ;;
    esac
fi
echo -e "${GREEN}[OK] uv: $(uv --version)${NC}"

# ============================================================================
# STAGE 2/4: ENVIRONMENT + DEPENDENCY SYNC (uv.lock driven)
# ============================================================================
echo -e "\n${CYAN}[STAGE 2/4] Environment + Dependency Sync${NC}"
echo "----------------------------------------------------------------------------"

# uv sync creates .venv if missing and installs everything from uv.lock.
# torch/torchvision/torchaudio are routed to cu130 index automatically via
# [tool.uv.sources] in pyproject.toml; resolved per platform.
#
# --inexact: do NOT remove packages that aren't in the lock. This protects
# PyG native extensions (installed in Stage 3) from being purged on repeat
# deploys.
echo -e "[INFO] Running 'uv sync --inexact' (creates .venv, installs lock contents)..."
if ! uv sync --inexact; then
    echo -e "${RED}[ERROR] uv sync failed. Check network and that uv.lock matches pyproject.toml.${NC}"
    echo -e "${YELLOW}[HINT] If uv.lock is stale, regenerate with 'uv lock'.${NC}"
    exit 2
fi
echo -e "${GREEN}[OK] uv.lock contents synced into .venv${NC}"

# Pointer for downstream Python invocations
PY=".venv/bin/python"

# NOTE on sm_121 warning (Blackwell GB10 / DGX Spark):
#   torch 2.10.0+cu130 still emits: "Found GPU0 NVIDIA GB10 which is of cuda
#   capability 12.1. Minimum and Maximum cuda capability supported by this
#   version of PyTorch is (8.0) - (12.0)". This is HARMLESS: sm_121 is SASS/PTX
#   binary-compatible with sm_120, so kernels run fine (the CUDA smoke test
#   below passes). It is just a conservative capability-range check that has not
#   been widened to include sm_121 yet; not fixed as of 2.10.
#   Ref: https://discuss.pytorch.org/t/nvidia-dgx-spark-support/223677

# Validate (CUDA smoke test when available; graceful CPU-only fallback)
"$PY" -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA build: {torch.version.cuda} | available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    x = torch.zeros(2, 2, device='cuda')
    print(f'[OK] CUDA smoke test passed | device: {x.device} | GPU: {torch.cuda.get_device_name(0)}')
else:
    print('[WARN] CUDA not detected at deploy time; launch-time detection will retry')
" || echo -e "${YELLOW}[WARN] PyTorch validation returned non-zero${NC}"

# ============================================================================
# STAGE 3/4: PyG NATIVE EXTENSIONS (out-of-lock, graceful fallback)
# ============================================================================
echo -e "\n${CYAN}[STAGE 3/4] PyG Native Extensions${NC}"
echo "----------------------------------------------------------------------------"
echo -e "[INFO] PyG native ext are installed outside uv.lock because their wheel"
echo -e "[INFO] index is torch-version-coupled HTML and three of them lack Windows"
echo -e "[INFO] wheels. Missing wheels fall back to torch.scatter_reduce (built-in)."

# torch_geometric: pure Python high-level API; install via uv pip (post-sync)
echo -e "\n[INFO] Installing torch_geometric (high-level PyG API)..."
if ! uv pip install torch_geometric; then
    echo -e "${RED}[ERROR] Failed to install torch_geometric${NC}"
    exit 3
fi

# ---------------------------------------------------------------------------
# Native extensions: pyg-lib, torch-scatter, torch-sparse, torch-cluster.
#   - non-ARM / no-CUDA: data.pyg.org has wheels (linux_x86_64 / win_amd64) ->
#     install via find-links with graceful skip (built-in fallback covers misses).
#   - ARM + CUDA (DGX): data.pyg.org has NO aarch64 wheel for ANY torch/CUDA, so
#     offer: prebuilt-from-Release (only if host torch matches what we prebuilt,
#     "Case A") / self-compile / skip-with-fallback / abort.
# Non-interactive override: PYG_ARM_STRATEGY=pull|compile|skip|abort
# ---------------------------------------------------------------------------
NATIVE_PKGS="pyg-lib torch-scatter torch-sparse torch-cluster"

TVER="$("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
TBASE="${TVER%%+*}"
TCU="$("$PY" -c 'import torch;print((torch.version.cuda or "").replace(".",""))' 2>/dev/null)"
CUDA_AVAIL="$("$PY" -c 'import torch;print(1 if torch.cuda.is_available() else 0)' 2>/dev/null)"

install_pyg_from_index() {  # original non-ARM path: data.pyg.org find-links
    echo -e "\n[INFO] Installing pyg-lib (native GNN kernels)..."
    if uv pip install pyg-lib --find-links "$PYG_WHEEL_URL"; then
        echo -e "${GREEN}[OK] pyg-lib installed${NC}"
    else
        echo -e "${YELLOW}[SKIP] pyg-lib: no pre-built wheel; using torch.scatter_reduce fallback${NC}"
    fi
    echo -e "\n[INFO] Installing torch-scatter/sparse/cluster (third-party ext, optional)..."
    if uv pip install torch-scatter torch-sparse torch-cluster \
        --find-links "$PYG_WHEEL_URL" --only-binary :all:; then
        echo -e "${GREEN}[OK] PyG third-party extensions installed${NC}"
    else
        echo -e "${YELLOW}[SKIP] No pre-built wheels for this platform/torch combination${NC}"
        echo -e "${YELLOW}       Impact: NONE — HeteroConv + GAT/SAGE auto-use torch.scatter_reduce${NC}"
        echo -e "${YELLOW}       (PyTorch 2.0+ native, equivalent perf for our architecture)${NC}"
    fi
}

run_source_build() {  # self-compile via the standalone builder
    if [ ! -f scripts/build_pyg_arm.sh ]; then
        echo -e "${RED}[ERROR] scripts/build_pyg_arm.sh not found.${NC}"; return 1
    fi
    echo -e "[INFO] Compiling from source via scripts/build_pyg_arm.sh ..."
    # ASSUME_YES: deploy menu already obtained consent. Builder targets ./.venv.
    ASSUME_YES=1 INSTALL_AFTER_BUILD=1 bash scripts/build_pyg_arm.sh
}

fetch_prebuilt_wheels() {  # downloads + extracts Release tarball; sets WHEELDIR
    local pyver asset url tmp
    pyver="$("$PY" -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
    asset="pyg-ext-torch${PYG_PREBUILT_TORCH}-cu${PYG_PREBUILT_CUDA}-${pyver}-linux_aarch64.tar.gz"
    url="https://github.com/${PYG_RELEASE_REPO}/releases/download/${PYG_RELEASE_TAG}/${asset}"
    tmp="$(mktemp -d)"
    echo -e "[INFO] Downloading prebuilt wheels: $url"
    if curl -fSL "$url" -o "$tmp/$asset" && tar xzf "$tmp/$asset" -C "$tmp"; then
        WHEELDIR="$tmp"; return 0
    fi
    return 1
}

if [ "$ARCH" = "aarch64" ] && [ "$CUDA_AVAIL" = "1" ]; then
    echo -e "[INFO] ARM + CUDA detected; data.pyg.org has no aarch64 wheels."
    echo -e "[INFO] Host torch: ${TVER:-unknown}  |  prebuilt target: torch ${PYG_PREBUILT_TORCH}+cu${PYG_PREBUILT_CUDA}"
    if [ "$TBASE" = "$PYG_PREBUILT_TORCH" ] && [ "$TCU" = "$PYG_PREBUILT_CUDA" ]; then
        PYG_CASE=A; echo -e "${GREEN}[INFO] Host matches prebuilt target -> prebuilt wheels are usable.${NC}"
    else
        PYG_CASE=B; echo -e "${YELLOW}[INFO] Host differs from prebuilt target -> prebuilt NOT usable; source build only.${NC}"
    fi

    PYG_STRAT=""
    if [ -n "${PYG_ARM_STRATEGY:-}" ]; then
        PYG_STRAT="$PYG_ARM_STRATEGY"
        echo -e "[INFO] PYG_ARM_STRATEGY=$PYG_STRAT (non-interactive override)"
    elif [ ! -t 0 ]; then
        PYG_STRAT="skip"
        echo -e "${YELLOW}[INFO] No TTY; defaulting to 'skip' (fallback). Set PYG_ARM_STRATEGY=pull|compile|skip|abort to override.${NC}"
    elif [ "$PYG_CASE" = "A" ]; then
        echo ""
        echo -e "  ${GREEN}1)${NC} Download prebuilt wheels from GitHub Release   ${GREEN}[recommended]${NC}"
        echo -e "  ${CYAN}2)${NC} Compile from source now (slower)"
        echo -e "  ${CYAN}3)${NC} Skip — continue with built-in torch.scatter_reduce fallback"
        echo -e "  ${CYAN}4)${NC} Abort deployment"
        read -r -p "Select [1-4] (default 1): " c
        case "${c:-1}" in 1) PYG_STRAT=pull;; 2) PYG_STRAT=compile;; 3) PYG_STRAT=skip;; 4) PYG_STRAT=abort;; *) PYG_STRAT=pull;; esac
    else
        echo ""
        echo -e "  ${YELLOW}Host torch (${TVER}) != prebuilt (torch ${PYG_PREBUILT_TORCH}+cu${PYG_PREBUILT_CUDA}); prebuilt wheels are NOT usable.${NC}"
        echo -e "  ${GREEN}1)${NC} Compile from source now   ${GREEN}[recommended]${NC}"
        echo -e "  ${CYAN}2)${NC} Skip — continue with built-in torch.scatter_reduce fallback"
        echo -e "  ${CYAN}3)${NC} Abort deployment"
        read -r -p "Select [1-3] (default 1): " c
        case "${c:-1}" in 1) PYG_STRAT=compile;; 2) PYG_STRAT=skip;; 3) PYG_STRAT=abort;; *) PYG_STRAT=compile;; esac
    fi

    # Case B can never use prebuilt wheels.
    if [ "$PYG_CASE" = "B" ] && [ "$PYG_STRAT" = "pull" ]; then
        echo -e "${YELLOW}[WARN] prebuilt wheels don't match host torch; switching to source build.${NC}"
        PYG_STRAT=compile
    fi

    case "$PYG_STRAT" in
        pull)
            if fetch_prebuilt_wheels && uv pip install --no-deps --find-links "$WHEELDIR" $NATIVE_PKGS; then
                echo -e "${GREEN}[OK] Prebuilt PyG wheels installed.${NC}"
            else
                echo -e "${YELLOW}[WARN] Prebuilt download/install failed (Release asset missing?). Falling back to source build...${NC}"
                run_source_build || echo -e "${YELLOW}[WARN] source build failed; using torch.scatter_reduce fallback${NC}"
            fi ;;
        compile)
            run_source_build || echo -e "${YELLOW}[WARN] source build failed; using torch.scatter_reduce fallback${NC}" ;;
        skip)
            echo -e "${YELLOW}[SKIP] PyG native ext not installed; using torch.scatter_reduce fallback (Impact: none for our architecture).${NC}" ;;
        abort)
            echo -e "${RED}[ABORT] Deployment terminated by user at PyG native-extension stage.${NC}"; exit 10 ;;
        *)
            echo -e "${YELLOW}[WARN] Unknown strategy '$PYG_STRAT'; skipping (fallback).${NC}" ;;
    esac
else
    install_pyg_from_index
fi

# cuVS: Linux GPU-accelerated vector backend, optional (Voyager is the CPU fallback).
# Use the cu13 build (cuvs-cu12 would pull a CUDA-12 binding line). cuVS's RAPIDS
# chain depends on a set of shared deps that torch also uses (cuda-bindings plus the
# nvidia-* CUDA libs), and torch 2.10.0+cu130 pins them to specific versions. An
# unconstrained 'uv pip install' resolves only cuVS's subtree (torch's pins are not
# in view), so it upgrades those shared deps past torch's pins, which the next
# 'uv sync' then reverts -> version flip-flop on every deploy. Constrain the install
# to the lock's resolved versions: cuVS may add its own packages but can never bump
# a torch-pinned shared dep. cuvs-cu13 26.4.0 resolves cleanly under these pins
# (verified); if a future cuVS needs newer, the constrained install fails -> Voyager.
echo -e "\n[INFO] Attempting cuVS (NVIDIA RAPIDS GPU vector backend)..."
CUVS_CONSTRAINTS="$(mktemp)"
uv export --format requirements-txt --no-hashes --no-emit-project -o "$CUVS_CONSTRAINTS" 2>/dev/null
if uv pip install --extra-index-url https://pypi.nvidia.com \
       -c "$CUVS_CONSTRAINTS" "cuvs-cu13>=24.12" 2>/dev/null; then
    echo -e "${GREEN}[OK] cuVS installed (GPU vector backend available)${NC}"
else
    echo -e "${YELLOW}[INFO] cuVS not available; Voyager (CPU) remains the primary backend${NC}"
    echo -e "${YELLOW}[HINT] For GPU acceleration, manually try:${NC}"
    echo -e "${YELLOW}       uv pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13${NC}"
fi
rm -f "$CUVS_CONSTRAINTS"

# ============================================================================
# STAGE 4/4: VALIDATION & FINALIZATION
# ============================================================================
echo -e "\n${CYAN}[STAGE 4/4] Validation & Finalization${NC}"
echo "----------------------------------------------------------------------------"

if [ -f "scripts/validate_installation.py" ]; then
    echo -e "[INFO] Running installation validation..."
    "$PY" scripts/validate_installation.py || echo -e "${YELLOW}[WARN] Validation returned warnings${NC}"
fi

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}   Deployment Complete!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "[INFO] Virtual environment: .venv"
echo -e "[INFO] Python interpreter:  $PY"
echo -e "[INFO] Lock file:           uv.lock (commit'd to repo)"
echo -e "\nNext steps:"
echo -e "   1. Verify the installation:"
echo -e "      ${YELLOW}$PY -c \"import torch; print(torch.cuda.is_available())\"${NC}"
echo -e ""
echo -e "   2. Quick demo (synthetic data, ~1 min):"
echo -e "      ${YELLOW}$PY scripts/setup_demo.py --train-model${NC}"
echo -e ""
echo -e "   3. Real data pipeline (manual HPO download required first;"
echo -e "      see data/external/README.md for the required annotation files):"
echo -e "      ${YELLOW}$PY scripts/build_knowledge_graph.py${NC}"
echo -e "      ${YELLOW}$PY scripts/compute_shortest_paths.py${NC}"
echo -e "      ${YELLOW}$PY scripts/train_model.py${NC}"
echo -e "      ${YELLOW}$PY scripts/build_index.py${NC}"
echo -e ""
echo -e "   4. Launch the system:"
echo -e "      ${YELLOW}./launch_shepherd.sh${NC}"
echo -e ""
echo -e "   5. Run tests (optional):"
echo -e "      ${YELLOW}$PY -m pytest tests/unit/${NC}"
echo -e ""
echo -e "[TIP] Optional accelerators (FlashAttention, xFormers, SageAttention):"
echo -e "      These are auto-installed at launch time via arguments:"
echo -e "        ${YELLOW}./launch_shepherd.sh --flash-attn${NC}    (FlashAttention-2)"
echo -e "        ${YELLOW}./launch_shepherd.sh --xformers${NC}      (xFormers)"
echo -e "        ${YELLOW}./launch_shepherd.sh --sage-attn${NC}     (SageAttention)"
echo -e "      Compatibility not guaranteed; fallback to PyTorch SDPA is always available."
echo -e ""
echo -e "[TIP] If you compiled PyG natively (ARM host, or a GPU newer than the"
echo -e "      prebuilt wheels support), confirm the compiled kernels actually run"
echo -e "      correctly on your GPU (not just that they import):"
echo -e "        ${YELLOW}$PY scripts/validate_pyg_ext.py${NC}"
echo -e ""
echo -e "[TIP] For development (pytest, linting, etc.):"
echo -e "      ${YELLOW}uv sync --extra dev${NC}"
echo -e ""
echo -e "[TIP] To regenerate uv.lock after editing pyproject.toml:"
echo -e "      ${YELLOW}uv lock${NC}"
echo -e ""
echo -e "[TIP] If using DGX Spark, ensure 'nvcc --version' matches torch CUDA (13.0)."
echo ""
