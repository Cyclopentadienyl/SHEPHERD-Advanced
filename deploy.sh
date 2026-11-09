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
# uv.lock for cross-platform reproducibility, and so is torch_geometric, PyG's
# pure-Python core, at the verified version (BACKLOG 50). PyG's native
# extensions (pyg-lib, torch-scatter/sparse/cluster) remain out of the lock and
# come per platform via 'uv pip install', with graceful fallback on missing
# wheels.
#
# Usage:
#   ./deploy.sh
#
# Environment Variables:
#   PYTHON_EXE   - Python executable used to bootstrap uv venv (default: python3)
#   PYG_ARM_STRATEGY - ARM + CUDA only: pull | compile | skip | abort, instead of
#                  the menu (with no TTY and no setting: skip)
#   PYG_ARM_ON_PULL_FAILURE - ARM + CUDA, pull only: what a failed pull does.
#                  stop (default): the deployment stops and names what failed;
#                  build: it builds from source instead (scripts/build_pyg_arm.sh),
#                  non-interactively too. Decided once, before anything is fetched.
#                  A pull whose wheels passed their check but whose source receipt
#                  could not be written stops either way (17): a build would keep
#                  its receipt in the same place.
#   SCATTER_SPEC, SPARSE_SPEC, CLUSTER_SPEC, PYGLIB_SPEC, PYG_WHEEL_OUT - passed to
#                  the source build (see scripts/build_pyg_arm.sh)
#
# Exit status: 0 done; 1 Python or uv missing; 2 uv sync failed; 3 torch_geometric
#   does not import; 4 an invalid PYG_ARM_ON_PULL_FAILURE; 10 aborted at the PyG
#   menu. A stopped prebuilt pull (ARM + CUDA): 11 the release is not usable on
#   this host (its torch or CUDA, or its Python, is not the release's); 12 the
#   download failed; 13 the checksum failed or could not be taken; 14 the install
#   failed; 15 the check failed (the installed extensions are not shown to be the
#   release's wheels).
#   After an install of PyG native extensions on ARM + CUDA (the pull, or a source
#   build under compile or PYG_ARM_ON_PULL_FAILURE=build): 16 what is installed is
#   NOT verified (an install, an import or the check failed after something was
#   installed, and no later step replaced it); 17 they are installed and checked,
#   but the source receipt could not be written.
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
# The PyG wheel index is not configured here: Stage 3 derives it from the
# installed torch, which uv.lock pins (BACKLOG 50).

# What a failed prebuilt pull does (ARM + CUDA): stop unless a source build was
# explicitly allowed (BACKLOG 70; the owner's O5, 2026-11-10). Decided here, once,
# so nothing waits for a person after the pull has started.
PYG_ON_PULL_FAILURE="${PYG_ARM_ON_PULL_FAILURE:-}"
case "$PYG_ON_PULL_FAILURE" in
    ""|stop|build) ;;
    *) echo -e "${RED}[ERROR] PYG_ARM_ON_PULL_FAILURE='$PYG_ON_PULL_FAILURE' is not 'stop' or 'build'. Nothing was changed.${NC}"
       exit 4 ;;
esac

# Prebuilt PyG ARM wheels (Phase D). data.pyg.org publishes NO aarch64 wheels,
# so for ARM+CUDA hosts we host self-compiled wheels on a GitHub Release built
# for exactly this torch+CUDA combo. deploy compares the host's actual torch
# against these to decide whether the prebuilt wheels are usable (Case A) or a
# source build is required (Case B). When the prebuilt target changes, update
# all of these together with the Release, each asset's SHA-256 included (see
# medical-kg-todo.md Phase 3 note).
PYG_PREBUILT_TORCH="2.10.0"
PYG_PREBUILT_CUDA="130"
PYG_RELEASE_REPO="cyclopentadienyl/shepherd-advanced"
PYG_RELEASE_TAG="pyg-arm-cu130-torch2.10.0"
# SHA-256 of each Release asset, by CPython tag. The tarball holds native
# wheels that are installed as downloaded, so an asset with no entry here is
# not downloaded at all, and one that does not match is not installed. The
# Release holds cp312 only; the value is GitHub's asset digest, confirmed
# against a fresh download.
PYG_PREBUILT_SHA256_cp312="7cba46d26ba819aa6439cd319654b3a10e2a611e8637b2bec3684a51e97b8967"

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

# torch_geometric, PyG's pure-Python core, is in uv.lock at the verified version
# (pyproject.toml, BACKLOG 50), so the sync above installed it; only its native
# extensions are installed here.
if ! TG_VER="$("$PY" -c 'import torch_geometric;print(torch_geometric.__version__)' 2>/dev/null)"; then
    echo -e "${RED}[ERROR] torch_geometric (from uv.lock) does not import; re-run 'uv sync --inexact'${NC}"
    exit 3
fi
echo -e "\n${GREEN}[OK] torch_geometric ${TG_VER} (from uv.lock)${NC}"

# ---------------------------------------------------------------------------
# Native extensions: pyg-lib, torch-scatter, torch-sparse, torch-cluster.
#   - non-ARM / no-CUDA: data.pyg.org has wheels (linux_x86_64 / win_amd64) ->
#     install via find-links with graceful skip (built-in fallback covers misses).
#   - ARM + CUDA (DGX): data.pyg.org has NO aarch64 wheel for ANY torch/CUDA, so
#     offer: prebuilt-from-Release (only if host torch matches what we prebuilt,
#     "Case A") / self-compile / skip-with-fallback / abort.
# Non-interactive override: PYG_ARM_STRATEGY=pull|compile|skip|abort; with pull,
# PYG_ARM_ON_PULL_FAILURE=stop (default) | build. A failed pull stops unless a
# build was allowed; Case B (host torch/CUDA is not the release's) is a failed
# pull too, so PYG_ARM_STRATEGY=pull there stops too unless a build was allowed.
# ---------------------------------------------------------------------------
NATIVE_PKGS="pyg-lib torch-scatter torch-sparse torch-cluster"

TVER="$("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
TBASE="${TVER%%+*}"
TCU="$("$PY" -c 'import torch;print((torch.version.cuda or "").replace(".",""))' 2>/dev/null)"
CUDA_AVAIL="$("$PY" -c 'import torch;print(1 if torch.cuda.is_available() else 0)' 2>/dev/null)"
# The wheel index for the installed torch (e.g. torch-2.10.0+cu130), so a torch
# upgrade in pyproject.toml carries the index with it.
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TVER}.html"

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

run_source_build() {  # self-compile via the standalone builder; returns its exit status
    if [ ! -f scripts/build_pyg_arm.sh ]; then
        echo -e "${RED}[ERROR] scripts/build_pyg_arm.sh not found.${NC}"; return 1
    fi
    echo -e "[INFO] Compiling from source via scripts/build_pyg_arm.sh ..."
    # ASSUME_YES: deploy menu already obtained consent. Builder targets ./.venv.
    # It installs exactly the wheels it built, checks them and writes the source
    # receipt; *_SPEC and PYG_WHEEL_OUT reach it through the environment.
    ASSUME_YES=1 INSTALL_AFTER_BUILD=1 bash scripts/build_pyg_arm.sh
}

# An install this run cannot state as checked and receipted stops the
# deployment: it never ends as if nothing, or something verified, were installed.
stop_pyg() {  # stop_pyg <exit status: 16 or 17> <what is installed, and what failed>
    echo -e "\n${RED}[STOP] PyG native extensions: $2.${NC}"
    echo -e "${RED}       The deployment stops here (exit $1).${NC}"
    if [ "$1" = 17 ]; then
        echo -e "${YELLOW}[HINT] Let .venv/share/shepherd-advanced be written (a directory, with free space),${NC}"
        echo -e "${YELLOW}       and re-run.${NC}"
    else
        echo -e "${YELLOW}[HINT] Re-run when the cause is fixed: the pull and the build each reinstall all four${NC}"
        echo -e "${YELLOW}       by file path. To run on the torch.scatter_reduce fallback instead, remove them:${NC}"
        echo -e "${YELLOW}         uv pip uninstall pyg-lib torch-scatter torch-sparse torch-cluster${NC}"
    fi
    exit "$1"
}

# What a source build left installed, by scripts/build_pyg_arm.sh's exit status:
# 0 its wheels, imported and checked; 1 or 2 nothing of its own (it stopped before
# its install), which the caller states; 5 its wheels, checked, with no receipt;
# anything else (3 the install, 4 an import or the check) wheels NOT verified.
stop_if_the_build_installed_unverified() {  # <builder exit status>
    case "$1" in
        0|1|2) return 0 ;;
        5) stop_pyg 17 "the source build's wheels are installed and checked, but the source receipt could not be written (see above)" ;;
        3) stop_pyg 16 "the install of the source build's wheels failed (see above): what is installed under these names, whole or in part, is NOT verified" ;;
        *) stop_pyg 16 "the source build's wheels are installed but NOT verified: an extension does not import, or what is installed is not exactly what the build made (see above; builder exit $1)" ;;
    esac
}

# The prebuilt pull: fetch, verify, install exactly the release's wheels, check
# them and write the source receipt. On failure it returns non-zero with
# PULL_FAILED set to what failed (target, fetch, checksum, install, check,
# receipt); a pull is "verified" only when the checksum matched, and "installed"
# only when the installed bytes are the release's wheels. PULL_INSTALLED is 1
# once the release's wheels went to the installer: from then on a failure leaves
# them installed, unverified, until something replaces them.
PULL_FAILED=""
PULL_INSTALLED=0
PULL_RECEIPT_WORDS=""
pull_prebuilt() {
    local pyver asset url tmp expected_var expected actual pkg found checked wheels=()
    if [ "$PYG_CASE" != "A" ]; then
        echo -e "${RED}[ERROR] Pull not usable: host torch ${TVER:-unknown} is not the release's torch ${PYG_PREBUILT_TORCH}+cu${PYG_PREBUILT_CUDA}.${NC}"
        PULL_FAILED=target; return 1
    fi
    pyver="$("$PY" -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
    asset="pyg-ext-torch${PYG_PREBUILT_TORCH}-cu${PYG_PREBUILT_CUDA}-${pyver}-linux_aarch64.tar.gz"
    expected_var="PYG_PREBUILT_SHA256_${pyver}"
    expected="${!expected_var:-}"
    if [ -z "$expected" ]; then
        echo -e "${RED}[ERROR] Pull not usable: the release has no asset pinned for ${pyver} ($asset); not downloading it.${NC}"
        PULL_FAILED=target; return 1
    fi
    if ! command -v sha256sum > /dev/null 2>&1; then
        echo -e "${RED}[ERROR] Checksum: sha256sum not found, so $asset cannot be verified; not downloading it.${NC}"
        PULL_FAILED=checksum; return 1
    fi
    url="https://github.com/${PYG_RELEASE_REPO}/releases/download/${PYG_RELEASE_TAG}/${asset}"
    tmp="$(mktemp -d)"
    echo -e "[INFO] Downloading prebuilt wheels: $url"
    if ! curl -fSL "$url" -o "$tmp/$asset"; then
        echo -e "${RED}[ERROR] Download failed: $asset could not be fetched (see curl's message above).${NC}"
        rm -rf "$tmp"; PULL_FAILED=fetch; return 1
    fi
    actual="$(sha256sum "$tmp/$asset" | cut -d' ' -f1)"
    if [ "$actual" != "$expected" ]; then
        echo -e "${RED}[ERROR] SHA-256 mismatch for $asset: expected $expected, got $actual. The download is NOT verified and is not installed.${NC}"
        rm -rf "$tmp"; PULL_FAILED=checksum; return 1
    fi
    echo -e "${GREEN}[OK] SHA-256 verified: $asset${NC}"
    if ! tar xzf "$tmp/$asset" -C "$tmp"; then
        echo -e "${RED}[ERROR] Install: the verified $asset could not be unpacked.${NC}"
        rm -rf "$tmp"; PULL_FAILED=install; return 1
    fi
    # Exactly one wheel of each native package, from the asset's top level.
    for pkg in $NATIVE_PKGS; do
        found=( "$tmp/$(echo "$pkg" | tr '-' '_')"-*.whl )
        if [ "${#found[@]}" -ne 1 ] || [ ! -f "${found[0]}" ]; then
            echo -e "${RED}[ERROR] Install: $asset does not hold exactly one $pkg wheel.${NC}"
            rm -rf "$tmp"; PULL_FAILED=install; return 1
        fi
        wheels+=("${found[0]}")
    done
    echo -e "[INFO] Installing exactly the release's wheels, by file path (--reinstall --no-deps):"
    for w in "${wheels[@]}"; do echo -e "       $(basename "$w")"; done
    PULL_INSTALLED=1
    if ! uv pip install --python "$PY" --reinstall --no-deps "${wheels[@]}"; then
        echo -e "${RED}[ERROR] Install: installing the release's wheels failed (see above).${NC}"
        rm -rf "$tmp"; PULL_FAILED=install; return 1
    fi
    echo -e "[INFO] Checking what is installed against the release's wheels:"
    # compiled_against in the notation the source build records (torch.__version__
    # and torch.version.cuda: 2.10.0+cu130, 13.0), so one target reads alike from
    # both routes. A cuXYZ tag's last digit is its CUDA minor version.
    "$PY" scripts/validate_pyg_ext.py --check-wheels "${wheels[@]}" \
        --write-receipt "prebuilt release" --release-tag "$PYG_RELEASE_TAG" \
        --release-asset "$asset" --release-asset-sha256 "$actual" \
        --compiled-against-torch "${PYG_PREBUILT_TORCH}+cu${PYG_PREBUILT_CUDA}" \
        --compiled-against-cuda "${PYG_PREBUILT_CUDA%?}.${PYG_PREBUILT_CUDA: -1}"
    checked=$?   # scripts/validate_pyg_ext.py --check-wheels' statuses
    rm -rf "$tmp"
    case "$checked" in
        0) PULL_RECEIPT_WORDS="source receipt written" ;;
        4) PULL_RECEIPT_WORDS="no source receipt (the interpreter is not a virtual environment)" ;;
        3) echo -e "${RED}[ERROR] Receipt: the release's wheels are installed and checked, but the source receipt could not be written (see above).${NC}"
           PULL_FAILED=receipt; return 1 ;;
        *) echo -e "${RED}[ERROR] Check: the installed PyG extensions are NOT shown to be the release's wheels (see above).${NC}"
           PULL_FAILED=check; return 1 ;;
    esac
    return 0
}

# Set when the user asked for native ext (pull/compile) but this run installed
# none, so Stage 4 can surface a remediation warning.
PYG_FALLBACK_REASON=""
# The route this run took, stated at the end of Stage 3 and in Stage 4.
PYG_ROUTE=""

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
        echo -e "${YELLOW}[INFO] No TTY; defaulting to 'skip' (fallback). Set PYG_ARM_STRATEGY=pull|compile|skip|abort to override${NC}"
        echo -e "${YELLOW}       (with pull, PYG_ARM_ON_PULL_FAILURE=stop|build says what a failed pull does).${NC}"
    elif [ "$PYG_CASE" = "A" ]; then
        echo ""
        echo -e "  ${GREEN}1)${NC} Download prebuilt wheels from GitHub Release   ${GREEN}[recommended]${NC}"
        echo -e "  ${CYAN}2)${NC} Compile from source now (slower)"
        echo -e "  ${CYAN}3)${NC} Skip — continue with built-in torch.scatter_reduce fallback"
        echo -e "  ${CYAN}4)${NC} Abort deployment"
        read -r -p "Select [1-4] (default 1): " c
        case "${c:-1}" in 1) PYG_STRAT=pull;; 2) PYG_STRAT=compile;; 3) PYG_STRAT=skip;; 4) PYG_STRAT=abort;; *) PYG_STRAT=pull;; esac
        # Asked now, once: nothing waits for a person after the pull has started.
        if [ "$PYG_STRAT" = "pull" ] && [ -z "$PYG_ON_PULL_FAILURE" ]; then
            echo ""
            echo -e "  If the download, its checksum, its install or its check fails:"
            echo -e "  ${GREEN}1)${NC} Stop the deployment and say what failed   ${GREEN}[default]${NC}"
            echo -e "  ${CYAN}2)${NC} Build from source instead (scripts/build_pyg_arm.sh; other versions may result)"
            read -r -p "Select [1-2] (default 1): " c
            case "${c:-1}" in 2) PYG_ON_PULL_FAILURE=build;; *) PYG_ON_PULL_FAILURE=stop;; esac
        fi
    else
        echo ""
        echo -e "  ${YELLOW}Host torch (${TVER}) != prebuilt (torch ${PYG_PREBUILT_TORCH}+cu${PYG_PREBUILT_CUDA}); prebuilt wheels are NOT usable.${NC}"
        echo -e "  ${GREEN}1)${NC} Compile from source now   ${GREEN}[recommended]${NC}"
        echo -e "  ${CYAN}2)${NC} Skip — continue with built-in torch.scatter_reduce fallback"
        echo -e "  ${CYAN}3)${NC} Abort deployment"
        read -r -p "Select [1-3] (default 1): " c
        case "${c:-1}" in 1) PYG_STRAT=compile;; 2) PYG_STRAT=skip;; 3) PYG_STRAT=abort;; *) PYG_STRAT=compile;; esac
    fi
    PYG_ON_PULL_FAILURE="${PYG_ON_PULL_FAILURE:-stop}"
    if [ "$PYG_STRAT" = "pull" ]; then
        echo -e "[INFO] If the pull fails: ${PYG_ON_PULL_FAILURE} (PYG_ARM_ON_PULL_FAILURE)"
    fi

    case "$PYG_STRAT" in
        pull)
            # Case B can never use the prebuilt wheels: that is a failed pull
            # too, and it builds only where a build was allowed.
            if pull_prebuilt; then
                PYG_ROUTE="prebuilt release ${PYG_RELEASE_TAG}: installed and checked against its wheels"
                echo -e "${GREEN}[OK] Prebuilt PyG wheels installed and checked; ${PULL_RECEIPT_WORDS}.${NC}"
            else
                # The release's wheels passed their check; only their receipt is
                # missing. A source build would keep its receipt in the same
                # place, so the build fallback does not apply.
                if [ "$PULL_FAILED" = receipt ]; then
                    stop_pyg 17 "the release's wheels are installed and checked, but the source receipt could not be written (see above)"
                fi
                case "$PULL_FAILED" in
                    target)   PULL_WORDS="the prebuilt release is not usable on this host"; PULL_EXIT=11 ;;
                    fetch)    PULL_WORDS="the download failed"; PULL_EXIT=12 ;;
                    checksum) PULL_WORDS="the checksum failed or could not be taken (the download is not verified)"; PULL_EXIT=13 ;;
                    install)  PULL_WORDS="the install failed"; PULL_EXIT=14 ;;
                    *)        PULL_WORDS="the check failed: the installed extensions are not shown to be the release's wheels"; PULL_EXIT=15 ;;
                esac
                if [ "$PYG_ON_PULL_FAILURE" != "build" ]; then
                    echo -e "\n${RED}[STOP] Prebuilt PyG pull failed: ${PULL_WORDS}.${NC}"
                    echo -e "${RED}       The deployment stops here (exit ${PULL_EXIT}); nothing was built in its place.${NC}"
                    echo -e "${YELLOW}[HINT] Re-run when the cause is fixed, or choose another route: PYG_ARM_STRATEGY=compile, or${NC}"
                    echo -e "${YELLOW}       PYG_ARM_STRATEGY=pull PYG_ARM_ON_PULL_FAILURE=build to build from source when a pull fails.${NC}"
                    exit "$PULL_EXIT"
                fi
                echo -e "${YELLOW}[WARN] Prebuilt PyG pull failed: ${PULL_WORDS}. Building from source instead,${NC}"
                echo -e "${YELLOW}       as PYG_ARM_ON_PULL_FAILURE=build allows (other versions than the release's may result).${NC}"
                run_source_build; BUILD_STATUS=$?
                stop_if_the_build_installed_unverified "$BUILD_STATUS"
                if [ "$BUILD_STATUS" -eq 0 ]; then
                    PYG_ROUTE="local build, after the prebuilt pull failed (${PULL_FAILED}): installed and checked against the wheels it built"
                elif [ "$PULL_INSTALLED" = 1 ]; then
                    # The build installed nothing, so the release's wheels the
                    # pull installed stay, as unverified as its failure left them.
                    stop_pyg 16 "the prebuilt pull installed the release's wheels but failed (${PULL_FAILED}), and the source build allowed in its place did not complete: what is installed is NOT verified"
                else
                    echo -e "${YELLOW}[WARN] The source build did not complete; it installed nothing (see its output above).${NC}"
                    PYG_ROUTE="none: the prebuilt pull failed (${PULL_FAILED}) and the source build did not complete"
                    PYG_FALLBACK_REASON="build-failed"
                fi
            fi ;;
        compile)
            run_source_build; BUILD_STATUS=$?
            stop_if_the_build_installed_unverified "$BUILD_STATUS"
            if [ "$BUILD_STATUS" -eq 0 ]; then
                PYG_ROUTE="local build: installed and checked against the wheels it built"
            else
                echo -e "${YELLOW}[WARN] The source build did not complete; it installed nothing (see its output above).${NC}"
                PYG_ROUTE="none: the source build did not complete"
                PYG_FALLBACK_REASON="build-failed"
            fi ;;
        skip)
            echo -e "${YELLOW}[SKIP] PyG native ext not installed; using torch.scatter_reduce fallback.${NC}"
            PYG_ROUTE="none: skipped"
            PYG_FALLBACK_REASON="user-skipped" ;;
        abort)
            echo -e "${RED}[ABORT] Deployment terminated by user at PyG native-extension stage.${NC}"; exit 10 ;;
        *)
            echo -e "${YELLOW}[WARN] Unknown strategy '$PYG_STRAT'; skipping (fallback).${NC}"
            PYG_ROUTE="none: unknown strategy '$PYG_STRAT', skipped" ;;
    esac
    echo -e "[INFO] PyG native route: ${PYG_ROUTE}"
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
uv export --format requirements-txt --no-hashes --no-emit-project -o "$CUVS_CONSTRAINTS" >/dev/null 2>&1
if uv pip install --extra-index-url https://pypi.nvidia.com \
       -c "$CUVS_CONSTRAINTS" "cuvs-cu13>=24.12" 2>/dev/null; then
    # Installed is not usable: constructing the backend also needs cupy, and this
    # script never builds or searches an index. Say only what was established.
    if "$PY" -c "import cuvs.neighbors, cupy" >/dev/null 2>&1; then
        echo -e "${GREEN}[OK] cuVS and cupy importable; functional GPU search not tested${NC}"
    else
        echo -e "${YELLOW}[WARN] cuVS installed but its prerequisites are not importable${NC}"
        echo -e "${YELLOW}       (cupy missing?) — the GPU backend cannot be constructed.${NC}"
        echo -e "${YELLOW}       Voyager (CPU) remains the primary backend.${NC}"
    fi
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

# Advise whenever we end up on the torch.scatter_reduce fallback -- whether the
# source build failed OR the user deliberately skipped. The app still runs
# correctly (torch_geometric falls back), so this is a performance advisory with
# remediation. Only the opening line differs by reason; impact + how-to are shared.
if [ -n "$PYG_FALLBACK_REASON" ]; then
    echo -e "\n${YELLOW}----------------------------------------------------------------------------${NC}"
    echo -e "${YELLOW}[WARN] This deployment did not install the PyG native extensions; any that${NC}"
    echo -e "${YELLOW}       are missing run on the torch.scatter_reduce fallback.${NC}"
    if [ "$PYG_FALLBACK_REASON" = "build-failed" ]; then
        echo -e "${YELLOW}       You chose to build PyG from source but the build did not complete${NC}"
        echo -e "${YELLOW}       (commonly: missing system build tools — see the [STAGE 2/5] output above).${NC}"
        echo -e "${YELLOW}       It stopped before its install, so it installed nothing (an install, an${NC}"
        echo -e "${YELLOW}       import or a check that fails after an install stops the deployment).${NC}"
    else
        echo -e "${YELLOW}       You chose to skip the PyG native extensions.${NC}"
    fi
    echo -e "${YELLOW}       What imports now: $("$PY" -c 'from src.utils.pyg_native_check import format_status_line; print(format_status_line())' 2>/dev/null || echo 'not known')${NC}"
    echo -e "${YELLOW}       Impact: training and inference still RUN correctly, but some GNN ops${NC}"
    echo -e "${YELLOW}       (notably HGTConv message-passing and neighbor sampling) run slower${NC}"
    echo -e "${YELLOW}       than with the native CUDA kernels.${NC}"
    echo -e "${YELLOW}       To enable them later, install the toolchain (if needed) and re-run,${NC}"
    echo -e "${YELLOW}       choosing prebuilt download or compile:${NC}"
    echo -e "${YELLOW}         sudo apt install build-essential cmake ninja-build python3.12-dev${NC}"
    echo -e "${YELLOW}         ./deploy.sh${NC}"
    echo -e "${YELLOW}       Then confirm with: ${PY} scripts/validate_pyg_ext.py${NC}"
    echo -e "${YELLOW}----------------------------------------------------------------------------${NC}"
fi

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}   Deployment Complete!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "[INFO] Virtual environment: .venv"
echo -e "[INFO] Python interpreter:  $PY"
echo -e "[INFO] Lock file:           uv.lock (commit'd to repo)"
if [ -n "$PYG_ROUTE" ]; then
    echo -e "[INFO] PyG native route:    ${PYG_ROUTE}"
fi
echo -e "\nNext steps:"
echo -e "   1. Verify the installation:"
echo -e "      ${YELLOW}$PY -c \"import torch; print(torch.cuda.is_available())\"${NC}"
echo -e ""
echo -e "   2. Quick demo (small built-in graph, ~1 min). Training takes the GPU lock and"
echo -e "      is refused without one (deployment guide §4.11): in this shell, name the"
echo -e "      workspaces root the service will be given, so both take the same lock:"
echo -e "      ${YELLOW}export SHEPHERD_WORKSPACES_ROOT=\"$(pwd)/data/workspaces\"${NC}"
echo -e "      ${YELLOW}$PY scripts/setup_demo.py --train-model${NC}"
echo -e ""
echo -e "   3. Real data pipeline (manual HPO download required first;"
echo -e "      see data/external/README.md for the required annotation files)."
echo -e "      --generate-samples is not optional: without it the workspace has"
echo -e "      no split manifest, and training and serving both refuse it. Train in a"
echo -e "      shell where SHEPHERD_WORKSPACES_ROOT is set as in step 2: a base first,"
echo -e "      then its task models on the dashboard's Research › Prepare › Fine-tune"
echo -e "      (or train_model.py --stage finetune --task <task> --finetune-from <base>)."
echo -e "      ${YELLOW}$PY scripts/build_knowledge_graph.py --workspace data/workspaces/main \\${NC}"
echo -e "      ${YELLOW}    --external-dir data/external --generate-samples \\${NC}"
echo -e "      ${YELLOW}    --num-train 200000 --num-val 40000${NC}"
echo -e "      ${YELLOW}$PY scripts/train_model.py --data-dir data/workspaces/main --stage base${NC}"
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
echo -e "      correctly on your GPU (not just that they import); it also shows the"
echo -e "      source receipt against what is installed now:"
echo -e "        ${YELLOW}$PY scripts/validate_pyg_ext.py${NC}"
echo -e ""
echo -e "[TIP] For development (pytest, linting, etc.):"
echo -e "      ${YELLOW}uv sync --inexact --extra dev${NC}"
echo -e "      (--inexact stops uv removing the PyG native ext and cuVS installed"
echo -e "       above simply for being outside uv.lock.)"
echo -e ""
echo -e "[TIP] To regenerate uv.lock after editing pyproject.toml:"
echo -e "      ${YELLOW}uv lock${NC}"
echo -e ""
echo -e "[TIP] If using DGX Spark, ensure 'nvcc --version' matches torch CUDA (13.0)."
echo ""
