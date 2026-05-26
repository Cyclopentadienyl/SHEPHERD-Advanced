#!/bin/bash
set -uo pipefail

# ============================================================================
# SHEPHERD-Advanced | PyG native-extension builder for Linux aarch64 (ARM+CUDA)
# ============================================================================
#
# Why this exists (Phase D):
#   PyG's native extensions (pyg-lib, torch-scatter, torch-sparse, torch-cluster)
#   publish prebuilt wheels at https://data.pyg.org ONLY for linux_x86_64 and
#   win_amd64 -- there is NO linux_aarch64 wheel for ANY torch/CUDA combo. So on
#   ARM+CUDA hosts (NVIDIA DGX Spark/GB10, Grace Hopper, Jetson Orin, ...) they
#   must be compiled from source against the locally-installed torch.
#
# Two ways this script is used:
#   1. Maintainer: run it on a DGX whose deploy .venv has the project's locked
#      torch (currently 2.10.0+cu130) to produce the wheels that get uploaded to
#      a GitHub Release. deploy.sh's "pull prebuilt" path then downloads those.
#   2. End-user fallback: deploy.sh calls it when the host's torch/CUDA does NOT
#      match the prebuilt Release wheels (e.g. after a future torch bump), to
#      compile against whatever torch the host actually has.
#
# Design: VERSION-AGNOSTIC. It builds against the torch already present in the
#   target venv (default: the project's ./.venv produced by `uv sync`), so the
#   compiled C++/CUDA ABI is guaranteed to match what will run it. GPU compute
#   capability and CUDA toolkit path are auto-detected. Nothing here is pinned
#   to 2.10/cu130 -- bumping torch needs no edit to this script, only a rebuild.
#
# Run on the ARM target (NOT in CI / x86):
#   bash scripts/build_pyg_arm.sh
#
# Overridable via environment:
#   TARGET_PYTHON         - python whose torch to build against (default ./.venv)
#   PYG_WHEEL_OUT         - where built wheels land  (default $HOME/pyg-wheels-arm)
#   CUDA_HOME             - CUDA toolkit root         (default: auto-detect)
#   TORCH_CUDA_ARCH_LIST  - GPU targets, e.g. "12.0;12.1" (default: auto-detect)
#   MAX_JOBS              - parallel build jobs       (default: nproc-4)
#   ASSUME_YES=1          - skip the (y/N) dep-install prompt (for deploy.sh)
#   INSTALL_AFTER_BUILD=0 - build wheels only, don't install into target venv
#   {SCATTER,SPARSE,CLUSTER,PYGLIB}_SPEC - pip spec override per package
# ============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
say()  { echo -e "$@"; }
die()  { echo -e "${RED}[ERROR] $*${NC}" >&2; exit 1; }

OUT_DIR="${PYG_WHEEL_OUT:-$HOME/pyg-wheels-arm}"
ASSUME_YES="${ASSUME_YES:-0}"
INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-1}"

confirm() {  # confirm "prompt" -> 0 if yes
    [ "$ASSUME_YES" = "1" ] && return 0
    if [ ! -t 0 ]; then
        say "${YELLOW}[!] $1 -- no TTY; set ASSUME_YES=1 to auto-accept. Assuming NO.${NC}"
        return 1
    fi
    local ans; read -r -p "$(echo -e "${YELLOW}$1 [y/N] ${NC}")" ans
    case "$ans" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

say "\n${CYAN}============================================================================${NC}"
say "${CYAN}   SHEPHERD-Advanced | PyG ARM wheel builder (version-agnostic)${NC}"
say "${CYAN}============================================================================${NC}"

# === Architecture guard =====================================================
[ "$(uname -m)" = "aarch64" ] || die "This compiles linux_aarch64 wheels and must run on an ARM target (got $(uname -m))."

# === Resolve uv + target venv ===============================================
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH (install it / source deploy env first)."

if [ -n "${TARGET_PYTHON:-}" ]; then
    TPY="$TARGET_PYTHON"
elif [ -x "$(pwd)/.venv/bin/python" ]; then
    TPY="$(pwd)/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    TPY="$VIRTUAL_ENV/bin/python"
else
    die "No target venv found. Run 'uv sync' first, or set TARGET_PYTHON=/path/to/python."
fi
[ -x "$TPY" ] || die "TARGET_PYTHON '$TPY' is not executable."
say "[INFO] Target python : $TPY"

# === Detect torch / CUDA / GPU arch from the target venv ====================
say "\n${CYAN}[STAGE 1/5] Detect build target${NC}"
say "----------------------------------------------------------------------------"
DETECT="$("$TPY" - <<'PY' 2>/dev/null
try:
    import torch
except Exception as e:
    print("NO_TORCH"); raise SystemExit(0)
ver = torch.__version__
cuda = torch.version.cuda or "none"
try:
    caps = sorted({f"{a}.{b}" for i in range(torch.cuda.device_count())
                   for (a, b) in [torch.cuda.get_device_capability(i)]})
except Exception:
    caps = []
print(f"{ver}|{cuda}|{';'.join(caps) if caps else 'none'}")
PY
)"
[ -n "$DETECT" ] && [ "$DETECT" != "NO_TORCH" ] || die "torch not importable in target venv. Run 'uv sync' first."
TORCH_VER="${DETECT%%|*}"; rest="${DETECT#*|}"
TORCH_CUDA="${rest%%|*}"; DET_ARCH="${rest#*|}"
say "[INFO] torch         : $TORCH_VER"
say "[INFO] torch CUDA    : $TORCH_CUDA"
[ "$TORCH_CUDA" = "none" ] && die "Target torch is a CPU build (no CUDA). PyG native ext need a CUDA torch."

# CUDA_HOME: honor env if valid, else derive from torch CUDA version / nvcc.
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "${CUDA_HOME:-/nonexistent}" ]; then
    if [ -d "/usr/local/cuda-$TORCH_CUDA" ]; then
        CUDA_HOME="/usr/local/cuda-$TORCH_CUDA"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    elif [ -d /usr/local/cuda ]; then
        CUDA_HOME="/usr/local/cuda"
    else
        die "Cannot locate CUDA toolkit. Set CUDA_HOME=/usr/local/cuda-XX.Y."
    fi
fi
# GPU arch: auto-detect from the live GPU, else require an override.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    [ "$DET_ARCH" != "none" ] || die "Could not detect GPU compute capability. Set TORCH_CUDA_ARCH_LIST (e.g. 12.0;12.1)."
    TORCH_CUDA_ARCH_LIST="$DET_ARCH"
fi

_NCPU="$(nproc)"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-$(( _NCPU > 4 ? _NCPU - 4 : 1 ))}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
say "[INFO] CUDA_HOME     : $CUDA_HOME"
say "[INFO] arch list     : $TORCH_CUDA_ARCH_LIST   FORCE_CUDA=1"
say "[INFO] parallelism   : MAX_JOBS=$MAX_JOBS (of $_NCPU cores; ~4 reserved for OS/browser)"

# === System toolchain check =================================================
say "\n${CYAN}[STAGE 2/5] System toolchain${NC}"
say "----------------------------------------------------------------------------"
FAIL=0
chk() { if "$@" >/dev/null 2>&1; then say "${GREEN}[OK]${NC} $1"; else say "${RED}[MISSING]${NC} $1"; FAIL=1; fi; }
chk "nvcc"  nvcc --version
chk "gcc"   gcc --version
chk "g++"   g++ --version
chk "cmake" cmake --version
chk "ninja" ninja --version
PYVER="$("$TPY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ -f "/usr/include/python$PYVER/Python.h" ]; then
    say "${GREEN}[OK]${NC} python$PYVER-dev (Python.h)"
else
    say "${RED}[MISSING]${NC} python$PYVER-dev -> sudo apt install python$PYVER-dev"; FAIL=1
fi
if [ "$FAIL" -ne 0 ]; then
    die "Missing system build tools (see above). On Ubuntu: sudo apt install build-essential cmake ninja-build python$PYVER-dev (CUDA toolkit separately)."
fi

# === Python build deps in the target venv ===================================
say "\n${CYAN}[STAGE 3/5] Build dependencies (in target venv)${NC}"
say "----------------------------------------------------------------------------"
# pip wheel + --no-build-isolation need pip/setuptools/wheel present in the venv
# (torch is already there). uv venvs ship none of these by default.
MISSING=()
for mod in pip setuptools wheel; do
    "$TPY" -c "import $mod" >/dev/null 2>&1 || MISSING+=("$mod")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    say "${YELLOW}[!] Build deps missing from target venv: ${MISSING[*]}${NC}"
    if confirm "Install them into $TPY (build-only; not added to uv.lock)?"; then
        uv pip install --python "$TPY" "${MISSING[@]}" || die "Failed to install build deps."
        say "${GREEN}[OK] build deps installed${NC}"
    else
        die "Build deps required. Re-run and accept, or 'uv pip install --python $TPY pip setuptools wheel'."
    fi
else
    say "${GREEN}[OK]${NC} pip / setuptools / wheel present"
fi

# === Compile =================================================================
say "\n${CYAN}[STAGE 4/5] Compiling (this can take many minutes per package)${NC}"
say "----------------------------------------------------------------------------"
mkdir -p "$OUT_DIR"
LOGDIR="$(mktemp -d)"
declare -a RESULTS
declare -a OK_PKGS

run_build() {  # run_build "<label>" "<logfile>" <cmd...>
    local label="$1" logf="$2"; shift 2
    local start=$SECONDS
    "$@" >"$logf" 2>&1 &
    local pid=$!
    if [ -t 1 ]; then
        local spin='|/-\' i=0
        while kill -0 "$pid" 2>/dev/null; do
            i=$(( (i + 1) % 4 ))
            printf "\r  ${CYAN}%s${NC} compiling %s ... %ds elapsed" "${spin:$i:1}" "$label" "$(( SECONDS - start ))"
            sleep 0.5
        done
        printf "\r\033[K"
    else
        say "  compiling $label ... (live log: $logf)"
    fi
    wait "$pid"; local rc=$?
    local dur=$(( SECONDS - start ))
    if [ $rc -eq 0 ]; then
        printf "  ${GREEN}[OK]${NC} %s (%ds)\n" "$label" "$dur"
    else
        printf "  ${RED}[FAIL]${NC} %s (%ds) -- last 30 log lines:\n" "$label" "$dur"
        tail -30 "$logf" | sed 's/^/      /'
        say "      ${YELLOW}full log: $logf${NC}"
    fi
    return $rc
}

build_one() {  # build_one <pkg-name> <pip-spec>
    local name="$1" spec="$2"
    say "\n${CYAN}>>> $name${NC}  ($spec)"
    # --no-deps: build ONLY this package's wheel, not its runtime deps
    # (otherwise pip drags numpy/scipy/etc. into the output dir).
    if run_build "$name" "$LOGDIR/$name.log" \
        "$TPY" -m pip wheel "$spec" -w "$OUT_DIR" --no-build-isolation --no-deps; then
        RESULTS+=("OK   $name")
        OK_PKGS+=("$name")
    else
        RESULTS+=("FAIL $name")
    fi
}

# Order: scatter first (sparse/cluster use it at runtime), then sparse/cluster,
# then the independent pyg-lib. scatter/sparse/cluster have PyPI sdists; pyg-lib
# does NOT publish to PyPI, so it must be built from its git source (submodules
# are fetched by pip). Specs overridable via *_SPEC for reproducible builds.
build_one "torch-scatter" "${SCATTER_SPEC:-torch-scatter}"
build_one "torch-sparse"  "${SPARSE_SPEC:-torch-sparse}"
build_one "torch-cluster" "${CLUSTER_SPEC:-torch-cluster}"
build_one "pyg-lib"       "${PYGLIB_SPEC:-git+https://github.com/pyg-team/pyg-lib.git}"

# === Summary + optional install =============================================
say "\n${CYAN}[STAGE 5/5] Summary${NC}"
say "----------------------------------------------------------------------------"
NFAIL=0
for r in "${RESULTS[@]}"; do
    case "$r" in OK*) say "  ${GREEN}$r${NC}" ;; *) say "  ${RED}$r${NC}"; NFAIL=$((NFAIL+1)) ;; esac
done
say "\n[INFO] Wheels in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.whl 2>/dev/null | sed 's/^/  /' || say "${YELLOW}  (none)${NC}"

if [ "$INSTALL_AFTER_BUILD" = "1" ] && [ "${#OK_PKGS[@]}" -gt 0 ]; then
    say "\n[INFO] Installing built wheels into target venv (--no-deps, from $OUT_DIR):"
    say "       ${OK_PKGS[*]}"
    # Install only what actually built, so one failure can't block the rest.
    uv pip install --python "$TPY" --no-deps --find-links "$OUT_DIR" "${OK_PKGS[@]}" 2>&1 | tail -8 \
        || say "${YELLOW}[WARN] install reported issues${NC}"
fi

say "\n[INFO] Import smoke test (in target venv):"
"$TPY" - <<'PY' || true
import importlib, torch
print(f"  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
for mod in ("torch_scatter", "torch_sparse", "torch_cluster", "pyg_lib"):
    try:
        m = importlib.import_module(mod)
        print(f"  [OK] {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  [--] {mod}: {e}")
PY

if [ "$NFAIL" -eq 0 ]; then
    say "\n${GREEN}All 4 wheels built.${NC}"
    say "${CYAN}Next (maintainer):${NC} upload $OUT_DIR/*.whl to a GitHub Release so deploy.sh can fetch them."
    say "${YELLOW}[REMINDER] These wheels are ABI-locked to torch $TORCH_VER (cu$TORCH_CUDA). Rebuild if torch changes.${NC}"
    exit 0
else
    say "\n${RED}$NFAIL package(s) failed.${NC} See logs in $LOGDIR (re-run a single one with e.g. SCATTER_SPEC=torch-scatter)."
    exit 1
fi
