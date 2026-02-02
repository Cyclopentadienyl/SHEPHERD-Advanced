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
# Usage:
#   ./deploy.sh                     (Standard deployment)
#   ./deploy.sh requirements.txt    (Custom requirements file)
#
# Environment Variables:
#   PYTHON_EXE        - Python executable (default: python3)
#   TORCH_INDEX_URL   - PyTorch index URL (default: cu128 for Linux)
#   INSTALL_XFORMERS  - Set to 1 to install xformers
#   FLASHATTN_WHEEL   - Path to prebuilt flash-attn wheel (Recommended for ARM)
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
# [NOTE] Using cu130 (CUDA 13.0) unified across all platforms.
# PyTorch 2.9.0 + cu130 for best ecosystem compatibility and latest hardware support.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

REQ_FILE="${1:-requirements.txt}"
if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="requirements_linux.txt"
    # Fallback to generic if linux specific doesn't exist
    if [ ! -f "$REQ_FILE" ]; then REQ_FILE="requirements.txt"; fi
fi

# Detect Architecture
ARCH=$(uname -m)
echo -e "${CYAN}[INFO] Detected Architecture: ${ARCH}${NC}"
if [ "$ARCH" == "aarch64" ]; then
    echo -e "${YELLOW}[INFO] Running on ARM64 (Likely DGX Spark / Grace Hopper)${NC}"
    echo -e "${YELLOW}[INFO] Note: Some pre-compiled wheels (like flash-attn) might be rare for aarch64.${NC}"
fi

# ============================================================================
# STAGE 1: ENVIRONMENT SETUP
# ============================================================================
echo -e "\n${CYAN}[STAGE 1/5] Environment Setup${NC}"
echo "----------------------------------------------------------------------------"

# Check Python
echo -e "[INFO] Python executable: $PYTHON_EXE"
if ! $PYTHON_EXE --version > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] Python not found. Please install Python 3.10+${NC}"
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

# Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo -e "[INFO] Creating virtual environment at .venv..."
    $PYTHON_EXE -m venv .venv || { echo -e "${RED}[ERROR] Failed to create venv${NC}"; exit 1; }
    echo -e "${GREEN}[OK] Virtual environment created${NC}"
else
    echo -e "[INFO] Virtual environment already exists"
fi

# Activate Environment
# We use the full path to avoid shell sourcing issues in scripts
PY=".venv/bin/python"
PIP=".venv/bin/pip"

# Upgrade pip
echo -e "[INFO] Upgrading pip, setuptools, wheel..."
"$PY" -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || {
    echo -e "${RED}[ERROR] Failed to upgrade pip${NC}"; exit 1;
}
echo -e "${GREEN}[OK] Pip upgraded${NC}"

# ============================================================================
# STAGE 2: PYTORCH INSTALLATION
# ============================================================================
echo -e "\n${CYAN}[STAGE 2/5] PyTorch Installation${NC}"
echo "----------------------------------------------------------------------------"

echo -e "[INFO] Installing PyTorch stack (torch 2.9.x + cu130)"
echo -e "[INFO] Index URL: $TORCH_INDEX_URL"

# Install Torch
"$PIP" install --index-url "$TORCH_INDEX_URL" "torch>=2.9,<2.10" "torchvision>=0.20,<0.21" "torchaudio>=2.9,<2.10" || {
    echo -e "${RED}[ERROR] Failed to install PyTorch stack${NC}"
    echo -e "${YELLOW}[HINT] If on DGX Spark, ensure you have internet access or use the local NVIDIA mirror.${NC}"
    exit 2
}
echo -e "${GREEN}[OK] PyTorch stack installed${NC}"

# Validate
"$PY" -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" || {
    echo -e "${YELLOW}[WARN] PyTorch validation returned non-zero${NC}"
}

# ============================================================================
# STAGE 3: CORE DEPENDENCIES
# ============================================================================
echo -e "\n${CYAN}[STAGE 3/5] Core Dependencies Installation${NC}"
echo "----------------------------------------------------------------------------"

if [ ! -f "$REQ_FILE" ]; then
    echo -e "${YELLOW}[WARN] Requirements file not found: $REQ_FILE${NC}"
    echo -e "[INFO] Skipping core dependencies"
else
    mkdir -p .tmp
    echo -e "[INFO] Filtering requirements (removing flash-attn, faiss)..."
    
    # Generate Python script to filter requirements safely
    cat <<EOF > .tmp/filter_reqs.py
import re, sys
input_file = "$REQ_FILE"
output_file = ".tmp/req_filtered.txt"
try:
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()
    # Filter out flash-attn and faiss
    pat = re.compile(r'^\s*(flash[-_]?attn|faiss[-_]?gpu|faiss\s*$|faiss-cpu)\b', re.I)
    filtered = [l for l in lines if not pat.match(l)]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered) + '\n')
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

    "$PY" .tmp/filter_reqs.py || { echo -e "${RED}[ERROR] Filtering failed${NC}"; exit 3; }

    echo -e "[INFO] Installing core dependencies..."
    "$PIP" install -r .tmp/req_filtered.txt || { echo -e "${RED}[ERROR] Pip install failed${NC}"; exit 3; }
    echo -e "${GREEN}[OK] Core dependencies installed${NC}"
fi

# Install FAISS
echo -e "[INFO] Installing vector index backend..."
# Strategy: Try faiss-gpu -> faiss-cpu -> hnswlib
# On ARM, faiss-gpu via pip is often missing.
FAISS_INSTALLED=0

if [ "$ARCH" == "aarch64" ]; then
    echo -e "${YELLOW}[INFO] ARM64 detected: Skipping direct faiss-gpu pip install (often unavailable).${NC}"
    echo -e "[INFO] Trying faiss-cpu..."
    if "$PIP" install "faiss-cpu>=1.9.0"; then
        echo -e "${GREEN}[OK] faiss-cpu installed${NC}"
        FAISS_INSTALLED=1
    fi
else
    # x86_64 Logic
    if "$PIP" install "faiss-gpu-cu12"; then
        echo -e "${GREEN}[OK] faiss-gpu-cu12 installed${NC}"
        FAISS_INSTALLED=1
    elif "$PIP" install "faiss-cpu>=1.9.0"; then
        echo -e "${GREEN}[OK] faiss-cpu installed${NC}"
        FAISS_INSTALLED=1
    fi
fi

if [ $FAISS_INSTALLED -eq 0 ]; then
    echo -e "${YELLOW}[WARN] FAISS failed, installing hnswlib as fallback${NC}"
    "$PIP" install hnswlib || echo -e "${YELLOW}[WARN] hnswlib failed too${NC}"
fi

# ============================================================================
# STAGE 4: OPTIONAL ACCELERATORS
# ============================================================================
echo -e "\n${CYAN}[STAGE 4/5] Optional Accelerators${NC}"
echo "----------------------------------------------------------------------------"

# xFormers
if [ "${INSTALL_XFORMERS:-0}" == "1" ]; then
    echo -e "[INFO] Installing xformers..."
    # Note: On ARM, installing xformers via pip might trigger a long build process if wheels aren't found.
    if "$PIP" install --only-binary=:all: --index-url "$TORCH_INDEX_URL" xformers; then
        echo -e "${GREEN}[OK] xformers installed${NC}"
    else
        echo -e "${YELLOW}[WARN] xformers install failed (or no binary wheel found)${NC}"
    fi
fi

# FlashAttention-2
if [ -n "${FLASHATTN_WHEEL:-}" ] && [ -f "$FLASHATTN_WHEEL" ]; then
    echo -e "[INFO] Installing flash-attn from wheel: $FLASHATTN_WHEEL"
    "$PIP" install "$FLASHATTN_WHEEL" || echo -e "${YELLOW}[WARN] Wheel install failed${NC}"
else
    echo -e "[INFO] FLASHATTN_WHEEL not set. Use 'pip install flash-attn' manually if needed."
    echo -e "${YELLOW}[HINT] On DGX Spark (ARM), compiling flash-attn takes time. Pre-built wheels are recommended.${NC}"
fi

# ============================================================================
# STAGE 5: VALIDATION & FINALIZATION
# ============================================================================
echo -e "\n${CYAN}[STAGE 5/5] Validation & Finalization${NC}"
echo "----------------------------------------------------------------------------"

# Validation
if [ -f "scripts/validate_installation.py" ]; then
    echo -e "[INFO] Running validation..."
    "$PY" scripts/validate_installation.py || echo -e "${YELLOW}[WARN] Validation returned warnings${NC}"
fi

# Config Generation
mkdir -p configs
if [ ! -f "configs/platform.yaml" ]; then
    echo -e "[INFO] Generating platform.yaml..."
    cat <<EOF > configs/platform.yaml
platform: linux_${ARCH}_blackwell
cuda_version: auto
model_config:
  attention_backend: auto
vector_index:
  backend: auto
EOF
    echo -e "${GREEN}[OK] Config generated${NC}"
fi

# Knowledge Graph
if [ -f "scripts/build_knowledge_graph.py" ]; then
    echo -e "[INFO] Building knowledge graph..."
    "$PY" scripts/build_knowledge_graph.py
fi

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}   Deployment Complete!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "[INFO] Virtual environment: .venv"
echo -e "[INFO] Python Interpreter:  $PY"
echo -e "\nNext steps:"
echo -e "   1. Activate environment:"
echo -e "      ${YELLOW}source .venv/bin/activate${NC}"
echo -e "   2. Launch system:"
echo -e "      ${YELLOW}./launch_shepherd.sh${NC}"
echo -e "\n[TIP] If using DGX Spark, ensure 'nvcc --version' matches your torch CUDA version."
echo ""