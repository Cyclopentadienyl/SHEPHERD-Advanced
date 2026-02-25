@echo off
rem --- Wrapper: ensure the window never closes before the user reads output ---
rem   All "exit /b N" inside :main return here, then we pause.
call :main %*
set "_EXIT_CODE=%ERRORLEVEL%"
echo.
if %_EXIT_CODE% neq 0 (
    echo ============================================================================
    echo   [ERROR] Deployment failed at stage %_EXIT_CODE%. Review the messages above.
    echo ============================================================================
    echo.
)
pause
exit /b %_EXIT_CODE%

rem ============================================================================
rem :main - actual deployment logic (called as subroutine)
rem ============================================================================
:main
setlocal EnableExtensions DisableDelayedExpansion

rem ============================================================================
rem SHEPHERD-Advanced | Windows Deployment Script (Unified)
rem ============================================================================
rem
rem This script handles the complete deployment process:
rem   1. Environment setup (virtual environment)
rem   2. PyTorch installation (CUDA-specific, via --index-url)
rem   3. Core dependencies (from pyproject.toml via pip install .)
rem   4. Installation validation & platform configuration
rem
rem Usage:
rem   deploy.cmd
rem
rem Environment Variables:
rem   PYTHON_EXE        - Python launcher (default: py -3.12)
rem   TORCH_INDEX_URL   - PyTorch index URL (default: cu130)
rem
rem Note on Optional Accelerators (xFormers, FlashAttention, SageAttention):
rem   These are NOT installed during deployment. They are auto-installed
rem   at launch time via command-line arguments. See launch_shepherd.cmd.
rem
rem ============================================================================

echo.
echo ============================================================================
echo   SHEPHERD-Advanced Deployment Script
echo ============================================================================
echo.

rem === Configuration ===
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=py -3.12"
if "%TORCH_INDEX_URL%"=="" set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"

rem Dependencies are now managed via pyproject.toml (single source of truth).
rem Platform-specific CUDA packages (torch, torchvision, torchaudio) are
rem installed in Stage 2 with --index-url. All other deps come from
rem pip install . in Stage 3.

rem ============================================================================
rem STAGE 1: ENVIRONMENT SETUP
rem ============================================================================
echo.
echo [STAGE 1/4] Environment Setup
echo ----------------------------------------------------------------------------

echo [INFO] Python launcher: %PYTHON_EXE%
%PYTHON_EXE% --version || (
  echo [ERROR] Python not found. Please install Python 3.12+ from python.org
  exit /b 1
)

rem Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  echo [INFO] NVIDIA GPU detected
  for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv^,noheader') do echo [INFO] GPU: %%i
) else (
  echo [WARN] nvidia-smi not found. GPU may not be available.
)

rem Create virtual environment
if not exist ".venv" (
  echo [INFO] Creating virtual environment at .venv
  %PYTHON_EXE% -m venv .venv || (
    echo [ERROR] Failed to create virtual environment
    exit /b 1
  )
  echo [OK] Virtual environment created
) else (
  echo [INFO] Virtual environment already exists
)

rem Activate environment
set "PY=.venv\Scripts\python.exe"
set "PIP=.venv\Scripts\pip.exe"

rem Upgrade pip
echo [INFO] Upgrading pip, setuptools, wheel
"%PY%" -m pip install --upgrade pip setuptools wheel >nul 2>&1 || (
  echo [ERROR] Failed to upgrade pip
  exit /b 1
)
echo [OK] pip upgraded

rem ============================================================================
rem STAGE 2: PYTORCH INSTALLATION
rem ============================================================================
echo.
echo [STAGE 2/4] PyTorch Installation
echo ----------------------------------------------------------------------------

echo [INFO] Installing PyTorch stack (torch 2.9.0 + cu130)
echo [INFO] Index URL: %TORCH_INDEX_URL%
rem CRITICAL: Use exact torch==2.9.0 to ensure pyg-lib compatibility (torch 2.9.1+ breaks pyg-lib)
"%PIP%" install --index-url %TORCH_INDEX_URL% "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" || (
  echo [ERROR] Failed to install PyTorch stack
  echo [HINT] Check CUDA version and index URL from https://pytorch.org/get-started/locally/
  exit /b 2
)
echo [OK] PyTorch stack installed

rem Validate PyTorch + CUDA
echo [INFO] Validating PyTorch installation
"%PY%" -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || (
  echo [WARN] PyTorch validation failed
)

rem --- PyTorch Geometric (PyG) ---
rem PyG is required for heterogeneous GNN message passing (HeteroGNNLayer).
rem The companion native libraries must match the exact torch + CUDA version.
echo.
echo [INFO] Installing PyTorch Geometric (PyG)
"%PIP%" install torch_geometric || (
  echo [ERROR] Failed to install torch_geometric
  exit /b 2
)
echo [INFO] Installing PyG native extensions (pyg-lib, torch-sparse, torch-scatter, torch-cluster)
"%PIP%" install pyg-lib torch-sparse torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cu130.html || (
  echo [WARN] Some PyG native extensions failed to install.
  echo [HINT] The model will still work but may be slower without native sparse ops.
  echo [HINT] Check compatibility at https://data.pyg.org/whl/
)
echo [OK] PyTorch Geometric installed

rem ============================================================================
rem STAGE 3: CORE DEPENDENCIES
rem ============================================================================
echo.
echo [STAGE 3/4] Core Dependencies Installation
echo ----------------------------------------------------------------------------

rem Install all pure-Python dependencies from pyproject.toml (single source of truth).
rem This includes: pronto, networkx, pandas, numpy, scipy, pydantic, fastapi,
rem uvicorn, gradio, voyager, tqdm, requests, python-dotenv, etc.
rem
rem NOTE: torch/CUDA packages are NOT in pyproject.toml (installed in Stage 2).
rem       pip install . will NOT modify the existing torch installation.
echo [INFO] Installing project dependencies from pyproject.toml
"%PIP%" install . || (
  echo [ERROR] Failed to install core dependencies
  exit /b 3
)
echo [OK] Core dependencies installed

rem ============================================================================
rem STAGE 4: VALIDATION & FINALIZATION
rem ============================================================================
echo.
echo [STAGE 4/4] Validation ^& Finalization
echo ----------------------------------------------------------------------------

rem Run validation script (if exists)
if exist "scripts\validate_installation.py" (
  echo [INFO] Running installation validation
  "%PY%" scripts\validate_installation.py && (
    echo [OK] Validation passed
  ) || (
    echo [WARN] Validation failed or returned non-zero
  )
) else (
  echo [INFO] scripts\validate_installation.py not found, skipping validation
)

rem Generate platform configuration (Dynamic via Python)
echo [INFO] Generating platform configuration...
"%PY%" scripts\generate_config.py || (
    echo [WARN] Failed to generate platform config automatically
)

echo.
echo ============================================================================
echo   Deployment Complete!
echo ============================================================================
echo.
echo [INFO] Virtual environment: .venv
echo [INFO] Python: %PY%
echo [INFO] Platform config: configs\platform.yaml
echo.
echo Next steps:
echo   1. Test the installation:
echo      .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
echo.
echo   2. Prepare data (first time only):
echo      .venv\Scripts\python.exe scripts\data_preparation\download_ontologies.py
echo      .venv\Scripts\python.exe scripts\build_knowledge_graph.py
echo.
echo   3. Launch the system:
echo      launch_shepherd.cmd
echo.
echo   4. Run tests (optional):
echo      .venv\Scripts\python.exe -m pytest tests\unit\
echo.
echo [TIP] Optional accelerators (FlashAttention, xFormers, SageAttention):
echo       These are auto-installed at launch time via arguments:
echo         launch_shepherd.cmd --flash-attn    (FlashAttention-2)
echo         launch_shepherd.cmd --xformers      (xFormers)
echo         launch_shepherd.cmd --sage-attn     (SageAttention)
echo       Compatibility not guaranteed; fallback to PyTorch SDPA is always available.
echo.
echo [TIP] For development (pytest, linting, etc.):
echo       .venv\Scripts\pip.exe install -e ".[dev]"
echo.
echo [TIP] See docs/deployment-guide.md for troubleshooting
echo.

exit /b 0
