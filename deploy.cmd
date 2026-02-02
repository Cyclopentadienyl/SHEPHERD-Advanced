@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem ============================================================================
rem SHEPHERD-Advanced | Windows Deployment Script (Unified)
rem ============================================================================
rem
rem This script handles the complete deployment process:
rem   1. Environment setup (virtual environment + PyTorch)
rem   2. Core dependencies installation (with smart filtering)
rem   3. Installation validation & platform configuration
rem   4. Knowledge graph construction (optional)
rem
rem Usage:
rem   deploy.cmd                           (standard deployment)
rem   deploy.cmd requirements_arm.txt      (custom requirements file)
rem
rem Environment Variables:
rem   PYTHON_EXE        - Python launcher (default: py -3.12)
rem   TORCH_INDEX_URL   - PyTorch index URL (default: cu130)
rem   REQUIREMENTS_FILE - Custom requirements file
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

set "REQ_FILE=%~1"
if "%REQ_FILE%"=="" (
  if not "%REQUIREMENTS_FILE%"=="" (
    set "REQ_FILE=%REQUIREMENTS_FILE%"
  ) else (
    set "REQ_FILE=requirements_windows.txt"
  )
)

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

rem ============================================================================
rem STAGE 3: CORE DEPENDENCIES
rem ============================================================================
echo.
echo [STAGE 3/4] Core Dependencies Installation
echo ----------------------------------------------------------------------------

if not exist "%REQ_FILE%" (
  echo [WARN] Requirements file not found: %REQ_FILE%
  echo [INFO] Skipping core dependencies installation
  goto :skip_requirements
)

rem Filter out problematic packages (flash-attn, xformers - handled at launch time)
if not exist ".tmp" mkdir .tmp
echo [INFO] Filtering requirements (removing flash-attn, xformers - installed at launch)

rem [Bulletproof Fix] Construct regex using ASCII codes to bypass CMD syntax errors entirely
rem 94 = ^ (Caret), 124 = | (Pipe). We do not type these chars in the echo command.
echo import re, sys > ".tmp\filter_reqs.py"
echo input_file = r"%REQ_FILE%" >> ".tmp\filter_reqs.py"
echo output_file = r".tmp\req_filtered.txt" >> ".tmp\filter_reqs.py"
echo try: >> ".tmp\filter_reqs.py"
echo     with open(input_file, 'r', encoding='utf-8', errors='ignore') as f: >> ".tmp\filter_reqs.py"
echo         lines = f.read().splitlines() >> ".tmp\filter_reqs.py"
echo     # Construct regex safely using chr() >> ".tmp\filter_reqs.py"
echo     caret = chr(94) >> ".tmp\filter_reqs.py"
echo     pipe = chr(124) >> ".tmp\filter_reqs.py"
echo     keywords = ["flash[-_]?attn", "xformers", "sage[-_]?attention"] >> ".tmp\filter_reqs.py"
echo     pattern_str = caret + r"\s*(" + pipe.join(keywords) + r")\b" >> ".tmp\filter_reqs.py"
echo     pat = re.compile(pattern_str, re.I) >> ".tmp\filter_reqs.py"
echo     filtered = [l for l in lines if not pat.match(l)] >> ".tmp\filter_reqs.py"
echo     with open(output_file, 'w', encoding='utf-8') as f: >> ".tmp\filter_reqs.py"
echo         f.write('# -*- coding: utf-8 -*-\n' + '\n'.join(filtered) + '\n') >> ".tmp\filter_reqs.py"
echo     print("Filtered requirements written.") >> ".tmp\filter_reqs.py"
echo except Exception as e: >> ".tmp\filter_reqs.py"
echo     print(f"Error: {e}") >> ".tmp\filter_reqs.py"
echo     sys.exit(1) >> ".tmp\filter_reqs.py"

echo [INFO] Running filter script...
"%PY%" ".tmp\filter_reqs.py" || (
  echo [ERROR] Failed to filter requirements
  exit /b 3
)

echo [INFO] Installing core dependencies from filtered requirements
"%PIP%" install -r ".tmp\req_filtered.txt" || (
  echo [ERROR] Failed to install core dependencies
  exit /b 3
)
echo [OK] Core dependencies installed

:skip_requirements

rem Install Vector Index Backend (Voyager)
rem Strategy (v3.2): Windows uses Voyager (CPU) - cuVS not supported on Windows
echo [INFO] Installing vector index backend (Voyager)
echo [INFO] Note: cuVS is Linux-only; Windows uses Voyager (Spotify HNSW)
"%PIP%" install "voyager>=2.0" && (
  echo [OK] Voyager installed
) || (
  echo [ERROR] Voyager installation failed
  echo [HINT] Try manually: pip install voyager
  exit /b 3
)

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

rem Build knowledge graph (if script exists)
if exist "scripts\build_knowledge_graph.py" (
  echo [INFO] Building knowledge graph
  "%PY%" scripts\build_knowledge_graph.py && (
    echo [OK] Knowledge graph built
  ) || (
    echo [WARN] KG build returned non-zero
  )
) else (
  echo [INFO] scripts\build_knowledge_graph.py not found, skipping KG build
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
echo   2. Launch the system:
echo      launch_shepherd.cmd                     (default, PyTorch SDPA)
echo.
echo   3. Download data (if needed):
echo      .venv\Scripts\python.exe scripts\data_preparation\download_ontologies.py
echo.
echo   4. Run tests:
echo      .venv\Scripts\python.exe -m pytest tests\unit\
echo.
echo [TIP] Optional accelerators (FlashAttention, xFormers, SageAttention):
echo       These are auto-installed at launch time via arguments:
echo         launch_shepherd.cmd --flash-attn    (FlashAttention-2)
echo         launch_shepherd.cmd --xformers      (xFormers)
echo         launch_shepherd.cmd --sage-attn     (SageAttention)
echo       Compatibility not guaranteed; fallback to PyTorch SDPA is always available.
echo.
echo [TIP] See docs/deployment-guide.md for troubleshooting
echo.

pause

exit /b 0
