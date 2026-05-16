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
rem Phase C onwards: this script uses uv (https://docs.astral.sh/uv/) as the
rem primary deployment tool. PyTorch (torch/torchvision/torchaudio) is now
rem managed by uv via [tool.uv.sources] in pyproject.toml and recorded in
rem uv.lock for cross-platform reproducibility. PyG native extensions
rem (pyg-lib, torch-scatter/sparse/cluster) remain out of the lock and are
rem installed via 'uv pip install' with graceful fallback on missing wheels.
rem
rem Stages:
rem   1. Hardware detection + uv setup (auto-install with y/N confirmation)
rem   2. Environment + dependency sync (uv sync from uv.lock)
rem   3. PyG native extensions (out-of-lock, graceful skip)
rem   4. Validation
rem
rem Usage:
rem   deploy.cmd
rem
rem Environment Variables:
rem   PYTHON_EXE   - Python launcher used by uv to find a base interpreter
rem                  (default: py -3.12)
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
set "TORCH_VER=2.10.0"
set "CUDA_TAG=cu130"
set "PYG_WHEEL_URL=https://data.pyg.org/whl/torch-%TORCH_VER%+%CUDA_TAG%.html"

rem ============================================================================
rem STAGE 1/4: HARDWARE DETECTION & uv SETUP
rem ============================================================================
echo.
echo [STAGE 1/4] Hardware Detection ^& uv Setup
echo ----------------------------------------------------------------------------

echo [INFO] Bootstrap Python launcher: %PYTHON_EXE%
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

rem --- uv detection ---
rem Use goto-style flow rather than if-blocks because set/p variables don't
rem expand correctly inside a parenthesized if-block without DelayedExpansion.
where uv >nul 2>&1
if errorlevel 1 goto UV_NEED_INSTALL
goto UV_VERIFY

:UV_NEED_INSTALL
echo.
echo [INFO] uv is not installed. uv is required for SHEPHERD-Advanced deployment.
echo [INFO] Official installer: https://docs.astral.sh/uv/getting-started/installation/
echo.
set "ans="
set /p ans="Install uv now via official PowerShell installer? (y/N): "
if /i "%ans%"=="y"   goto UV_DO_INSTALL
if /i "%ans%"=="yes" goto UV_DO_INSTALL
echo [ERROR] Deployment aborted: uv is required but not installed.
exit /b 1

:UV_DO_INSTALL
echo [INFO] Installing uv via PowerShell installer...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
  echo [ERROR] uv installer failed ^(network error or PowerShell policy^)
  exit /b 1
)
rem uv installs to %USERPROFILE%\.local\bin by default; add to PATH for this session
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:UV_VERIFY
where uv >nul 2>&1
if errorlevel 1 (
  echo [ERROR] uv installed but not found on PATH.
  echo [HINT] Open a new terminal so PATH is refreshed, then re-run deploy.cmd
  exit /b 1
)
echo [OK] uv detected:
uv --version

rem ============================================================================
rem STAGE 2/4: ENVIRONMENT + DEPENDENCY SYNC (uv.lock driven)
rem ============================================================================
echo.
echo [STAGE 2/4] Environment + Dependency Sync
echo ----------------------------------------------------------------------------

rem uv sync creates .venv if missing and installs everything from uv.lock.
rem torch/torchvision/torchaudio are routed to cu130 index via [tool.uv.sources].
rem --inexact: do NOT remove packages outside the lock; protects Stage 3 ext.
echo [INFO] Running 'uv sync --inexact' (creates .venv, installs lock contents)
uv sync --inexact
if errorlevel 1 (
  echo [ERROR] uv sync failed. Check network connectivity.
  echo [HINT] If uv.lock is out of sync with pyproject.toml, run 'uv lock' first.
  exit /b 2
)
echo [OK] uv.lock contents synced into .venv

set "PY=.venv\Scripts\python.exe"

rem NOTE on sm_121 warning (Blackwell GB10 / DGX Spark):
rem   PyTorch ^< 2.10 may emit "Found GPU0 ... with cuda capability sm_121.
rem   PyTorch supports cuda capability sm_80 - sm_120." This is harmless
rem   (sm_121 is SASS binary compatible with sm_120, confirmed by maintainer
rem   ptrblck). Fixed in PyTorch 2.10+ — kept for reference.

rem Validate PyTorch + CUDA
echo [INFO] Validating PyTorch installation
"%PY%" -c "import torch; print(f'PyTorch {torch.__version__} | CUDA build: {torch.version.cuda} | available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A (CPU-only at deploy time)\"}')" || (
  echo [WARN] PyTorch validation returned non-zero
)

rem ============================================================================
rem STAGE 3/4: PyG NATIVE EXTENSIONS (out-of-lock, graceful fallback)
rem ============================================================================
echo.
echo [STAGE 3/4] PyG Native Extensions
echo ----------------------------------------------------------------------------
echo [INFO] PyG native ext are installed outside uv.lock because their wheel
echo [INFO] index is torch-version-coupled HTML and three of them lack Windows
echo [INFO] wheels. Missing wheels fall back to torch.scatter_reduce (built-in).
echo.

rem torch_geometric: pure Python high-level API; install via uv pip post-sync
echo [INFO] Installing torch_geometric (high-level PyG API)
uv pip install torch_geometric
if errorlevel 1 (
  echo [ERROR] Failed to install torch_geometric
  exit /b 3
)
echo [OK] torch_geometric installed

rem pyg-lib: native kernels, has wheels for Linux x86 + ARM + Windows
echo.
echo [INFO] Installing pyg-lib (native GNN kernels)
uv pip install pyg-lib --find-links %PYG_WHEEL_URL% 2>nul && (
  echo [OK] pyg-lib installed
) || (
  echo [SKIP] pyg-lib: no pre-built wheel for this platform/torch combination.
  echo        This is non-critical; PyG will use PyTorch-native fallbacks.
)

rem torch-scatter / torch-sparse / torch-cluster: third-party ext
rem Linux wheels available; Windows wheels never published by upstream maintainers.
echo.
echo [INFO] Installing torch-scatter/sparse/cluster (third-party ext, optional)
uv pip install torch-scatter torch-sparse torch-cluster --find-links %PYG_WHEEL_URL% --only-binary :all: 2>nul && (
  echo [OK] PyG third-party extensions installed
) || (
  echo [SKIP] PyG third-party extensions: no pre-built wheels for this platform/torch combo.
  echo        Expected on Windows ^(maintainers never published Windows wheels^).
  echo        Impact: NONE -- HeteroConv + GAT/SAGE auto-use torch.scatter_reduce
  echo        ^(PyTorch 2.0+ native, equivalent perf for our architecture^).
)

rem ============================================================================
rem STAGE 4/4: VALIDATION & FINALIZATION
rem ============================================================================
echo.
echo [STAGE 4/4] Validation ^& Finalization
echo ----------------------------------------------------------------------------

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

echo.
echo ============================================================================
echo   Deployment Complete!
echo ============================================================================
echo.
echo [INFO] Virtual environment: .venv
echo [INFO] Python:              %PY%
echo [INFO] Lock file:           uv.lock (commit'd to repo)
echo.
echo Next steps:
echo   1. Verify the installation:
echo      .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
echo.
echo   2. Quick demo (synthetic data, ~1 min):
echo      .venv\Scripts\python.exe scripts\setup_demo.py --train-model
echo.
echo   3. Real data pipeline (manual HPO download required first;
echo      see data\external\README.md for the required annotation files):
echo      .venv\Scripts\python.exe scripts\build_knowledge_graph.py
echo      .venv\Scripts\python.exe scripts\compute_shortest_paths.py
echo      .venv\Scripts\python.exe scripts\train_model.py
echo      .venv\Scripts\python.exe scripts\build_index.py
echo.
echo   4. Launch the system:
echo      launch_shepherd.cmd
echo.
echo   5. Run tests (optional):
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
echo       uv sync --extra dev
echo.
echo [TIP] To regenerate uv.lock after editing pyproject.toml:
echo       uv lock
echo.
echo [TIP] See deployment-guide.md for troubleshooting
echo.

exit /b 0
