@echo off
rem --- Wrapper: keep the window open on error so the user can read messages ---
call :main %*
set "_EXIT_CODE=%ERRORLEVEL%"
if %_EXIT_CODE% neq 0 (
    echo.
    echo ============================================================================
    echo   [ERROR] Launch failed ^(exit code %_EXIT_CODE%^). Review the messages above.
    echo ============================================================================
    echo.
    pause
)
exit /b %_EXIT_CODE%

rem ============================================================================
rem :main - actual launch logic (called as subroutine)
rem ============================================================================
:main
setlocal EnableExtensions

REM =======================
REM User-configurable flags
REM =======================
REM Example presets (uncomment ONE line you want; combine as needed)
REM set "COMMANDLINE_ARGS=--cudnn-sdpa"
REM set "COMMANDLINE_ARGS=--xformers"
REM set "COMMANDLINE_ARGS=--flash-attn --reinstall-flash-attn"
REM set "COMMANDLINE_ARGS=--xformers --attention-order=xformers,cudnn_sdpa,torch_sdpa"
REM set "COMMANDLINE_ARGS=--flash-attn --no-auto-install --print-plan"

REM Optionally support project-specific variable name (SHEP_COMMANDLINE_ARGS)
if defined SHEP_COMMANDLINE_ARGS (
    set "COMMANDLINE_ARGS=%SHEP_COMMANDLINE_ARGS%"
)

REM Resolve repo root from this .cmd location
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

REM Create/activate venv if missing
if not exist "%REPO_ROOT%\.venv" (
    echo [SHEPHERD] Creating .venv under %REPO_ROOT% ...
    python -m venv "%REPO_ROOT%\.venv" || (
        echo [ERROR] Failed to create venv. Ensure Python is on PATH.
        exit /b 1
    )
)

call "%REPO_ROOT%\.venv\Scripts\activate" || (
    echo [ERROR] Failed to activate venv.
    exit /b 1
)

REM Ensure pip is up-to-date (silent)
python -m pip install --upgrade pip >nul 2>&1

REM Call Python launcher (reads COMMANDLINE_ARGS env)
python -u "%REPO_ROOT%\scripts\launch\shep_launch.py"
set EXITCODE=%ERRORLEVEL%

REM Deactivate venv (optional)
call "%REPO_ROOT%\.venv\Scripts\deactivate" >nul 2>&1

exit /b %EXITCODE%