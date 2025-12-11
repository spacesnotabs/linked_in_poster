@echo off
setlocal enabledelayedexpansion

rem Run the FastAPI web UI from the repo root.
rem Usage:
rem   run_web.bat                 -> starts at http://127.0.0.1:8000
rem   run_web.bat --reload        -> starts with auto-reload
rem   run_web.bat --host 0.0.0.0 --port 9000

rem Ensure we run from the repository root regardless of where the script is called.
cd /d "%~dp0"

rem Activate local venv if present.
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [run_web] No venv found at venv\Scripts\activate.bat. Using system Python.
)

rem Pick a Python executable.
set "PYTHON_EXE=python"
where python >nul 2>nul
if errorlevel 1 set "PYTHON_EXE=py"

rem Defaults.
set "HOST=127.0.0.1"
set "PORT=8000"
set "RELOAD="
set "EXTRA_ARGS="

rem Parse a small set of common options; forward unknown flags to uvicorn.
:parse_args
if "%~1"=="" goto run_server

if /i "%~1"=="--reload" (
    set "RELOAD=--reload"
    shift
    goto parse_args
)
if /i "%~1"=="-r" (
    set "RELOAD=--reload"
    shift
    goto parse_args
)
if /i "%~1"=="--host" (
    set "HOST=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)

set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto parse_args

:run_server
rem Make sure the repo root is on PYTHONPATH so `src.*` imports work.
set "PYTHONPATH=%CD%"

echo [run_web] Starting server on %HOST%:%PORT% %RELOAD%
%PYTHON_EXE% -m uvicorn src.apps.web.main:app --host %HOST% --port %PORT% %RELOAD% %EXTRA_ARGS%

endlocal
