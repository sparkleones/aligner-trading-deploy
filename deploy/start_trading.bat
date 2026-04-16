@echo off
REM ╔══════════════════════════════════════════════════════════════╗
REM ║  Aligner Trading System — 24/7 Launcher                     ║
REM ║  Starts: Trading Engine + Dashboard + Watchdog               ║
REM ║  Usage:  start_trading.bat [--paper] [--live]                ║
REM ╚══════════════════════════════════════════════════════════════╝

setlocal enabledelayedexpansion

cd /d "%~dp0\.."
set PROJECT_ROOT=%cd%
set PYTHON=python

REM Parse arguments
set MODE=--paper
if "%1"=="--live" set MODE=
if "%1"=="--paper" set MODE=--paper

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║         ALIGNER TRADING SYSTEM v14 R5                   ║
echo  ║         Autonomous NIFTY Options Engine                 ║
echo  ╠══════════════════════════════════════════════════════════╣
if "%MODE%"=="--paper" (
    echo  ║  MODE: PAPER TRADING (safe simulation)                 ║
) else (
    echo  ║  MODE: ** LIVE TRADING ** (REAL MONEY)                 ║
)
echo  ║  Dashboard: http://localhost:8510/terminal              ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM Activate venv if exists
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
    echo  [OK] Virtual environment activated
) else if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
    echo  [OK] Virtual environment activated
) else (
    echo  [WARN] No virtual environment found — using system Python
)

REM Load .env
if exist "%PROJECT_ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_ROOT%\.env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%b"=="" set "%%a=%%b"
        )
    )
    echo  [OK] Environment variables loaded from .env
)

REM Create log directory
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"

echo.
echo  Starting watchdog (manages trading engine + dashboard)...
echo  Press Ctrl+C to stop everything gracefully.
echo  ─────────────────────────────────────────────────────────
echo.

REM Start via watchdog (auto-restart on crash)
%PYTHON% "%PROJECT_ROOT%\deploy\watchdog.py" %MODE%

echo.
echo  Trading system stopped.
pause
