@echo off
REM ╔══════════════════════════════════════════════════════════════╗
REM ║  Aligner Trading System — Stop All Processes                 ║
REM ╚══════════════════════════════════════════════════════════════╝

setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set PROJECT_ROOT=%cd%

echo.
echo  Stopping Aligner Trading System...
echo.

REM Stop the scheduled task if running
schtasks /end /tn "AlignerTradingSystem" 2>nul
if %errorlevel% == 0 (
    echo  [OK] Scheduled task stopped
) else (
    echo  [--] No scheduled task running
)

REM Kill watchdog processes
taskkill /f /fi "WINDOWTITLE eq *watchdog*" 2>nul

REM Kill trading engine (run_autonomous.py)
for /f "tokens=2" %%i in ('tasklist /fi "IMAGENAME eq python.exe" /v 2^>nul ^| findstr /i "run_autonomous"') do (
    taskkill /pid %%i /f 2>nul
    echo  [OK] Trading engine stopped (PID %%i)
)

REM Kill dashboard (uvicorn on port 8510)
for /f "tokens=2" %%i in ('tasklist /fi "IMAGENAME eq python.exe" /v 2^>nul ^| findstr /i "uvicorn"') do (
    taskkill /pid %%i /f 2>nul
    echo  [OK] Dashboard stopped (PID %%i)
)

REM Clean up lock files so next startup doesn't see stale locks
if exist "%PROJECT_ROOT%\data\.trading_watchdog.lock" (
    del /f "%PROJECT_ROOT%\data\.trading_watchdog.lock" 2>nul
    echo  [OK] Watchdog lock file removed
)
if exist "%PROJECT_ROOT%\data\.trading_engine.lock" (
    del /f "%PROJECT_ROOT%\data\.trading_engine.lock" 2>nul
    echo  [OK] Engine lock file removed
)

echo.
echo  All processes stopped. Lock files cleaned.
echo.
pause
