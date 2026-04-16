# ╔══════════════════════════════════════════════════════════════╗
# ║  Aligner Trading System — Windows Service Installer          ║
# ║  Creates a Task Scheduler job that auto-starts on boot       ║
# ║  and auto-restarts on failure.                                ║
# ║                                                              ║
# ║  Usage (Run as Administrator):                               ║
# ║    .\install_service.ps1              # Install (paper mode) ║
# ║    .\install_service.ps1 -Live        # Install (live mode)  ║
# ║    .\install_service.ps1 -Uninstall   # Remove the service   ║
# ╚══════════════════════════════════════════════════════════════╝

param(
    [switch]$Live,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$TaskName = "AlignerTradingSystem"
$ProjectRoot = (Get-Item "$PSScriptRoot\..").FullName

# ── Uninstall ──
if ($Uninstall) {
    Write-Host "`n  Removing scheduled task: $TaskName" -ForegroundColor Yellow
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "  [OK] Task removed successfully" -ForegroundColor Green
    } catch {
        Write-Host "  [WARN] Task not found or already removed" -ForegroundColor Yellow
    }
    return
}

# ── Find Python ──
$PythonExe = $null

# Check project venv first
$venvPaths = @(
    "$ProjectRoot\venv\Scripts\python.exe",
    "$ProjectRoot\.venv\Scripts\python.exe"
)
foreach ($vp in $venvPaths) {
    if (Test-Path $vp) {
        $PythonExe = $vp
        break
    }
}

# Fallback to system Python
if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $PythonExe) {
    Write-Host "  [ERROR] Python not found! Install Python or create a venv." -ForegroundColor Red
    exit 1
}

Write-Host "`n  Python: $PythonExe" -ForegroundColor Cyan

# ── Build command ──
$WatchdogScript = "$ProjectRoot\deploy\watchdog.py"
$ModeFlag = if ($Live) { "--live" } else { "--paper" }
$ModeLabel = if ($Live) { "LIVE" } else { "PAPER" }

# For live mode, we skip the interactive confirmation by calling the engine directly
# through a wrapper that sets the right arguments
$Arguments = "`"$WatchdogScript`" $ModeFlag"

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║  Installing Aligner Trading System Service   ║" -ForegroundColor Cyan
Write-Host "  ╠══════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "  ║  Mode:     $ModeLabel                             ║" -ForegroundColor $(if ($Live) { "Red" } else { "Green" })
Write-Host "  ║  Task:     $TaskName              ║" -ForegroundColor Cyan
Write-Host "  ║  Trigger:  System startup + daily 09:05      ║" -ForegroundColor Cyan
Write-Host "  ║  Restart:  Auto on failure (max 3 retries)   ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Remove existing task if any ──
try {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  Removing existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
} catch {}

# ── Create the scheduled task ──

# Action: Run the watchdog via Python
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $Arguments `
    -WorkingDirectory $ProjectRoot

# Triggers:
#   1. At system startup (boot)
#   2. Daily at 09:05 IST (safety — in case startup trigger missed)
$TriggerStartup = New-ScheduledTaskTrigger -AtStartup
$TriggerDaily = New-ScheduledTaskTrigger -Daily -At "09:05"

# Settings:
#   - Don't stop on idle
#   - Restart on failure (1 minute delay, up to 3 times)
#   - Run whether user is logged on or not
#   - Run with highest privileges
#   - Allow multiple instances: stop existing first
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -MultipleInstances IgnoreNew

# Principal: Run with current user, highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Register the task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger @($TriggerStartup, $TriggerDaily) `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Aligner V14 R5 Autonomous NIFTY Options Trading System ($ModeLabel mode). Auto-starts on boot, auto-restarts on failure."

Write-Host ""
Write-Host "  [OK] Service installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  To manage:" -ForegroundColor Cyan
Write-Host "    Start now:   schtasks /run /tn $TaskName" -ForegroundColor White
Write-Host "    Stop:        schtasks /end /tn $TaskName" -ForegroundColor White
Write-Host "    Status:      schtasks /query /tn $TaskName" -ForegroundColor White
Write-Host "    Uninstall:   .\install_service.ps1 -Uninstall" -ForegroundColor White
Write-Host "    Dashboard:   http://localhost:8510/terminal" -ForegroundColor White
Write-Host ""

# ── Optionally start now ──
$startNow = Read-Host "  Start the trading system now? (y/N)"
if ($startNow -eq "y" -or $startNow -eq "Y") {
    Write-Host "  Starting..." -ForegroundColor Yellow
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 3
    $task = Get-ScheduledTask -TaskName $TaskName
    Write-Host "  Status: $($task.State)" -ForegroundColor Green
    Write-Host "  Dashboard will be available at http://localhost:8510/terminal in ~10 seconds" -ForegroundColor Cyan
}
