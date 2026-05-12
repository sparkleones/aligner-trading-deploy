# Register Windows Task Scheduler entry for weekly DCA Telegram alerts.
#
# Runs every Monday at 09:30 IST (15 minutes after market open).
# The runner is smart enough to handle holidays (it just uses the
# previous trading day's data via yfinance).
#
# Run ONCE from PowerShell (Admin not required for current-user tasks):
#   .\screener\schedule_weekly_dca.ps1
#
# Remove later:
#   Unregister-ScheduledTask -TaskName "AlignerDCAWeekly" -Confirm:$false

$ProjectRoot = "E:\Aligner\Trading"
$Python = "C:\Users\ssura\AppData\Local\Programs\Python\Python312\python.exe"
$TaskName = "AlignerDCAWeekly"

$Action = New-ScheduledTaskAction -Execute $Python `
    -Argument "-m screener.weekly_dca" `
    -WorkingDirectory $ProjectRoot

# Weekly trigger: every Monday at 09:30
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:30"

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $TaskName `
    -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings `
    -Description "Weekly DCA tranche alert (Mon 09:30 IST)"

Write-Host "Registered task: $TaskName"
Write-Host "Next run: next Monday at 09:30 IST"
Write-Host "View it: Task Scheduler (taskschd.msc) > Task Scheduler Library"
Write-Host ""
Write-Host "REMEMBER: Run --init first to create the plan:"
Write-Host "  python -m screener.weekly_dca --init --capital 100000 --tranches 4"
