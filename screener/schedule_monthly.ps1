# Register a Windows Task Scheduler entry that runs the monthly screener
# Telegram alert on the 1st business day of every month at 09:20 IST.
#
# Run this ONCE from PowerShell (Run as Administrator):
#   .\screener\schedule_monthly.ps1
#
# To remove later:
#   Unregister-ScheduledTask -TaskName "AlignerScreenerMonthly" -Confirm:$false

$ProjectRoot = "E:\Aligner\Trading"
$Python = "C:\Users\ssura\AppData\Local\Programs\Python\Python312\python.exe"
$TaskName = "AlignerScreenerMonthly"

# Run at 09:20 every month on the 1st (Windows can't natively do "first
# BUSINESS day", but cron @reboot + script-side weekend check works).
$Action = New-ScheduledTaskAction -Execute $Python `
    -Argument "-m screener.monthly_telegram --capital 100000 --tier BLEND" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Monthly -DaysOfMonth 1 -At "09:20"

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries

# Remove existing task if present
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $TaskName `
    -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings `
    -Description "Monthly stock screener picks via Telegram"

Write-Host "Registered task: $TaskName"
Write-Host "Next run on the 1st of next month at 09:20 IST"
Write-Host "View it: Task Scheduler (taskschd.msc)"
