#Requires -Version 5.1
<#
.SYNOPSIS
    Auto-accepts incoming AnyDesk connections from MacBook ID 315229108.

.NOTES
    Run on Windows. Launch from WSL2 with: ./start_anydesk_autoaccept.sh
#>

# CONFIG
$TRUSTED_ID     = "315229108"
$CHECK_INTERVAL = 600
$LOG_FILE       = "$env:TEMP\anydesk_autoaccept.log"

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
"@

function Write-Log {
    param([string]$Msg, [string]$Level = "INFO")
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts][$Level] $Msg"
    $color = "Cyan"
    if ($Level -eq "OK")   { $color = "Green"  }
    if ($Level -eq "WARN") { $color = "Yellow" }
    if ($Level -eq "ERR")  { $color = "Red"    }
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG_FILE -Value $line -ErrorAction SilentlyContinue
}

# Find AnyDesk.exe path for CLI calls
function Get-AnyDeskExe {
    $paths = @(
        "$env:ProgramFiles\AnyDesk\AnyDesk.exe",
        "$env:LOCALAPPDATA\AnyDesk\AnyDesk.exe",
        "C:\Program Files (x86)\AnyDesk\AnyDesk.exe"
    )
    foreach ($p in $paths) { if (Test-Path $p) { return $p } }
    # Fallback: find by process
    $proc = Get-Process -Name "AnyDesk" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc) { return $proc.Path }
    return $null
}

function Get-AnyDeskLogPath {
    $candidates = @(
        "$env:APPDATA\AnyDesk\ad_svc.trace",
        "$env:APPDATA\AnyDesk\connection_trace.txt",
        "$env:APPDATA\AnyDesk\ad.trace"
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

# -----------------------------------------------------------------------
# Try to accept by clicking the Accept/Allow button via UI Automation.
# $SkipTitleCheck = $true  -> search all AnyDesk windows (used after log hit)
# $SkipTitleCheck = $false -> only windows whose title looks like a dialog
# -----------------------------------------------------------------------
function Try-AcceptViaUI {
    param([bool]$SkipTitleCheck = $false)

    $procs = Get-Process -Name "AnyDesk" -ErrorAction SilentlyContinue
    if (-not $procs) { return $false }

    $btnNames = @("Accept", "Allow", "Akzeptieren", "Aceptar", "Accepter", "Accetta")

    foreach ($proc in $procs) {
        # Enumerate ALL automation elements owned by this PID
        try {
            $pidCond = New-Object System.Windows.Automation.PropertyCondition(
                [System.Windows.Automation.AutomationElement]::ProcessIdProperty,
                [int]$proc.Id)
            $desktop  = [System.Windows.Automation.AutomationElement]::RootElement
            $allWins  = $desktop.FindAll(
                [System.Windows.Automation.TreeScope]::Children, $pidCond)
        } catch {
            # Fallback: just use MainWindowHandle
            if ($proc.MainWindowHandle -eq [IntPtr]::Zero) { continue }
            try {
                $elem = [System.Windows.Automation.AutomationElement]::FromHandle(
                    $proc.MainWindowHandle)
                $allWins = @($elem)
            } catch { continue }
        }

        foreach ($win in $allWins) {
            try {
                $title = $win.Current.Name
            } catch { continue }

            if (-not $SkipTitleCheck) {
                # Only check windows that look like an incoming request dialog
                $looksLikeDialog = $title -match "(?i)(incoming|session request|verbindung|accept|allow)"
                if (-not $looksLikeDialog) { continue }
            }

            # Look for Accept / Allow button inside this window
            foreach ($name in $btnNames) {
                try {
                    $cond = New-Object System.Windows.Automation.PropertyCondition(
                        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
                    $btn = $win.FindFirst(
                        [System.Windows.Automation.TreeScope]::Descendants, $cond)
                    if ($btn) {
                        $pattern = $btn.GetCurrentPattern(
                            [System.Windows.Automation.InvokePattern]::Pattern)
                        $pattern.Invoke()
                        Write-Log "Clicked '$name' in window '$title' - accepted!" "OK"
                        return $true
                    }
                } catch {}
            }

            # If we found a window but no named button, try bringing it to front
            # and pressing Enter (the default/highlighted button is usually Accept)
            if ($SkipTitleCheck -or ($title -match "(?i)(incoming|session|accept)")) {
                try {
                    $hwnd = [IntPtr]$win.Current.NativeWindowHandle
                    if ($hwnd -ne [IntPtr]::Zero) {
                        [Win32]::ShowWindow($hwnd, 9) | Out-Null   # SW_RESTORE
                        [Win32]::SetForegroundWindow($hwnd) | Out-Null
                        Start-Sleep -Milliseconds 200
                        [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
                        Write-Log "Sent ENTER to window '$title' (fallback)" "OK"
                        return $true
                    }
                } catch {}
            }
        }
    }
    return $false
}

# -----------------------------------------------------------------------
# Try accepting via AnyDesk CLI (some builds support --accept)
# -----------------------------------------------------------------------
function Try-AcceptViaCLI {
    $exe = Get-AnyDeskExe
    if (-not $exe) { return $false }
    try {
        $r = Start-Process -FilePath $exe -ArgumentList "--accept" `
             -PassThru -WindowStyle Hidden -Wait -ErrorAction Stop
        if ($r.ExitCode -eq 0) {
            Write-Log "CLI --accept succeeded" "OK"
            return $true
        }
    } catch {}
    return $false
}

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
Write-Log "AnyDesk Auto-Accept started"
Write-Log "Trusted MacBook ID : $TRUSTED_ID"
Write-Log "Log file           : $LOG_FILE"
Write-Log "Press Ctrl+C to stop."
Write-Host ""

# Ensure AnyDesk is running
$running = Get-Process -Name "AnyDesk" -ErrorAction SilentlyContinue
if (-not $running) {
    Write-Log "AnyDesk not running - attempting to start it..." "WARN"
    $exe = Get-AnyDeskExe
    if ($exe) {
        Start-Process $exe
        Start-Sleep -Seconds 2
        Write-Log "Launched: $exe" "OK"
    } else {
        Write-Log "Could not find AnyDesk.exe - install AnyDesk first." "ERR"
    }
}

# Open log file for tailing
$logPath   = Get-AnyDeskLogPath
$logReader = $null
if ($logPath) {
    try {
        $stream = [System.IO.File]::Open(
            $logPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite)
        $logReader = New-Object System.IO.StreamReader($stream)
        $stream.Seek(0, [System.IO.SeekOrigin]::End) | Out-Null
        Write-Log "Tailing: $logPath"
    } catch {
        Write-Log "Could not open log: $_" "WARN"
    }
} else {
    Write-Log "AnyDesk log not found yet (will retry)" "WARN"
}

$lastAcceptTime = [datetime]::MinValue

function Do-Accept {
    param([string]$Reason)
    Write-Log "Accepting connection ($Reason)..." "OK"

    # Give AnyDesk 400 ms to draw the dialog before we search
    Start-Sleep -Milliseconds 400

    # Attempt 1: UI automation (no title filter - any AnyDesk window)
    if (Try-AcceptViaUI -SkipTitleCheck $true) { return }
    Start-Sleep -Milliseconds 300

    # Attempt 2: UI automation again (dialog may have appeared now)
    if (Try-AcceptViaUI -SkipTitleCheck $true) { return }

    # Attempt 3: CLI --accept
    if (Try-AcceptViaCLI) { return }

    Write-Log "Could not find Accept button - dialog may not be visible yet." "WARN"
    Write-Log "Make sure AnyDesk window is not minimized to tray." "WARN"
}

try {
    while ($true) {
        $now      = [datetime]::Now
        $cooldown = ($now - $lastAcceptTime).TotalSeconds -gt 8

        # --- Primary: background UI scan (title-filtered, lightweight) ---
        if ($cooldown -and (Try-AcceptViaUI -SkipTitleCheck $false)) {
            $lastAcceptTime = $now
        }

        # --- Secondary: log-file tail ---
        if (-not $logReader) {
            # Retry finding log (may appear after AnyDesk fully starts)
            $logPath = Get-AnyDeskLogPath
            if ($logPath) {
                try {
                    $stream = [System.IO.File]::Open(
                        $logPath,
                        [System.IO.FileMode]::Open,
                        [System.IO.FileAccess]::Read,
                        [System.IO.FileShare]::ReadWrite)
                    $logReader = New-Object System.IO.StreamReader($stream)
                    $stream.Seek(0, [System.IO.SeekOrigin]::End) | Out-Null
                    Write-Log "Tailing: $logPath"
                } catch {}
            }
        }

        if ($logReader) {
            while (-not $logReader.EndOfStream) {
                $line = $logReader.ReadLine()
                if ($line -match $TRUSTED_ID -and
                    $line -match "(?i)(request|incoming|session|connect)") {
                    Write-Log "LOG: $($line.Trim())" "OK"
                    if ($cooldown) {
                        Do-Accept -Reason "log trigger"
                        $lastAcceptTime = [datetime]::Now
                    }
                }
            }
        }

        Start-Sleep -Milliseconds $CHECK_INTERVAL
    }
} finally {
    if ($logReader) { $logReader.Dispose() }
    Write-Log "Auto-accept stopped."
}
