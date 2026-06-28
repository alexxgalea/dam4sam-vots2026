#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_anydesk_autoaccept.sh
# Launch the AnyDesk auto-accept script from WSL2.
# This opens a visible PowerShell window on Windows that keeps running.
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS_SCRIPT="$SCRIPT_DIR/auto_accept_anydesk.ps1"
WIN_PATH="$(wslpath -w "$PS_SCRIPT")"

echo "════════════════════════════════════════════"
echo "  AnyDesk Auto-Accept Launcher (WSL2→Win)"
echo "════════════════════════════════════════════"
echo "  Script : $PS_SCRIPT"
echo "  Trusted: 315229108 (your MacBook)"
echo ""

if ! command -v powershell.exe &>/dev/null; then
    echo "ERROR: powershell.exe not found in PATH."
    echo "Make sure /mnt/c/Windows/System32/WindowsPowerShell/v1.0 is accessible."
    exit 1
fi

echo "Opening PowerShell window on Windows…"
echo "Keep that window open while you want auto-accept to run."
echo ""

# -NoExit keeps the window open; remove it if you want it to close on finish
powershell.exe -ExecutionPolicy Bypass -NoExit \
    -Command "& { & '$WIN_PATH' }"

# Note: this WSL terminal will block until you close the PowerShell window.
# To run it fully detached in the background, use:
#   cmd.exe /c start powershell.exe -ExecutionPolicy Bypass -File "$WIN_PATH"
