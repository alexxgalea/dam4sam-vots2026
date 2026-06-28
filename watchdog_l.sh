#!/bin/bash
# Watchdog for dam4sam_l (SAM2.1-Large) — overnight unsupervised operation
# Runs every 5 min; detects stuck progress; restarts if dead.

LOG=/home/alexx/Projects/VOTSTracker/watchdog_l.log
ALERT_LOG=/home/alexx/Projects/VOTSTracker/watchdog_l_alerts.log
RESULTS=/home/alexx/Projects/VOTSTracker/workspace2026/results/dam4sam_l/baseline
EVAL_LOG=/tmp/vot_eval_dam4sam_l.log
VENV=/home/alexx/Projects/VOTSTracker/dam4sam/.venv/bin/activate
DIR=/home/alexx/Projects/VOTSTracker
STATE_DIR=/tmp/dam4sam_l_watchdog
STUCK_FILE="$STATE_DIR/stuck_count"
LAST_DONE_FILE="$STATE_DIR/last_done"

mkdir -p "$STATE_DIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
alert() { echo "[$(ts)] !! $1" | tee -a "$ALERT_LOG" >> "$LOG"; }
info()  { echo "[$(ts)] $1" >> "$LOG"; }

SEQ_DONE=$(ls "$RESULTS" 2>/dev/null | wc -l)
RUNNING=$(pgrep -f "vot evaluate.*dam4sam_l" | head -1)
LAST_DONE=$(cat "$LAST_DONE_FILE" 2>/dev/null || echo 0)
STUCK_COUNT=$(cat "$STUCK_FILE" 2>/dev/null || echo 0)

# ── Completion check ──────────────────────────────────────────────────────────
if [ "$SEQ_DONE" -ge 170 ]; then
    info "dam4sam_l DONE — all 170 sequences complete. Watchdog exiting."
    crontab -l 2>/dev/null | grep -v "watchdog_l.sh" | crontab -
    powershell.exe -Command "shutdown /s /t 300 /c 'dam4sam_l evaluation complete (170/170). Shutdown in 5min. shutdown /a cancels.'" 2>/dev/null
    rm -rf "$STATE_DIR"
    exit 0
fi

# ── Progress tracking ─────────────────────────────────────────────────────────
if [ "$SEQ_DONE" -gt "$LAST_DONE" ]; then
    echo "$SEQ_DONE" > "$LAST_DONE_FILE"
    echo 0 > "$STUCK_FILE"
    STUCK_COUNT=0
else
    STUCK_COUNT=$((STUCK_COUNT + 1))
    echo "$STUCK_COUNT" > "$STUCK_FILE"
fi

# ── Process management ────────────────────────────────────────────────────────
if [ -n "$RUNNING" ]; then
    PCT=$(tail -5 "$EVAL_LOG" 2>/dev/null | grep '%' | tail -1 | tr -d ' ')
    info "OK — PID $RUNNING | $SEQ_DONE/170 | $PCT | stuck=$STUCK_COUNT cycles"
else
    if [ "$STUCK_COUNT" -ge 6 ]; then
        alert "STUCK on sequence ${SEQ_DONE} — $STUCK_COUNT cycles no progress, killing -9 + clearing GPU"
        pkill -9 -f "vot evaluate.*dam4sam_l" 2>/dev/null
        pkill -9 -f "vot_wrapper_large" 2>/dev/null
        pkill -9 -f "dam4sam/.venv.*python" 2>/dev/null
        sleep 5
    fi

    info "DEAD — restarting eval ($SEQ_DONE/170, stuck=$STUCK_COUNT)"
    cd "$DIR"
    source "$VENV"
    nohup vot evaluate --workspace workspace2026 dam4sam_l >> "$EVAL_LOG" 2>&1 &
    NEW_PID=$!
    info "Restarted with PID $NEW_PID"
fi
