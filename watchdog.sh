#!/bin/bash
# Enhanced VOT SAM3 watchdog — robust overnight operation
# Runs every 5 min; detects stuck progress; escalates recovery.

LOG=/home/alexx/Projects/VOTSTracker/watchdog.log
ALERT_LOG=/home/alexx/Projects/VOTSTracker/watchdog_alerts.log
RESULTS=/home/alexx/Projects/VOTSTracker/workspace2026/results/dam4sam_sam3/baseline
EVAL_LOG=/tmp/vot_eval_dam4sam_sam3.log
VENV=/home/alexx/Projects/VOTSTracker/sam3_venv/bin/activate
DIR=/home/alexx/Projects/VOTSTracker
STATE_DIR=/tmp/sam3_watchdog
STUCK_FILE="$STATE_DIR/stuck_count"
LAST_DONE_FILE="$STATE_DIR/last_done"
DRM_FILE=/home/alexx/Projects/VOTSTracker/sam3_upstream/sam3/model/sam3_tracking_predictor.py

mkdir -p "$STATE_DIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
alert() { echo "[$(ts)] !! $1" | tee -a "$ALERT_LOG" >> "$LOG"; }
info()  { echo "[$(ts)] $1" >> "$LOG"; }

SEQ_DONE=$(ls "$RESULTS" 2>/dev/null | wc -l)
RUNNING=$(pgrep -f "vot evaluate.*dam4sam_sam3" | head -1)
LAST_DONE=$(cat "$LAST_DONE_FILE" 2>/dev/null || echo 0)
STUCK_COUNT=$(cat "$STUCK_FILE" 2>/dev/null || echo 0)

# ── Completion check ──────────────────────────────────────────────────────
if [ "$SEQ_DONE" -ge 170 ]; then
    info "SAM3 DONE — all 170 sequences complete. Watchdog exiting."
    crontab -l 2>/dev/null | grep -v "watchdog.sh" | crontab -
    powershell.exe -Command "shutdown /s /t 300 /c 'SAM3 evaluation complete (170/170). Shutdown in 5min. shutdown /a cancels.'" 2>/dev/null
    rm -rf "$STATE_DIR"
    exit 0
fi

# ── Progress tracking ─────────────────────────────────────────────────────
if [ "$SEQ_DONE" -gt "$LAST_DONE" ]; then
    # progress made → reset stuck counter
    echo "$SEQ_DONE" > "$LAST_DONE_FILE"
    echo 0 > "$STUCK_FILE"
    STUCK_COUNT=0
else
    STUCK_COUNT=$((STUCK_COUNT + 1))
    echo "$STUCK_COUNT" > "$STUCK_FILE"
fi

# ── Process management & escalation ───────────────────────────────────────
if [ -n "$RUNNING" ]; then
    PCT=$(tail -5 "$EVAL_LOG" 2>/dev/null | grep '%' | tail -1 | tr -d ' ')
    info "OK — PID $RUNNING | $SEQ_DONE/170 | $PCT | stuck=$STUCK_COUNT cycles"
else
    # Process dead — restart
    if [ "$STUCK_COUNT" -ge 6 ]; then
        # 6+ cycles (30+ min) of no progress + repeated deaths → ESCALATE
        alert "STUCK on sequence ${SEQ_DONE} — $STUCK_COUNT cycles no progress, killing -9 + clearing GPU"
        pkill -9 -f "vot evaluate.*dam4sam_sam3" 2>/dev/null
        pkill -9 -f "vot_wrapper_sam3" 2>/dev/null
        sleep 5
        # Reset PyTorch CUDA cache by killing any orphan python processes
        pkill -9 -f "sam3_venv.*python" 2>/dev/null
        sleep 3
    fi

    # DRM escalation DISABLED — OOM was fixed via CPU offloading; auto-lowering DRM
    # would sabotage tracking quality. Watchdog now only kills+restarts stuck processes.

    info "DEAD — restarting eval ($SEQ_DONE/170, stuck=$STUCK_COUNT)"
    cd "$DIR"
    source "$VENV"
    nohup vot evaluate --workspace workspace2026 dam4sam_sam3 >> "$EVAL_LOG" 2>&1 &
    NEW_PID=$!
    info "Restarted with PID $NEW_PID"
fi
