#!/bin/bash
# Supervisor pentru rularea v3g pe toate 170 secventele VOTS2026 (candidat de submisie).
# Rezistent la caderi VSCode/WSL: lansat cu setsid (PID==SID, sesiune independenta).
# run_eval_170.py sare secventele deja facute -> reluabil dupa orice intrerupere.
# Prag VEL_RATIO_THR=0.03 (validat pe VOST-val). Catastrophe-counter dezactivat:
# pe 170 vrem rezultatul COMPLET (e candidatul de submisie), nu early-stop pe proxy.
set -u
PROJ=/home/alexx/Projects/VOTSTracker
PY=$PROJ/dam4sam/.venv/bin/python3
cd $PROJ

LOG=$PROJ/overnight_v3g_170.log

# Lockfile idempotent: refuza a doua instanta de supervisor (previne dublarea -> OOM).
LOCK=/tmp/overnight_v3g_170.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[$(date '+%m-%d %H:%M:%S')] supervisor deja activ (PID $(cat $LOCK)). Ies." >> $LOG
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
RETRY=0
MAX_RETRY=20

v3g_running(){ pgrep -f "run_eval_170.py.*v3g.*dam4sam_v3g" >/dev/null; }
v3g_done(){
  n=$(find $PROJ/workspace2026/results/dam4sam_v3g/baseline -maxdepth 1 -type d 2>/dev/null \
      | grep -v 'baseline$' | wc -l)
  [ "$n" -ge 170 ]
}

echo "[$(date '+%m-%d %H:%M:%S')] SUPERVISOR v3g-170 START" >> $LOG

while true; do
    if v3g_done; then
        echo "[$(date '+%m-%d %H:%M:%S')] v3g-170 COMPLET (170/170). Stop supervisor." >> $LOG
        echo "ALL_DONE" > /tmp/v3g_170_done.flag
        break
    fi

    if ! v3g_running; then
        if [ "$RETRY" -ge "$MAX_RETRY" ]; then
            echo "[$(date '+%m-%d %H:%M:%S')] MAX_RETRY atins ($MAX_RETRY). Abandon." >> $LOG
            break
        fi
        RETRY=$((RETRY+1))
        echo "[$(date '+%m-%d %H:%M:%S')] (re)pornesc v3g-170 (incercarea $RETRY)" >> $LOG
        setsid $PY run_eval_170.py --tracker v3g --out dam4sam_v3g --vel-ratio 0.03 \
            >> $PROJ/run_v3g_170.log 2>&1 < /dev/null &
        sleep 30
    fi

    N=$(find $PROJ/workspace2026/results/dam4sam_v3g/baseline -maxdepth 1 -type d 2>/dev/null \
        | grep -v 'baseline$' | wc -l)
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
    echo "[$(date '+%m-%d %H:%M:%S')] STARE: v3g-170=$N/170 | VRAM=$VRAM | retry=$RETRY" >> $LOG
    sleep 150
done
