#!/bin/bash
# SUPERVIZOR OVERNIGHT v2 — autonom, detasat, independent de VSCode/Claude/permisiuni.
# Supervizeaza CONCURENT (paralel):
#   A) Calibrarea VOST-val (baseline->v3->v6 -> vot analysis). Restart la moarte.
#   B) v3_170 (predictii complete pe 170 = pachet de submisie pt v3). Restart la moarte.
# Cand calibrarea termina -> salveaza Q real in calib_result.txt.
# Flag-uri/log-uri in directorul proiectului (persista chiar daca /tmp e curatat).
PROJ=/home/alexx/Projects/VOTSTracker
cd $PROJ
LOG=$PROJ/overnight.log
PY=$PROJ/dam4sam/.venv/bin/python3

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
calib_running(){ pgrep -f "run_vostval_calib.sh|vot evaluate.*workspace_vost_val|vot analysis.*workspace_vost_val" >/dev/null; }
v3170_running(){ pgrep -f "run_eval_170.py.*v3" >/dev/null; }
v3170_done(){ [ "$(find workspace2026/results/dam4sam_v3/baseline -maxdepth 1 -type d 2>/dev/null | grep -v 'baseline$' | wc -l)" -ge 170 ]; }
calib_done(){ [ -f /tmp/calib_done.flag ] || [ -f $PROJ/.calib_done.flag ]; }

log "=== SUPERVISOR v2 START (PID $$, SID $(ps -o sid= -p $$ | tr -d ' ')) ==="
CR=0; VR=0; QSAVED=0

while true; do
    # --- A) Calibrare ---
    if calib_done; then
        cp -f /tmp/calib_done.flag $PROJ/.calib_done.flag 2>/dev/null
        if [ $QSAVED -eq 0 ]; then
            log "CALIB GATA -> salvez Q real in calib_result.txt"
            $PY parse_vostval_analysis.py workspace_vost_val > $PROJ/calib_result.txt 2>>$LOG
            QSAVED=1
            log "Q real salvat."
        fi
    elif ! calib_running; then
        if [ $CR -lt 8 ]; then
            CR=$((CR+1)); log "CALIB moarta -> repornesc DETASAT (try $CR)"
            setsid bash run_vostval_calib.sh > /tmp/run_vostval_calib.log 2>&1 < /dev/null &
            sleep 25
        fi
    fi

    # --- B) v3_170 ---
    if v3170_done; then
        : # gata, nimic
    elif ! v3170_running; then
        if [ $VR -lt 15 ]; then
            VR=$((VR+1)); log "v3_170 mort -> pornesc DETASAT (try $VR)"
            setsid $PY run_eval_170.py --tracker v3 --out dam4sam_v3 \
                --max-catastrophes 5 --catastrophe-thr 0.25 >> /tmp/run_v3_170.log 2>&1 < /dev/null &
            sleep 25
        fi
    fi

    # --- stare + conditie de iesire ---
    CB=$(find workspace_vost_val/results/dam4sam_b/baseline  -name "*_001.bin" 2>/dev/null | wc -l)
    C3=$(find workspace_vost_val/results/dam4sam_v3/baseline -name "*_001.bin" 2>/dev/null | wc -l)
    C6=$(find workspace_vost_val/results/dam4sam_v6/baseline -name "*_001.bin" 2>/dev/null | wc -l)
    V170=$(find workspace2026/results/dam4sam_v3/baseline -maxdepth 1 -type d 2>/dev/null | grep -v 'baseline$' | wc -l)
    log "STARE: calib[b=$CB v3=$C3 v6=$C6 done=$(calib_done && echo Y || echo n)] | v3_170=$V170/170"

    if calib_done && { v3170_done || [ $VR -ge 15 ]; }; then
        log "=== SUPERVISOR DONE (calib gata + v3_170 gata/epuizat) ==="
        touch $PROJ/.overnight_done.flag
        break
    fi
    sleep 150
done
