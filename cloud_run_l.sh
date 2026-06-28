#!/bin/bash
# Hiera-L baseline run on the cloud instance. Modes:
#   smoke        — 1 short sequence end-to-end + integrity check (mandatory before full)
#   full         — all 170, resumable, with watcher log
#   half1|half2  — alphabetical split for a 2-instance setup
#
# DURABILITY: this script only computes. It does NOT pull results to safety.
# Right after launching `full`/`half*`, start the LOCAL puller on your WSL2 box:
#   nohup bash pull_cloud_results.sh <ssh_port> > pull_l_cloud.log 2>&1 &
# It mirrors results down every few minutes so a GPU fault can't strand the run.
set -euo pipefail
BASE=/home/alexx/Projects/VOTSTracker
cd "$BASE"
PY=dam4sam/.venv/bin/python
OUT=dam4sam_l_cloud
MODE="${1:-full}"

case "$MODE" in
  smoke)
    SEQ=$(for d in workspace2026/sequences/*/; do
            echo "$(ls "$d/color" | wc -l) $(basename "$d")"; done | sort -n | head -1 | cut -d' ' -f2)
    echo "smoke sequence: $SEQ"
    $PY run_eval_170.py --tracker baseline --backbone L --out $OUT --seqs "$SEQ"
    .venv/bin/python check_results_integrity.py --tracker $OUT --allow-pending
    echo "SMOKE OK — masks written and readable. Launch: bash cloud_run_l.sh full"
    ;;
  full|half1|half2)
    ARGS=()
    if [ "$MODE" != "full" ]; then
      ALL=$(ls workspace2026/sequences | sort)
      N=$(echo "$ALL" | wc -l); H=$(( (N+1)/2 ))
      [ "$MODE" = half1 ] && SEL=$(echo "$ALL" | head -$H) || SEL=$(echo "$ALL" | tail -n +$((H+1)))
      ARGS=(--seqs "$(echo "$SEL" | paste -sd,)")
    fi
    nohup $PY run_eval_170.py --tracker baseline --backbone L --out $OUT "${ARGS[@]}" \
      > run_l_$MODE.log 2>&1 &
    PID=$!
    echo "launched PID $PID, log run_l_$MODE.log"
    # watcher: progress every 5 min; on exit, run integrity check (guardrail §6.2)
    nohup bash -c "
      while kill -0 $PID 2>/dev/null; do
        D=\$(ls workspace2026/results/$OUT/baseline 2>/dev/null | wc -l)
        echo \"[\$(date '+%m-%d %H:%M')] L-170 ($MODE): \$D dirs | \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)\"
        sleep 300
      done
      echo '=== RUN ENDED — integrity check ==='
      .venv/bin/python check_results_integrity.py --tracker $OUT --allow-pending
    " > watch_l_$MODE.log 2>&1 &
    echo "watcher log: watch_l_$MODE.log"
    ;;
  *) echo "usage: $0 smoke|full|half1|half2"; exit 1;;
esac
